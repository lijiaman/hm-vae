import torch
import os
import sys
import argparse
import shutil
import numpy as np 

from torch.utils.tensorboard import SummaryWriter

from utils_motion_vae import get_train_loaders_all_data_seq
from utils_common import get_config, make_result_folders
from utils_common import write_loss, show3Dpose_animation

from trainer_motion_vae import Trainer

import torch.backends.cudnn as cudnn
# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='./',
                    help="outputs path")
parser.add_argument('--test_batch_size',
                    type=int,
                    default=10)
parser.add_argument('--multigpus',
                    action="store_true")
parser.add_argument("--resume",
                    action="store_true")
parser.add_argument('--test_model',
                    type=str,
                    default='',
                    help="trained model for evaluation")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)
max_iter = config['max_iter']

trainer = Trainer(config)
trainer.cuda()
trainer.model = trainer.model.cuda()
if opts.multigpus:
    ngpus = torch.cuda.device_count()
    config['gpus'] = ngpus
    print("Number of GPUs: %d" % ngpus)
    trainer.model = torch.nn.DataParallel(trainer.model, device_ids=range(ngpus))
else:
    config['gpus'] = 1

data_loaders = get_train_loaders_all_data_seq(config)

train_loader = data_loaders[0]
val_loader = data_loaders[1]
test_loader = data_loaders[2]

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]

output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml'))
train_writer = SummaryWriter(
    os.path.join(output_directory, "logs"))

iterations = trainer.resume(checkpoint_directory,
                            hp=config,
                            multigpus=opts.multigpus) if opts.resume else 0

if opts.test_model:
    trainer.load_ckpt(opts.test_model)

epoch = 0
while True:
    epoch += 1
    train_dataset = train_loader 
    val_dataset = val_loader
    test_dataset = test_loader
    for it, input_data in enumerate(train_dataset):
        loss_all, loss_kl, loss_6d_rec, loss_rot_rec, loss_pose_rec, \
        loss_joint_pos_rec, loss_root_v_rec, loss_linear_v_rec, loss_angular_v_rec = \
        trainer.gen_update(input_data, config, iterations, opts.multigpus)
        if it % 50 == 0:
            print('Training: Total loss: %.4f, KL loss: %.8f, Rec 6D loss: %.4f, Rec Rot loss: %.4f, Rec Pose loss: %.4f, \
                Rec joint pos loss: %.4f, Rec root v loss: %.4f, Rec linear v loss: %.4f, Rec angular v loss: %.4f' % \
                (loss_all, loss_kl, loss_6d_rec, loss_rot_rec, loss_pose_rec, loss_joint_pos_rec, \
                loss_root_v_rec, loss_linear_v_rec, loss_angular_v_rec))
        # torch.cuda.synchronize()
        
            
        # Check loss in validation set
        if (iterations + 1) % config['validation_iter'] == 0:
            with torch.no_grad():
                for val_it, val_input_data in enumerate(val_dataset):
                    if val_it >= 50:
                        break;
                    val_loss_all, val_loss_kl, val_loss_6d_rec, val_loss_rot_rec, val_loss_pose_rec, \
                    val_loss_joint_pos_rec, val_loss_root_v, val_loss_linear_v, val_loss_angular_v = trainer.gen_update(val_input_data, \
                                                        config, iterations, opts.multigpus, validation_flag=True)
                    print("*********************************************************************************************")
                    print('Val Total loss: %.4f, Val KL loss: %.8f, Val Rec 6D loss: %.4f, Val Rec Rot loss: %.4f, Val Rec Pose loss: %.4f, \
                    Val Rec joint pos loss: %.4f, Val Rec root v loss: %.4f, Val Rec linear v loss: %.4f, Val Rec angular v loss: %.4f' % \
                        (val_loss_all, val_loss_kl, val_loss_6d_rec, val_loss_rot_rec, val_loss_pose_rec, \
                        val_loss_joint_pos_rec, val_loss_root_v, val_loss_linear_v, val_loss_angular_v))

        # Visulization
        if (iterations + 1) % config['image_save_iter'] == 0:
            with torch.no_grad():
                for test_it, test_input_data in enumerate(test_dataset):
                    if test_it >= opts.test_batch_size:
                        break;
                    
                    # Generate long sequences
                    gt_seq, mean_seq_out_pos, sampled_seq_out_pos, \
                        zero_seq_out_pos \
                        = trainer.gen_seq(test_input_data, config, iterations)
                    # T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3
                    for bs_idx in range(0, 1): # test data loader set bs to 1
                        gt_seq_for_vis = gt_seq[:, bs_idx, :, :][None, :, :, :] # 1 X T X 24 X 3

                        mean_out_for_vis = mean_seq_out_pos[:, bs_idx, :, :] # T X 24 X 3
                        mean_out_for_vis = mean_out_for_vis[None, :, :, :] # 1 X T X 24 X 3

                        cat_gt_mean_rot_seq_for_vis = torch.cat((gt_seq_for_vis, mean_out_for_vis), dim=0)

                        sampled_out_for_vis = sampled_seq_out_pos[:, bs_idx, :, :] # T X 24 X 3
                        sampled_out_for_vis = sampled_out_for_vis[None, :, :, :] # 1 X T X 24 X 3
                        
                        if config['model_name'] == "TrajectoryModel":
                            show3Dpose_animation(cat_gt_mean_rot_seq_for_vis.data.cpu().numpy(), image_directory, \
                                iterations, "mean_seq_rot_6d", test_it, use_amass=True)
                            show3Dpose_animation(sampled_out_for_vis.data.cpu().numpy(), image_directory, \
                                iterations, "sampled_seq_rot_6d", test_it, use_amass=True)
                        else:
                            if config['random_root_rot_flag']:
                                show3Dpose_animation(cat_gt_mean_rot_seq_for_vis.data.cpu().numpy(), image_directory, \
                                    iterations, "mean_seq_rot_6d", test_it, use_amass=False)
                                show3Dpose_animation(sampled_out_for_vis.data.cpu().numpy(), image_directory, \
                                    iterations, "sampled_seq_rot_6d", test_it, use_amass=False)
                            else:
                                show3Dpose_animation(cat_gt_mean_rot_seq_for_vis.data.cpu().numpy(), image_directory, \
                                    iterations, "mean_seq_rot_6d", test_it, use_amass=True)
                                show3Dpose_animation(sampled_out_for_vis.data.cpu().numpy(), image_directory, \
                                    iterations, "sampled_seq_rot_6d", test_it, use_amass=True)
                            
        if (iterations + 1) % config['log_iter'] == 0:
            print("Iteration: %08d/%08d" % (iterations + 1, max_iter))
            write_loss(iterations, trainer, train_writer)

        if (iterations + 1) % config['snapshot_save_iter'] == 0:
            trainer.save(checkpoint_directory, iterations, opts.multigpus)
            print('Saved model at iteration %d' % (iterations + 1))

        iterations += 1
        if iterations >= max_iter:
            print("Finish Training")
            sys.exit(0)
