import torch
import os
import sys
import argparse
import shutil
import numpy as np 
import imageio
import json

from tensorboardX import SummaryWriter

from utils_motion_vae import get_train_loaders_all_data_seq_for_eval
from utils_common import get_config, make_result_folders
from utils_common import write_loss, show3Dpose_animation, show3Dpose_animation_with_mask

from trainer_motion_vae import Trainer

import torch.backends.cudnn as cudnn

def rot_mat_to_6d(rotMatrices):
    # bs X 24 X 3 X 3
    # Convert rotation matrix to 6D representation
    cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # bs X 24 X 2 X 3
    cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # bs X 24 X 6

    return cont6DRep

# Enable auto-tuner to find the best algorithm to use for your hardware.
cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=str,
                    default='',
                    help='configuration file for training and testing')
parser.add_argument('--trajectory_config',
                    type=str,
                    default='',
                    help='configuration file for training and testing')
parser.add_argument('--output_path',
                    type=str,
                    default='./',
                    help="outputs path")
parser.add_argument('--vis_iters',
                    type=int,
                    default=1)
parser.add_argument('--vis_bs',
                    type=int,
                    default=32)
parser.add_argument('--test_model',
                    type=str,
                    default='',
                    help="trained model for evaluation")
parser.add_argument('--trajectory_test_model',
                    type=str,
                    default='',
                    help="trained model for evaluation")
parser.add_argument('--out_tag',
                    type=str,
                    default='',
                    help="output tag which indicates evaluated data type")
parser.add_argument('--seq_generation_npy_path',
                    type=str,
                    default='',
                    help="npy files for generated motion sequence")
parser.add_argument('--seq_generation_npy_folder',
                    type=str,
                    default='',
                    help="npy files folder for generated motion sequence")
parser.add_argument("--pred_trajectory_for_single_window",
                    action="store_true")
parser.add_argument("--debug_trajectory",
                    action="store_true")
opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)

trainer = Trainer(config)
trainer.cuda()
trainer.model = trainer.model.cuda()

config['gpus'] = 1

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.trajectory_config))[0]

if opts.pred_trajectory_for_single_window:
    output_directory = os.path.join(opts.output_path + "/eval_single_window_trajectory", model_name)
else:
    output_directory = os.path.join(opts.output_path + "/eval_final_long_seq_trajectory", model_name)

checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml'))


# Load trajectory model 
trajectory_config = get_config(opts.trajectory_config)

trajectory_trainer = Trainer(trajectory_config)
trajectory_trainer.cuda()
trajectory_trainer.model = trajectory_trainer.model.cuda()
trajectory_trainer.load_ckpt(opts.trajectory_test_model)


# For sampling from trained VAE model
if opts.pred_trajectory_for_single_window:
    trainer.load_ckpt(opts.test_model)                        
    encoder_input = trainer.sample_single_seq()
    # encoder_input: bs X T X 24 x 6
    pred_seq_res = trajectory_trainer.sampled_seq_test(encoder_input) # T X bs X 24 X 3

    bs = pred_seq_res.size()[1]
    for idx in range(bs):
        curr_seq_for_vis = pred_seq_res[:, idx, :, :][None, :, :, :] # 1 X T X 24 X 3
        show3Dpose_animation(curr_seq_for_vis.data.cpu().numpy(), image_directory, \
                                0, "sampled_short_seq_trajectory", idx, use_amass=True)
        
        # Save npy
        dest_npy_path = os.path.join(image_directory, str(0), "sampled_short_seq_trajectory", str(idx)+"_w_trajectory.npy")
        curr_npy_data = torch.cat((encoder_input[idx, :, :, :], pred_seq_res[:, idx, :, :]), dim=-1).data.cpu().numpy() # T X 24 X 9
        np.save(dest_npy_path, curr_npy_data)

# Load long seqeunce generation npy 
if opts.seq_generation_npy_path:
    npy_data = np.load(opts.seq_generation_npy_path) 
    rot_mat_input = torch.from_numpy(npy_data).float().cuda() # T X 24 X 3 X 3
    encoder_input = rot_mat_to_6d(rot_mat_input)[None] # 1 X T X 24 X 6
    pred_seq_res = trajectory_trainer.sampled_seq_test(encoder_input) # T X bs X 24 X 3

    bs = pred_seq_res.size()[1]
    for idx in range(bs):
        curr_seq_for_vis = pred_seq_res[:, idx, :, :][None, :, :, :] # 1 X T X 24 X 3
        show3Dpose_animation(curr_seq_for_vis.data.cpu().numpy(), image_directory, \
                                0, "sampled_long_seq_trajectory", idx, use_amass=True, dest_vis_path=opts.seq_generation_npy_path.replace(".npy", "_w_trajectory.mp4"))

if opts.seq_generation_npy_folder:
    npy_files = os.listdir(opts.seq_generation_npy_folder)
    for f_name in npy_files:
        if "our.npy" in f_name:
            npy_path = os.path.join(opts.seq_generation_npy_folder, f_name)
            npy_data = np.load(npy_path) 
            rot_mat_input = torch.from_numpy(npy_data).float().cuda() # T X 24 X 3 X 3
            encoder_input = rot_mat_to_6d(rot_mat_input)[None] # 1 X T X 24 X 6
            pred_seq_res = trajectory_trainer.sampled_seq_test(encoder_input) # T X bs X 24 X 3

            bs = pred_seq_res.size()[1]
            for idx in range(bs):
                curr_seq_for_vis = pred_seq_res[:, idx, :, :][None, :, :, :] # 1 X T X 24 X 3
                show3Dpose_animation(curr_seq_for_vis.data.cpu().numpy(), image_directory, \
                                        0, "sampled_long_seq_trajectory", idx, use_amass=False, dest_vis_path=npy_path.replace(".npy", "_w_trajectory.mp4"))

                # Save npy
                dest_npy_path = npy_path.replace(".npy", "w_trajectory.npy")
                curr_npy_data = torch.cat((encoder_input[idx, :, :, :], pred_seq_res[:, idx, :, :]), dim=-1).data.cpu().numpy() # T X 24 X 9
                np.save(dest_npy_path, curr_npy_data)

if opts.debug_trajectory:
    root_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/utils/data"

    data_folder = os.path.join(root_folder, "for_all_data_motion_model")
    rot_npy_folder = os.path.join(root_folder, "processed_all_amass_data")

    test_json = os.path.join(data_folder, "test_all_amass_motion_data.json")
    test_json_data = json.load(open(test_json, 'r'))
    block_size = 150
    seq_cnt = 0
    batch_num = 20

    ori_pose_seq_data_list = []
    for k in test_json_data:
        v_name = test_json_data[k]     
        rot_npy_path = os.path.join(rot_npy_folder, v_name)
        ori_pose_seq_data = np.load(rot_npy_path) # T X n_dim
        if ori_pose_seq_data.shape[0] >= block_size:
            ori_pose_seq_data = torch.from_numpy(ori_pose_seq_data).float()[:block_size, :]
            ori_pose_seq_data_list.append(ori_pose_seq_data)
            seq_cnt += 1

        if seq_cnt >= batch_num:
            break; 
    print("Total seq:{0}".format(seq_cnt))
        
    ori_pose_seq_data_list = torch.stack(ori_pose_seq_data_list).cuda() # K X T X n_dim

    seq_rot_6d = ori_pose_seq_data_list[:, :, :24*6] # K X T X (24*6)
    seq_rot_mat = ori_pose_seq_data_list[:, :, 24*6:24*6+24*3*3] # K X T X (24*3*3), used for loss, no need for normalization
    seq_rot_pos = ori_pose_seq_data_list[:, :, 24*6+24*3*3:24*6+24*3*3+24*3] # K X T X (24*3), used for loss, no need for normalization

    num_seq = seq_rot_mat.size()[0]
    for idx in range(num_seq):
        rot_mat_input = seq_rot_mat[idx].view(-1, 24, 3, 3).cuda() # T X 24 X 3 X 3
        encoder_input = rot_mat_to_6d(rot_mat_input)[None] # 1 X T X 24 X 6
        pred_seq_res = trajectory_trainer.sampled_seq_test(encoder_input) # T X bs X 24 X 3

        dest_folder = "./tmp_debug_trajectory_pred"
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        npy_path = os.path.join(dest_folder, str(idx)+".npy")

        bs = pred_seq_res.size()[1]
        for idx in range(bs):
            curr_seq_for_vis = pred_seq_res[:, idx, :, :][None, :, :, :] # 1 X T X 24 X 3
            show3Dpose_animation(curr_seq_for_vis.data.cpu().numpy(), image_directory, \
                                    0, "sampled_long_seq_trajectory", idx, use_amass=True, dest_vis_path=npy_path.replace(".npy", "_w_trajectory.mp4"))

            # Save npy
            dest_npy_path = npy_path.replace(".npy", "w_trajectory.npy")
            curr_npy_data = torch.cat((encoder_input[idx, :, :, :], pred_seq_res[:, idx, :, :]), dim=-1).data.cpu().numpy() # T X 24 X 9
            np.save(dest_npy_path, curr_npy_data)
