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
parser.add_argument('--out_tag',
                    type=str,
                    default='',
                    help="output tag which indicates evaluated data type")
parser.add_argument("--long_seq_generation",
                    action="store_true")
parser.add_argument("--condition_long_seq_generation",
                    action="store_true")
parser.add_argument("--opt_gen_multiple_motion_completion",
                    action="store_true")
parser.add_argument("--z_vector_test",
                    action="store_true")
parser.add_argument("--check_interpolation",
                    action="store_true")
parser.add_argument("--motion_completion_3dpw",
                    action="store_true")
parser.add_argument("--motion_completion_amass",
                    action="store_true")
parser.add_argument("--motion_completion_amass_w_trajectory",
                    action="store_true")
parser.add_argument("--input_gt",
                    action="store_true")
parser.add_argument("--new_long_seq_generation",
                    action="store_true")
parser.add_argument("--w_init_seed_long_seq_generation",
                    action="store_true")

parser.add_argument("--test_motion_reconstruction_amass",
                    action="store_true")
parser.add_argument("--test_motion_reconstruction_3dpw",
                    action="store_true")
parser.add_argument("--fix_transition",
                    action="store_true")
parser.add_argument("--try_interpolation",
                    action="store_true")

parser.add_argument("--test_motion_reconstruction_random_comb_motion",
                    action="store_true")

parser.add_argument("--try_interpolation_w_trajectory",
                    action="store_true")
parser.add_argument("--try_interpolation_w_trajectory_single_window",
                    action="store_true")
                    
parser.add_argument("--sample_single_window_w_trajectory",
                    action="store_true")

parser.add_argument("--save_z_vectors",
                    action="store_true")

parser.add_argument('--trajectory_config',
                    type=str,
                    default='',
                    help='configuration file for training and testing')
parser.add_argument('--trajectory_test_model',
                    type=str,
                    default='',
                    help="trained model for evaluation")

parser.add_argument("--check_shallow_deep_latent_space",
                    action="store_true")

parser.add_argument("--check_latent_space_w_trajectory",
                    action="store_true")

parser.add_argument("--check_latent_space_w_motion_input",
                    action="store_true")

parser.add_argument("--final_motion_completion",
                    action="store_true")

parser.add_argument("--final_motion_completion_use_matrix_completion",
                    action="store_true")

parser.add_argument("--vibe_add_trajectory",
                    action="store_true")

parser.add_argument("--try_final_long_seq_generation",
                    action="store_true")


parser.add_argument("--final_try_long_seq_interpolation",
                    action="store_true")

parser.add_argument("--final_motion_completion_long_seq",
                    action="store_true")

opts = parser.parse_args()

# Load experiment setting
config = get_config(opts.config)

trainer = Trainer(config)
trainer.cuda()
trainer.model = trainer.model.cuda()

config['gpus'] = 1

# Setup logger and output folders
model_name = os.path.splitext(os.path.basename(opts.config))[0]

if opts.opt_gen_multiple_motion_completion:
    output_directory = os.path.join(opts.output_path + "/eval_opt_gen_multiple_motion_completion", model_name)
elif opts.long_seq_generation:
    output_directory = os.path.join(opts.output_path + "/eval_long_seq_generation", model_name)
elif opts.condition_long_seq_generation:
    output_directory = os.path.join(opts.output_path + "/eval_condition_long_seq_generation", model_name)
elif opts.check_interpolation:
    output_directory = os.path.join(opts.output_path + "/eval_final_submit_supp_long_seq_interpolation", model_name)
elif opts.motion_completion_amass:
    output_directory = os.path.join(opts.output_path + "/eval_motion_completion_amass", model_name)
elif opts.try_interpolation:
    output_directory = os.path.join(opts.output_path + "/eval_try_interpolation", model_name)
elif opts.try_interpolation_w_trajectory:
    output_directory = os.path.join(opts.output_path + "/eval_try_interpolation_w_trajectory", model_name)
elif opts.try_interpolation_w_trajectory_single_window:
    output_directory = os.path.join(opts.output_path + "/eval_final_submit_try_interpolation_w_trajectory_single_window", model_name)
elif opts.motion_completion_amass_w_trajectory:
    output_directory = os.path.join(opts.output_path + "/eval_motion_completion_amass_w_trajectory", model_name)
elif opts.sample_single_window_w_trajectory:
    output_directory = os.path.join(opts.output_path + "/eval_final_single_window_w_trajectory", model_name)
elif opts.final_motion_completion:
    output_directory = os.path.join(opts.output_path + "/for_rebuttal_eval_two_missing_types_final_completion_single_window", model_name)
elif opts.check_latent_space_w_trajectory:
    output_directory = os.path.join(opts.output_path + "/eval_check_latent_space_w_trajectory", model_name)
elif opts.check_latent_space_w_motion_input:
    output_directory = os.path.join(opts.output_path + "/eval_check_latent_space_w_motion_input", model_name)
elif opts.vibe_add_trajectory:
    output_directory = os.path.join(opts.output_path + "/eval_vibe_w_trajectory_not_used", model_name)
elif opts.try_final_long_seq_generation:
    output_directory = os.path.join(opts.output_path + "/eval_try_final_long_seq_generation", model_name)
elif opts.final_try_long_seq_interpolation:
    output_directory = os.path.join(opts.output_path + "/eval_for_submit_try_long_seq_interpolation", model_name)
elif opts.final_motion_completion_long_seq:
    # output_directory = os.path.join(opts.output_path + "/input_vibe_prep_supp_eval_long_seq_two_missing_types_final_completion", model_name)
    output_directory = os.path.join(opts.output_path + "/for_talk_test_final_completion", model_name)

checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml'))

trainer.load_ckpt(opts.test_model)
                            
# trainer.batch_complete_seq_amass(image_directory, input_gt=True, gen_vis=True)
# trainer.batch_complete_seq_partial_input_w_gt_target(image_directory, input_gt=True, gen_vis=True)

# For generating multiple plausible lower-body motions
if opts.opt_gen_multiple_motion_completion:
    trainer.multiple_opt_batch_complete_seq_partial_input_w_gt_target(image_directory, input_gt=True, gen_vis=True)

# For generating long sequence
if opts.long_seq_generation:
    trainer.long_seq_generation(image_directory, input_gt=True, gen_vis=True)

# For checking interpolation results
if opts.check_interpolation:
    trainer.interpolate_long_seq(image_directory, input_gt=True, gen_vis=True)

# For long sequence generation around gt vector
if opts.condition_long_seq_generation:
    trainer.condition_long_seq_generation(image_directory, input_gt=True, gen_vis=True)

# For save z vector of gt data
# trainer.save_z_vector_for_motion(image_directory, input_gt=True, gen_vis=True)

# For k means center visualization
# trainer.vis_given_z_vec(image_directory)

# For testing each latent vector meaning
if opts.z_vector_test:
    trainer.separate_latent_vector_test(image_directory, input_gt=True, gen_vis=True)

# Deprecated:
# For generating sequence in autoregressive way
# trainer.autoregressive_long_seq_generation(image_directory, input_gt=True, gen_vis=True)

# For nn baseline
# trainer.nn_baseline_batch_complete_seq_partial_input_w_gt_target(image_directory, input_gt=True, gen_vis=True)

if opts.motion_completion_3dpw:
    trainer.for_cropped_3dpw_multiple_opt_batch_complete_seq_partial_input_w_gt_target(image_directory, input_gt=opts.input_gt, gen_vis=True) 

if opts.motion_completion_amass:
    trainer.for_cropped_3dpw_multiple_opt_batch_complete_seq_partial_input_w_gt_target(image_directory, input_gt=True, gen_vis=True, use_amass_data=True) 

if opts.test_motion_reconstruction_amass: # Final
    trainer.test_model_rec(image_directory, input_gt=True, gen_vis=False, use_amass_data=True)

if opts.test_motion_reconstruction_3dpw:
    trainer.test_model_rec(image_directory, input_gt=True, gen_vis=False, use_amass_data=False) 

if opts.new_long_seq_generation:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)
    
    trainer.diff_long_seq_gen(image_directory, input_gt=True, gen_vis=True, trajectory_trainer=trajectory_trainer)

if opts.w_init_seed_long_seq_generation:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.diff_long_seq_gen(image_directory, input_gt=True, gen_vis=True, use_seed_seq=True, trajectory_trainer=trajectory_trainer)

if opts.fix_transition:
    trainer.fix_transition(image_directory, input_gt=True, gen_vis=True, use_amass_data=True)

if opts.try_interpolation:
    trainer.try_interpolation(image_directory, input_gt=True, gen_vis=True, use_amass_data=True)

if opts.save_z_vectors:
    trainer.save_z_vector_for_motion(image_directory, input_gt=True, gen_vis=True)

if opts.try_interpolation_w_trajectory:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.try_interpolation(image_directory, input_gt=True, gen_vis=True, use_amass_data=True, trajectory_trainer=trajectory_trainer)


if opts.try_interpolation_w_trajectory_single_window: # Final
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.try_interpolation_single_window(image_directory, input_gt=True, gen_vis=False, use_amass_data=True, trajectory_trainer=trajectory_trainer)


if opts.final_try_long_seq_interpolation: # Final
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.final_long_seq_try_interpolation(image_directory, input_gt=True, gen_vis=True, use_amass_data=True, trajectory_trainer=trajectory_trainer)

if opts.motion_completion_amass_w_trajectory:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.for_cropped_3dpw_multiple_opt_batch_complete_seq_partial_input_w_gt_target(image_directory, input_gt=True, gen_vis=True, use_amass_data=True, \
        trajectory_trainer=trajectory_trainer)

if opts.sample_single_window_w_trajectory: # Final
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.sample_single_seq_w_trajectory(trajectory_trainer, output_directory)

if opts.check_shallow_deep_latent_space: # Final
    trainer.check_hier_latent_space(image_directory)

if opts.test_motion_reconstruction_random_comb_motion:
    trainer.test_model_rec_for_random_comb_motion(image_directory, input_gt=True, gen_vis=True, use_amass_data=True)

if opts.final_motion_completion:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.final_motion_completion_single_window(image_directory, input_gt=True, gen_vis=False, use_amass_data=True, \
        trajectory_trainer=trajectory_trainer, only_eval_visible=False, random_missing_joints=True)

if opts.final_motion_completion_long_seq:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.final_motion_completion_long_seq(image_directory, input_gt=True, gen_vis=True, use_amass_data=False, \
        trajectory_trainer=trajectory_trainer)

if opts.check_latent_space_w_trajectory: # Final
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.check_latent_space_sampling_w_trajectory(image_directory, trajectory_trainer=trajectory_trainer)

if opts.check_latent_space_w_motion_input:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.test_latent_vector_w_motion_input(image_directory, trajectory_trainer=trajectory_trainer)

def compute_rotation_matrix_from_euler(euler):
    # bs X 3
    batch = euler.shape[0]
        
    c1 = torch.cos(euler[:,0]).view(batch,1) #batch*1 
    s1 = torch.sin(euler[:,0]).view(batch,1) #batch*1 
    c2 = torch.cos(euler[:,2]).view(batch,1) #batch*1 
    s2 = torch.sin(euler[:,2]).view(batch,1) #batch*1 
    c3 = torch.cos(euler[:,1]).view(batch,1) #batch*1 
    s3 = torch.sin(euler[:,1]).view(batch,1) #batch*1 
        
    row1=torch.cat((c2*c3,          -s2,    c2*s3         ), 1).view(-1,1,3) #batch*1*3
    row2=torch.cat((c1*s2*c3+s1*s3, c1*c2,  c1*s2*s3-s1*c3), 1).view(-1,1,3) #batch*1*3
    row3=torch.cat((s1*s2*c3-c1*s3, s1*c2,  s1*s2*s3+c1*c3), 1).view(-1,1,3) #batch*1*3
        
    matrix = torch.cat((row1, row2, row3), 1) #batch*3*3
    
    return matrix

def adjust_root_rot(ori_seq_data):
    # ori_seq_data: bs X T X 24 X 3 X 3
    bs, timesteps, _, _, _ = ori_seq_data.size()
    # target_root_rot = torch.eye(3).cuda() # 3 X 3
    target_root_rot = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    target_root_rot = torch.from_numpy(target_root_rot).float()
    target_root_rot = target_root_rot[None, :, :].repeat(bs, 1, 1) # bs X 3 X 3
    
    # ori_root_rot = ori_seq_data[:, 0, 0, :, :] # bs x 3 X 3
    ori_root_rot = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    ori_root_rot = torch.from_numpy(ori_root_rot).float()
    ori_root_rot = ori_root_rot[None, :, :].repeat(bs, 1, 1) # bs X 3 X 3
    relative_rot = torch.matmul(target_root_rot, ori_root_rot.transpose(1, 2)) # bs X 3 X 3

    relative_rot = relative_rot[:, None, :, :].repeat(1, timesteps, 1, 1) # bs X T X 3 X 3
    # print("relative_rot:{0}".format(relative_rot[0,0]))

    converted_seq_data = torch.matmul(relative_rot.view(-1, 3, 3), ori_seq_data[:, :, 0, :, :].view(-1, 3, 3)) # (bs*T) X 3 X 3
    converted_seq_data = converted_seq_data.view(bs, timesteps, 3, 3)

    dest_seq_data = ori_seq_data.clone()
    # print("dest seq:{0}".format(dest_seq_data.size()))
    # print("converted seq data:{0}".format(converted_seq_data.size()))
    dest_seq_data[:, :, 0, :, :] = converted_seq_data

    return dest_seq_data, relative_rot 
    # bs X T X 24 X 3 X 3, bs X T X 3 X 3

if opts.vibe_add_trajectory:
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    data_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/final_iccv_mesh_vis_our_decoded_3dpw_eval_results/final_two_hier_data_aug_len_8_set_7"
    # data_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/eval_new_clean_cmp_human_dynamics_outputs/final_two_hier_model_data_aug_len_8_set_7/images"
    npy_files = os.listdir(data_folder)
    for npy_name in npy_files:
        if "rot" in npy_name and ".npy" in npy_name:
            npy_path = os.path.join(data_folder, npy_name)
            npy_data = np.load(npy_path) # T X 24 X 3 X 3
            rot_mat = torch.from_numpy(npy_data).float()

            # Rotate root
            # euler_angles = torch.from_numpy(np.asarray([[-90.0, 0.0, 0.0]])) # 1 X 3
            # root_rot_inv = compute_rotation_matrix_from_euler(euler_angles) # 1 X 3 X 3
            # random_root_rot = root_rot_inv.float()

            # timesteps = rot_mat.size()[0]
            # ori_root_rot = rot_mat.view(timesteps, -1)[:, :3*3] # T X (3*3)
            # ori_root_rot = ori_root_rot.contiguous().view(-1, 3, 3) # T X 3 X 3 
            # aug_root_rot = torch.matmul(random_root_rot, ori_root_rot) # T X 3 X 3

            # rot_mat[:, 0, :, :] = aug_root_rot

            rot_mat, _ = adjust_root_rot(rot_mat[None]) # 1 X T X 24 X 3 X 3
            # ori_seq_data: bs X T X 24 X 3 X 3
            rot_mat = rot_mat.squeeze(0)

            rotMatrices = rot_mat.cuda()

            # T X 24 X 3 X 3
            # Convert rotation matrix to 6D representation
            cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # T X 24 X 2 X 3
            cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # T X 24 X 6

            encoder_input = cont6DRep[None] # 1 X T X 24 X 6
            pred_trans_res = trajectory_trainer.sampled_seq_test(encoder_input) # T X 1 X 24 X 3

            dest_trans_path = npy_path.replace("rot", "root_trans")
            np.save(dest_trans_path, pred_trans_res[:, 0, :, :].data.cpu().numpy())

            for_vis_seq = pred_trans_res.transpose(0, 1)[:, :300, :, :] # 1 X T X 24 X 3
            # for_vis_seq[:, :, :, -1] = -for_vis_seq[:, :, :, -1]
            # for_vis_seq = pred_trans_res.transpose(0, 1) # 1 X T X 24 X 3
            show3Dpose_animation(for_vis_seq.data.cpu().numpy(), "./test_vibe_w_traj", \
                                str(0), "test", npy_name, use_amass=True)
            # show3Dpose_animation(for_vis_seq.data.cpu().numpy(), data_folder, \
            #                     str(0), "test_dance", npy_name, use_amass=True)

            # break


if opts.try_final_long_seq_generation: # Final
    # Load trajectory model 
    trajectory_config = get_config(opts.trajectory_config)

    trajectory_trainer = Trainer(trajectory_config)
    trajectory_trainer.cuda()
    trajectory_trainer.model = trajectory_trainer.model.cuda()
    trajectory_trainer.load_ckpt(opts.trajectory_test_model)

    trainer.try_final_long_seq_generation(image_directory, input_gt=True, gen_vis=True, use_amass_data=True, \
         trajectory_trainer=trajectory_trainer)
