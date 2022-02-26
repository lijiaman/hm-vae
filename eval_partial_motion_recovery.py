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
parser.add_argument("--input_gt",
                    action="store_true")
parser.add_argument("--try_interpolation_w_trajectory",
                    action="store_true")
parser.add_argument("--try_interpolation_w_trajectory_single_window",
                    action="store_true")
parser.add_argument('--trajectory_config',
                    type=str,
                    default='',
                    help='configuration file for training and testing')
parser.add_argument('--trajectory_test_model',
                    type=str,
                    default='',
                    help="trained model for evaluation")

parser.add_argument("--final_motion_completion",
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

if opts.try_interpolation_w_trajectory_single_window:
    output_directory = os.path.join(opts.output_path + "/eval_interpolation_w_trajectory_single_window", model_name)
elif opts.final_try_long_seq_interpolation:
    output_directory = os.path.join(opts.output_path + "/eval_long_seq_interpolation", model_name)
elif opts.final_motion_completion:
    output_directory = os.path.join(opts.output_path + "/eval_completion_single_window", model_name)
elif opts.final_motion_completion_long_seq:
    output_directory = os.path.join(opts.output_path + "/eval_long_seq_completion", model_name)

checkpoint_directory, image_directory = make_result_folders(output_directory)
shutil.copyfile(opts.config, os.path.join(output_directory, 'config.yaml'))

trainer.load_ckpt(opts.test_model)

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
