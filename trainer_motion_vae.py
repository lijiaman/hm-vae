import copy
import os
import math

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import lr_scheduler

from trajectory_pred_model import TrajectoryModel
from seq_two_hier_sa_vae import TwoHierSAVAEModel

from utils_common import write_loss, write_images, write_images_interpolation

class Trainer(nn.Module):
    def __init__(self, cfg):
        super(Trainer, self).__init__()
       
        if cfg['model_name'] == "TrajectoryModel":
            self.model = TrajectoryModel(cfg)
        elif cfg['model_name'] == "TwoHierSAVAEModel":
            self.model = TwoHierSAVAEModel(cfg)
            
        lr_gen = cfg['lr']
        gen_params = list(self.model.parameters()) 
        
        self.cfg = cfg

        self.gen_opt = torch.optim.Adam(
            [p for p in gen_params if p.requires_grad],
            lr=lr_gen, weight_decay=cfg['weight_decay'])

        self.gen_scheduler = get_scheduler(self.gen_opt, cfg)

        self.apply(weights_init(cfg['init']))

    def gen_update(self, data, hp, iterations, multigpus, validation_flag=False):
        if validation_flag:
            if self.cfg['model_name'] == "TwoHierSAVAEModel":
                al, kl, rec_6d, rec_rot, rec_pose, rec_joint_pos, rec_root_v, rec_linear_v, rec_angular_v, kl_list = \
                    self.model(data, hp, iterations, multigpus=multigpus, validation_flag=True)
            else:
                al, kl, rec_6d, rec_rot, rec_pose, rec_joint_pos, rec_root_v, rec_linear_v, rec_angular_v = \
                    self.model(data, hp, iterations, multigpus=multigpus, validation_flag=True)

            self.loss_val_total = torch.mean(al)
            self.loss_val_kl = torch.mean(kl)
            self.loss_val_rec_6d = torch.mean(rec_6d)
            self.loss_val_rec_rot = torch.mean(rec_rot)
            self.loss_val_rec_pose = torch.mean(rec_pose)
            self.loss_val_rec_joint_pos = torch.mean(rec_joint_pos)
            self.loss_val_rec_root_v = torch.mean(rec_root_v)
            self.loss_val_rec_linear_v = torch.mean(rec_linear_v)
            self.loss_val_rec_angular_v = torch.mean(rec_angular_v)

            if self.cfg['model_name'] == "TwoHierSAVAEModel":
                self.loss_hier_kl_1 = torch.mean(kl_list[0])
                self.loss_hier_kl_2 = torch.mean(kl_list[1])
                self.loss_hier_kl_3 = torch.mean(kl_list[2])
                self.loss_hier_kl_4 = torch.mean(kl_list[3])

            return self.loss_val_total.item(), self.loss_val_kl.item(),  \
            self.loss_val_rec_6d.item(), self.loss_val_rec_rot.item(), self.loss_val_rec_pose.item(), \
            self.loss_val_rec_joint_pos.item(), self.loss_val_rec_root_v.item(), \
            self.loss_val_rec_linear_v.item(), self.loss_val_rec_angular_v.item()
        else:
            self.gen_opt.zero_grad()
            
            if self.cfg['model_name'] == "TwoHierSAVAEModel":
                al, kl, rec_6d, rec_rot, rec_pose, rec_joint_pos, rec_root_v, rec_linear_v, rec_angular_v, kl_list = \
                    self.model(data, hp, iterations, multigpus=multigpus)
            else:
                al, kl, rec_6d, rec_rot, rec_pose, rec_joint_pos, rec_root_v, rec_linear_v, rec_angular_v = \
                    self.model(data, hp, iterations, multigpus=multigpus)

            self.loss_total = torch.mean(al)
            self.loss_kl = torch.mean(kl)
            self.loss_rec_6d = torch.mean(rec_6d)
            self.loss_rec_rot = torch.mean(rec_rot)
            self.loss_rec_pose = torch.mean(rec_pose)
            self.loss_rec_joint_pos = torch.mean(rec_joint_pos)
            self.loss_rec_root_v = torch.mean(rec_root_v)
            self.loss_rec_linear_v = torch.mean(rec_linear_v)
            self.loss_rec_angular_v = torch.mean(rec_angular_v)

            if self.cfg['model_name'] == "TwoHierSAVAEModel":
                self.loss_hier_kl_1 = torch.mean(kl_list[0])
                self.loss_hier_kl_2 = torch.mean(kl_list[1])
                self.loss_hier_kl_3 = torch.mean(kl_list[2])
                self.loss_hier_kl_4 = torch.mean(kl_list[3])

            self.gen_opt.step()
            self.gen_scheduler.step()

            return self.loss_total.item(), self.loss_kl.item(), \
            self.loss_rec_6d.item(), self.loss_rec_rot.item(), self.loss_rec_pose.item(), \
            self.loss_rec_joint_pos.item(), self.loss_rec_root_v.item(), \
            self.loss_rec_linear_v.item(), self.loss_rec_angular_v.item()

    def resume(self, checkpoint_dir, hp, multigpus):
        this_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        last_model_name = get_model_list(checkpoint_dir, "gen")
        state_dict = torch.load(last_model_name)
        
        model_dict = this_model.state_dict()
        pretrained_dict = state_dict['state_dict']
        model_dict.update(pretrained_dict)
        this_model.load_state_dict(model_dict)
        
        iterations = int(last_model_name[-11:-3])

        state_dict = torch.load(os.path.join(checkpoint_dir, 'optimizer.pt'))
        self.gen_opt.load_state_dict(state_dict['gen'])
        self.gen_scheduler = get_scheduler(self.gen_opt, hp, iterations)

        print('Resume from iteration %d' % iterations)
        return iterations

    def save(self, snapshot_dir, iterations, multigpus):
        # Save generators, discriminators, and optimizers
        this_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        gen_name = os.path.join(snapshot_dir, 'gen_%08d.pt' % (iterations + 1))
        opt_name = os.path.join(snapshot_dir, 'optimizer.pt')
        save_dict = {}
        save_dict['state_dict'] = this_model.state_dict()
        torch.save(save_dict, gen_name, _use_new_zipfile_serialization=False)
        torch.save({'gen': self.gen_opt.state_dict()}, opt_name, _use_new_zipfile_serialization=False)

    def load_ckpt(self, ckpt_name):
        this_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        state_dict = torch.load(ckpt_name)
        model_dict = this_model.state_dict()
        pretrained_dict = state_dict['state_dict']
        model_dict.update(pretrained_dict)    
        this_model.load_state_dict(model_dict)
       
    def test(self, data, hp, iterations, multigpus):
        this_model = self.model.module if multigpus else self.model
        return this_model.test(data, hp, iterations)

    def gen_seq(self, data, hp, iterations):
        return self.model.gen_seq(data, hp, iterations)

    def forward(self, *inputs):
        print('Forward function not implemented.')
        pass
    
    def test_latent_vector_w_motion_input(self, dest_image_directory, trajectory_trainer=None): 
        self.model.test_latent_vector_w_motion_input(dest_image_directory, trajectory_trainer)

    def check_hier_latent_space(self, dest_directory):
        self.model.check_hier_latent_space(dest_directory)

    def check_latent_space_sampling_w_trajectory(self, dest_directory, trajectory_trainer=None):
        self.model.check_latent_space_sampling_w_trajectory(dest_directory, trajectory_trainer)

    def sample_single_seq_w_trajectory(self, trajectory_trainer, dest_image_directory):
        self.model.sample_single_seq_w_trajectory(trajectory_trainer, dest_image_directory)

    def test_model_rec(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False):
        self.model.test_model_rec(dest_image_directory, input_gt, gen_vis, use_amass_data)

    def test_model_rec_for_random_comb_motion(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False):
        self.model.test_model_rec_for_random_comb_motion(dest_image_directory, input_gt, gen_vis, use_amass_data)

    # For 3DPW dataset, generate both direct decoded res and optimized res 
    def for_cropped_3dpw_multiple_opt_batch_complete_seq_partial_input_w_gt_target(self, dest_image_directory, input_gt=True, gen_vis=False, \
        use_amass_data=False, trajectory_trainer=None): 
        self.model.for_cropped_3dpw_multiple_opt_batch_complete_seq_partial_input_w_gt_target(dest_image_directory, \
            input_gt, gen_vis, use_amass_data, trajectory_trainer)

    # Generate long sequence by optimizing whole sequence together with temporal consistency loss, and random z vector for regularization loss
    def long_seq_generation(self, dest_image_directory, input_gt=True, gen_vis=False): 
        self.model.long_seq_generation(dest_image_directory, input_gt, gen_vis)
    
    # Interpolate beween two given long motion sequences
    def interpolate_long_seq(self, dest_image_directory, input_gt=True, gen_vis=False):
        self.model.interpolate_long_seq(dest_image_directory, input_gt, gen_vis)

    # Generae long sequence around a GT latent vector
    def condition_long_seq_generation(self, dest_image_directory, input_gt=True, gen_vis=False):
        self.model.condition_long_seq_generation(dest_image_directory, input_gt, gen_vis)

    # Visualize given saved z vector to motion sequences
    def vis_given_z_vec(self, dest_image_folder):
        self.model.vis_given_z_vec(dest_image_folder)

    # Sample a window-size sequence used for trajectory prediction model testing
    def sample_single_seq(self):
        return self.model.sample_single_seq() 

    # Compare our results with VIBE paper  
    def eval_pose_estimation(self, hp, image_directory):
        self.model.eval_pose_estimation(hp, image_directory)

    # For 3DPW dataset, given upper-body, directly input to model for whole motion
    def batch_complete_seq_partial_input_w_gt_target(self, dest_image_directory, input_gt=True, gen_vis=False):
        self.model.batch_complete_seq_partial_input_w_gt_target(dest_image_directory, input_gt, gen_vis)

    # For AMASS dataset, given upper-body, directly input to model for whole motion
    def batch_complete_seq_amass(self, dest_image_directory, input_gt=True, gen_vis=False):
        self.model.batch_complete_seq_amass(dest_image_directory, input_gt, gen_vis)

    # For AMASS dataset, optimizing all latent vectors of a long sequence at the same time
    def multiple_opt_batch_complete_seq_partial_input_w_gt_target(self, dest_image_directory, input_gt=True, gen_vis=False):
        self.model.multiple_opt_batch_complete_seq_partial_input_w_gt_target(dest_image_directory, input_gt, gen_vis)

    def sampled_seq_test(self, encoder_input):
        return self.model.sampled_seq_test(encoder_input)

    def try_interpolation(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False, trajectory_trainer=None):
        return self.model.try_interpolation(dest_image_directory, input_gt, gen_vis, use_amass_data, trajectory_trainer)

    def try_interpolation_single_window(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False, trajectory_trainer=None):
        return self.model.try_interpolation_single_window(dest_image_directory, input_gt, gen_vis, use_amass_data, trajectory_trainer)

    def final_long_seq_try_interpolation(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False, trajectory_trainer=None):
        return self.model.final_long_seq_try_interpolation(dest_image_directory, input_gt, gen_vis, use_amass_data, trajectory_trainer)

    def final_motion_completion(self, dest_image_directory, input_gt=True, \
        gen_vis=False, use_amass_data=True, trajectory_trainer=None): 
        self.model.final_motion_completion(dest_image_directory, input_gt=input_gt, \
        gen_vis=gen_vis, use_amass_data=use_amass_data, trajectory_trainer=trajectory_trainer)

    def final_motion_completion_single_window(self, dest_image_directory, input_gt=True, \
        gen_vis=False, use_amass_data=True, trajectory_trainer=None, only_eval_visible=False, random_missing_joints=False): 
        self.model.final_motion_completion_single_window(dest_image_directory, input_gt=input_gt, \
        gen_vis=gen_vis, use_amass_data=use_amass_data, trajectory_trainer=trajectory_trainer, \
        only_eval_visible=only_eval_visible, random_missing_joints=random_missing_joints)

    def final_motion_completion_long_seq(self, dest_image_directory, input_gt=True, \
        gen_vis=False, use_amass_data=True, trajectory_trainer=None): 
        self.model.final_motion_completion_long_seq(dest_image_directory, input_gt=input_gt, \
        gen_vis=gen_vis, use_amass_data=use_amass_data, trajectory_trainer=trajectory_trainer)

    def try_final_long_seq_generation(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False, trajectory_trainer=None):
        return self.model.try_final_long_seq_generation(dest_image_directory, input_gt, gen_vis, use_amass_data, trajectory_trainer)

def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and
                  key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def get_scheduler(optimizer, hp, it=-1):
    if 'lr_policy' not in hp or hp['lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hp['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hp['step_size'],
                                        gamma=hp['gamma'], last_epoch=it)
    elif hp['lr_policy'] == 'mstep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=hp['step_size'],
                                        gamma=hp['gamma'], last_epoch=it)
    else:
        return NotImplementedError('%s not implemented', hp['lr_policy'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun
