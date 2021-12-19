import copy
import numpy as np
import json 
import os 
import pickle as pkl 
import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

from torch.optim import lr_scheduler

import torchgeometry as tgm

import my_tools
from fk_layer import ForwardKinematicsLayer
from skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear, get_edges

from torch.distributions import Categorical
import torch.distributions.multivariate_normal as dist_mn

from utils_common import show3Dpose_animation, show3Dpose_animation_multiple

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

class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
        # if args.rotation == 'euler_angle': self.channel_base = [3]
        # elif args.rotation == 'quaternion': self.channel_base = [4]
        if args['trajectory_input_joint_pos']:
            self.channel_base = [3] # 6 + 3
        else:
            self.channel_base = [6] 
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        self.convs = []

        kernel_size = args['kernel_size']
        padding = (kernel_size - 1) // 2
        bias = True

        for i in range(args['num_layers']):
            self.channel_base.append(self.channel_base[-1] * 2) # 6, 12, 24, 48, 96

        for i in range(args['num_layers']):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args['skeleton_dist'])
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i+1] * self.edge_num[i]
            # print("edge num:{0}".format(self.edge_num[i]))
            # print("in channels:{0}".format(in_channels))
            # print("out channels:{0}".format(out_channels))
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)

            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
            # print("i:{0}".format(i))
            # print("neighbor list:{0}".format(neighbor_list))
            
            curr_stride = 1
        
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=self.edge_num[i], kernel_size=kernel_size, stride=curr_stride,
                                    padding=padding, padding_mode=args['padding_mode'], bias=bias, add_offset=False,
                                    in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
            self.convs.append(seq[-1])
            last_pool = True if i == args['num_layers'] - 1 else False
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args['skeleton_pool'],
                                channels_per_edge=out_channels // len(neighbor_list), last_pool=last_pool)
            seq.append(pool)
            seq.append(nn.LeakyReLU(negative_slope=0.2))
            self.layers.append(nn.Sequential(*seq))

            # self.topologies.append(topology)
            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            # print("topology:{0}".format(self.topologies))
            # print("pool list:{0}".format(self.pooling_list))
            self.edge_num.append(len(self.topologies[-1]))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            # print("layer input:{0}".format(input.size()))
            input = layer(input)
            # print("layer output:{0}".format(input.size()))
            # import pdb 
            # pdb.set_trace()
        return input

class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.args = args
        self.enc = enc
        self.convs = []

        kernel_size = args['kernel_size']
        padding = (kernel_size - 1) // 2

        for i in range(args['num_layers']):
            seq = []
            in_channels = enc.channel_list[args['num_layers'] - i]
            out_channels = in_channels // 2
            neighbor_list = find_neighbor(enc.topologies[args['num_layers'] - i - 1], args['skeleton_dist'])

            if i != 0 and i != args['num_layers'] - 1:
                bias = False
            else:
                bias = True

            self.unpools.append(SkeletonUnpool(enc.pooling_list[args['num_layers'] - i - 1], in_channels // len(neighbor_list)))

            seq.append(nn.Upsample(scale_factor=2, mode=args['upsampling'], align_corners=False))
            seq.append(self.unpools[-1])
            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=enc.edge_num[args['num_layers'] - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
            if i % 2 == 0:
                curr_stride = 1
            else:
                curr_stride = 2
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[args['num_layers'] - i - 1], kernel_size=kernel_size, stride=curr_stride,
                                    padding=padding, padding_mode=args['padding_mode'], bias=bias, add_offset=False,
                                    in_offset_channel=3 * enc.channel_base[args['num_layers'] - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != args['num_layers'] - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, input, offset=None):
        for i, layer in enumerate(self.layers):
            # print("Decoder i:{0}".format(i))
            # print("Decoder layer:{0}".format(layer))
            # print("Decoder layer input:{0}".format(input.size()))
            input = layer(input)
            # print("Decoder layer output:{0}".format(input.size()))
            # import pdb
            # pdb.set_trace()

        return input

class TrajectoryModel(nn.Module):
    def __init__(self, hp):
        super(TrajectoryModel, self).__init__()

        self.latent_d = hp['latent_d']
        self.n_joints = hp['n_joints']
        self.input_dim = hp['input_dim'] 
        self.output_dim = hp['output_dim']
        self.max_timesteps = hp['train_seq_len']

        parent_json = "./utils/data/joint24_parents.json"
        edges = get_edges(parent_json) # a list with 23(n_joints-1) elements, each elements represents a pair (parent_idx, idx)

        self.fk_layer = ForwardKinematicsLayer()

        self.hp = hp 

        self.enc = Encoder(hp, edges)
         
        self.d_model = self.enc.channel_base[-1]
      
        # self.fc_mapping = nn.Linear(self.d_model, 3)
        self.fc_mapping = nn.Linear(self.d_model*7, 3)

        mean_std_npy = "./utils/data/for_all_data_motion_model/all_amass_data_mean_std.npy"
              
        mean_std_data = np.load(mean_std_npy) # 2 X n_dim
        mean_std_data[1, mean_std_data[1, :]==0] = 1.0

        self.mean_vals = torch.from_numpy(mean_std_data[0, :]).float()[None, :].cuda() # 1 X n_dim
        self.std_vals = torch.from_numpy(mean_std_data[1, :]).float()[None, :].cuda() # 1 X n_dim

    def forward(self, data, hp, iterations, multigpus=False, validation_flag=False):
        offset = None 
        # input = torch.zeros(8, 24*6, 30).cuda() # For debug
        seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v = data
        seq_rot_6d = seq_rot_6d.float().cuda() # bs X T X (24*6), not normalized
        seq_rot_mat = seq_rot_mat.float().cuda() # bs X T X (24*3*3), not normalized
        seq_rot_pos = seq_rot_pos.float().cuda() # bs X T X (24*3), not normalized
        seq_joint_pos = seq_joint_pos.float().cuda() # bs X T X (24*3), normalized 
        bs, timesteps, _ = seq_rot_6d.size()
        # seq_rot_pos = self.fk_layer(seq_rot_mat.view(bs*timesteps, self.n_joints, 3, 3)).detach() # (bs*T) X 24 X 3
        # seq_rot_pos = seq_rot_pos.view(bs, timesteps, self.n_joints, 3) # bs X T X 24 X 3
        # seq_rot_pos = seq_rot_pos.contiguous().view(bs, timesteps, -1) # bs X T X (24*3)
        seq_root_v = seq_root_v.float().cuda() # bs X T X 3, normalized

        if hp['trajectory_input_joint_pos']:
            encoder_input = seq_joint_pos.view(bs, timesteps, self.n_joints, -1) # bs X T X 24 X 3
        else:
            encoder_input = seq_rot_6d.view(bs, timesteps, self.n_joints, -1) # bs X T X 24 X 6
       
        input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*6)
        input = input.transpose(1, 2) # bs X (24*6) X T
        latent = self.enc(input, offset) # input: bs X (n_edges*input_dim) X T
        # latent: bs X (n_edges*d) X T
     
        k_edges = latent.shape[1] // self.d_model
        encoder_map_input = latent.view(bs, k_edges, self.d_model, timesteps) # bs X k_edges X d X T
        encoder_map_input = encoder_map_input.transpose(2, 3).transpose(1, 2) # bs X T X k_edges X d
        # root_v_out = self.fc_mapping(encoder_map_input) # bs X T X k_edges X 3
        # root_v_out = root_v_out.mean(dim=2) # bs X T X 3
        root_v_out = self.fc_mapping(encoder_map_input.view(bs, timesteps, -1)) # bs X T X 3

        if hp['use_accumulation_root_v']:
            pred_trajectory = self.gen_motion_w_trajectory(seq_rot_pos.view(bs, timesteps, 24, 3).transpose(0, 1), root_v_out.transpose(0, 1))
            gt_trajectory = self.gen_motion_w_trajectory(seq_rot_pos.view(bs, timesteps, 24, 3).transpose(0, 1), seq_root_v.transpose(0, 1))
            l_rec_root_v =  self.l2_criterion(root_v_out, seq_root_v)
            l_rec_root_trans = self.l2_criterion(pred_trajectory, gt_trajectory)
        else:
            l_rec_root_v = self.l2_criterion(root_v_out, seq_root_v)
            l_rec_root_trans = torch.zeros(1).cuda() 

        l_rec_6d = torch.zeros(1).cuda()
        l_rec_rot_mat = torch.zeros(1).cuda()
        l_rec_pose = torch.zeros(1).cuda()
        l_rec_joint_pos = torch.zeros(1).cuda()
        l_rec_linear_v = torch.zeros(1).cuda()
        l_rec_angular_v = torch.zeros(1).cuda()
        l_kl = torch.zeros(1).cuda()

        l_total = hp['rec_root_v_w'] * l_rec_root_v + hp['rec_root_trans_w'] * l_rec_root_trans
        
        if not validation_flag:
            l_total.backward()

        # return l_total, l_kl, l_rec_6d, l_rec_rot_mat, l_rec_pose, l_rec_joint_pos, l_rec_root_v, l_rec_linear_v, l_rec_angular_v
        return l_total, l_kl, l_rec_6d, l_rec_rot_mat, l_rec_pose, l_rec_joint_pos, l_rec_root_v, l_rec_linear_v, l_rec_root_trans

    def l2_criterion(self, pred, gt):
        assert pred.size() == gt.size() # If the two have different dimensions, there would be weird issues!
        loss = (pred-gt)**2

        return loss.mean()

    def l2_criterion_for_save(self, pred, gt):
        assert pred.size() == gt.size() # If the two have different dimensions, there would be weird issues!
        loss = (pred-gt)**2

        bs = loss.size()[0]
        timesteps = loss.size()[1]
        save_loss = loss.view(bs, -1).sum(dim=1) # bs

        t_save_loss = loss.view(bs, timesteps, -1).sum(dim=-1) # bs X T
        return loss.mean(), save_loss.data.cpu().numpy(), t_save_loss

    def de_standardize(self, output_data, start_idx, end_idx):
        # output_data: bs X n_dim 
        # output_data: T X bs X n_dim
        if len(output_data.size()) == 2:
            de_data = self.mean_vals[:, start_idx:end_idx] + self.std_vals[:, start_idx:end_idx] * output_data
        elif len(output_data.size()) == 3:
            de_data = self.mean_vals[None, :, :][:, :, start_idx:end_idx] + self.std_vals[None, :, :][:, :, start_idx:end_idx] * output_data

        return de_data

    def gen_motion_w_trajectory(self, pose_data, root_v_data, need_destandardize=True):
        # pose_data: T X bs X 24 X 3, root are origin
        # root_v_data: T X bs X 3, each timestep t for root represents the relative translation with respect to previous timestep, default is normalized value! 
        tmp_root_v_data = root_v_data.clone() # T X bs X 3    
        if need_destandardize:
            tmp_root_v_data = self.de_standardize(tmp_root_v_data, 576, 579)

        timesteps, bs, _, _ = pose_data.size()
        absolute_pose_data = pose_data.clone()
        root_trans = torch.zeros(bs, 3).cuda() # bs X 3
        for t_idx in range(1, timesteps):
            root_trans += tmp_root_v_data[t_idx, :, :] # bs X 3
            absolute_pose_data[t_idx, :, :, :] += root_trans[:, None, :] # bs X 24 X 3  

        return absolute_pose_data # T X bs X 24 X 3

    def adjust_root_rot(self, ori_seq_data):
        # ori_seq_data: bs X T X 24 X 3 X 3
        bs, timesteps, _, _, _ = ori_seq_data.size()
        target_root_rot = torch.eye(3).cuda() # 3 X 3
        target_root_rot = target_root_rot[None, :, :].repeat(bs, 1, 1) # bs X 3 X 3
        
        ori_root_rot = ori_seq_data[:, 0, 0, :, :] # bs x 3 X 3
        relative_rot = torch.matmul(target_root_rot, ori_root_rot.transpose(1, 2)) # bs X 3 X 3

        relative_rot = relative_rot[:, None, :, :].repeat(1, timesteps, 1, 1) # bs X T X 3 X 3

        converted_seq_data = torch.matmul(relative_rot.view(-1, 3, 3), ori_seq_data[:, :, 0, :, :].view(-1, 3, 3)) # (bs*T) X 3 X 3
        converted_seq_data = converted_seq_data.view(bs, timesteps, 3, 3)

        dest_seq_data = ori_seq_data.clone()
        # print("dest seq:{0}".format(dest_seq_data.size()))
        # print("converted seq data:{0}".format(converted_seq_data.size()))
        dest_seq_data[:, :, 0, :, :] = converted_seq_data

        return dest_seq_data, relative_rot 
        # bs X T X 24 X 3 X 3, bs X T X 3 X 3

    def apply_root_rot_to_translation(self, relative_root, ori_root_v):
        # relative_root: bs X T X 3 X 3 
        # ori_root_v: bs X T X 3
        converted_root_v = torch.matmul(relative_root, ori_root_v[:, :, :, None]) # bs X T X 3 X 1

        return converted_root_v.squeeze(-1)
    
    def standardize_data_specify_dim(self, ori_data, start_idx, end_idx):
        # ori_data: T X n_dim
        mean_val = self.mean_vals[:, start_idx:end_idx] # 1 X n_dim
        std_val = self.std_vals[:, start_idx:end_idx] # 1 X n_dim
        dest_data = (ori_data - mean_val)/std_val # T X n_dim

        return dest_data

    def test(self, data, hp, iterations, gen_seq_len=None):
        self.eval()
        self.enc.eval()

        with torch.no_grad():
            offset = None 
            seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v = data
            seq_rot_6d = seq_rot_6d.float().cuda() # bs X T X (24*6), not normalized
            seq_rot_mat = seq_rot_mat.float().cuda() # bs X T X (24*3*3), not normalized
            seq_rot_pos = seq_rot_pos.float().cuda() # bs X T X (24*3), not normalized
            seq_joint_pos = seq_joint_pos.float().cuda() # normalized 
            bs, timesteps, _ = seq_rot_6d.size()

            # Convert rot mat to a visualization used global rotation
            # seq_rot_mat, relative_rot = self.adjust_root_rot(seq_rot_mat.view(bs, timesteps, 24, 3, 3)) # bs X T X 24 X 3 X 3, bs X T X 3 X 3
            # seq_rot_pos = self.fk_layer(seq_rot_mat.view(bs*timesteps, self.n_joints, 3, 3)).detach() # (bs*T) X 24 X 3
            # seq_rot_pos = seq_rot_pos.view(bs, timesteps, self.n_joints, 3) # bs X T X 24 X 3
            # seq_rot_pos = seq_rot_pos.contiguous().view(bs, timesteps, -1) # bs X T X (24*3)
            # seq_rot_pos = seq_rot_pos.float().cuda() # bs X T X (24*3), not normalized

            seq_root_v = seq_root_v.float().cuda() # bs X T X 3, normalized
            # seq_root_v = self.de_standardize(seq_root_v.transpose(0, 1), 576, 579).transpose(0, 1) # bs X T X 3
            # seq_root_v_for_vis = self.apply_root_rot_to_translation(relative_rot, seq_root_v) # bs X T X 3
            
            gt_seq_res = seq_rot_pos.transpose(0, 1).contiguous().view(timesteps, bs, 24, 3) # T X bs X 24 X 3
              
            # gt_root_v = seq_root_v_for_vis.transpose(0, 1) # T X bs X 3
            gt_seq_res_w_trans = self.gen_motion_w_trajectory(gt_seq_res, seq_root_v.transpose(0, 1))

            if hp['trajectory_input_joint_pos']:
                encoder_input = seq_joint_pos.view(bs, timesteps, self.n_joints, -1) # bs X T X 24 X 3
            else:
                encoder_input = seq_rot_6d.view(bs, timesteps, self.n_joints, -1) # bs X T X 24 X 6

            input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*6)
            input = input.transpose(1, 2) # bs X (24*6) X T
            latent = self.enc(input, offset) # input: bs X (n_edges*input_dim) X T
            # latent: bs X (k_edges*d) X T
            
            # print("latent size:{0}".format(latent.size()))
            k_edges = latent.shape[1] // self.d_model
            encoder_map_input = latent.view(bs, k_edges, self.d_model, timesteps) # bs X k_edges X d X T
            encoder_map_input = encoder_map_input.transpose(2, 3).transpose(1, 2) # bs X T X k_edges X d
            root_v_out = self.fc_mapping(encoder_map_input.view(bs, timesteps, -1)) # bs X T X 3
            # root_v_out = root_v_out.mean(dim=2) # bs X T X 3
            # root_v_out = self.de_standardize(root_v_out.transpose(0, 1), 576, 579).transpose(0, 1) # bs X T X 3
            # out_root_v_for_vis = self.apply_root_rot_to_translation(relative_rot, root_v_out) # bs X T X 3
              
            # out_root_v = out_root_v_for_vis.transpose(0, 1) # T X bs X 3
            pred_seq_res_w_trans = self.gen_motion_w_trajectory(gt_seq_res, root_v_out.transpose(0, 1))
           
        self.train()
        self.enc.train()
        
        return gt_seq_res_w_trans, pred_seq_res_w_trans, pred_seq_res_w_trans, None
        # T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3

    def sampled_seq_test(self, data_input):
        # with torch.no_grad():
        offset = None 
        # data_input: bs X T X 24 X 6/ bs X T X 24 X 3
        bs, timesteps, _, _ = data_input.size() 

        if self.hp['trajectory_input_joint_pos']:
            # First do fk, then normalize fk pose 
            if data_input.size()[-1] == 6:
                fk_pose = self.fk_layer(data_input.view(bs*timesteps, -1, 6)) # (bs*T) X 24 X 3
            else:
                fk_pose = data_input.view(bs*timesteps, 24, 3)
            # print("fk pose:{0}".format(fk_pose.size()))
            fk_pose = fk_pose.contiguous().view(-1, 72) # (bs*T) X 72
            normalized_fk_pose = self.standardize_data_specify_dim(fk_pose, 24*6+24*3*3, 24*6+24*3*3+24*3) # (bs*T) X 72, 24*6+24*3*3:24*6+24*3*3+24*3
            normalized_fk_pose = normalized_fk_pose.view(bs, timesteps, 72) # bs X T X 72
            encoder_input = normalized_fk_pose.view(bs, timesteps, 24, 3) # bs X T X 24 X 3
        else:
            encoder_input = data_input

        input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*6)
        input = input.transpose(1, 2) # bs X (24*6) X T
        latent = self.enc(input, offset) # input: bs X (n_edges*input_dim) X T
        # latent: bs X (k_edges*d) X T
        
        # print("latent size:{0}".format(latent.size()))
        k_edges = latent.shape[1] // self.d_model
        encoder_map_input = latent.view(bs, k_edges, self.d_model, timesteps) # bs X k_edges X d X T
        encoder_map_input = encoder_map_input.transpose(2, 3).transpose(1, 2) # bs X T X k_edges X d
        root_v_out = self.fc_mapping(encoder_map_input.view(bs, timesteps, -1)) # bs X T X 3
        # root_v_out = root_v_out.mean(dim=2) # bs X T X 3
        # root_v_out = self.de_standardize(root_v_out.transpose(0, 1), 576, 579).transpose(0, 1) # bs X T X 3
            
        out_root_v = root_v_out.transpose(0, 1) # T X bs X 3
        if data_input.size()[-1] == 6:
            pose_seq_res = self.fk_layer(data_input.view(-1, 24, 6)).view(bs, timesteps, 24, 3) # bs X T X 24 X 3
        else:
            pose_seq_res = data_input # bs X T X 24 X 3
        pred_seq_res = self.gen_motion_w_trajectory(pose_seq_res.transpose(0, 1), out_root_v) # T X bs X 24 X 3

        return pred_seq_res
        # T X bs X 24 X 3
        
    def gen_seq(self, data, hp, iterations):
        return self.test(data, hp, iterations, hp['max_input_timesteps'])

    def aa2matrot(self, pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nxnum_jointsx3X3
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
        # bs X 1 X n_joints X 9
        pose_body_matrot = pose_body_matrot.view(batch_size, 1, self.n_joints, 3, 3) # bs X 1 X n_joints X 3 X 3
        pose_body_matrot = pose_body_matrot.squeeze(1) # bs X n_joints X 3 X 3
        return pose_body_matrot

    def aa2others(self, aa_data):
        # aa_data: bs X T X 72
        # rest_skeleton_offsets: bs X T X 24 X 3
        bs, timesteps, _ = aa_data.size()
        aa_data = aa_data.view(bs*timesteps, self.n_joints, 3)[:, None, :, :] # (bs*T) X 1 X n_joints X 3
        rot_mat_data = self.aa2matrot(aa_data) # (bs*T) X n_joints X 3 X 3
        # Convert first timesteps's root rotation to Identity
       
        rotMatrices = rot_mat_data # (bs*T) X 24 X 3 X 3
        # Convert rotation matrix to 6D representation
        cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # (bs*T) X 24 X 2 X 3
        cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # (bs*T) X 24 X 6

        pose_pos = self.fk_layer(rotMatrices) # (bs*T) X 24 X 3

        cont6DRep = cont6DRep.view(bs, timesteps, -1) # bs X T X (24*6)
        rotMatrices = rotMatrices.view(bs, timesteps, -1) # bs X T X (24*3*3)
        pose_pos = pose_pos.contiguous().view(bs, timesteps, -1) # bs X T X (24*3)

        return cont6DRep, rotMatrices, pose_pos

    def eval_pose_estimation(self, hp, image_directory): # For Comparison with VIBE
        pass 

    def evaluate_metrics(self, pred_j3ds, target_j3ds, use_24joints=False):
        pass 

    def evaluate_metrics_w_mask(self, ori_pred_j3ds, ori_target_j3ds, vis):
        pass 

    def encode_to_z_vector(self, encoder_input):
        # encoder_input: bs X T X 24 X 6
        bs, timesteps, _, _ = encoder_input.size()
        input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*6)
        input = input.transpose(1, 2) # bs X (24*6) X T
        latent = self.enc(input, offset=None) # input: bs X (n_edges*input_dim) X T
        # latent: bs X (k_edges*d) X (T//2^n_layers)
        k_edges = latent.shape[1] // self.d_model
        encoder_map_input = latent.view(bs, k_edges, self.d_model*self.compressed_t) # bs X k_edges X (d*T//2^n_layers)
        distributions = self.mid_fc_encode_mapping(encoder_map_input) # bs X k_edges X (2*latent_d)
        mu = distributions[:, :, :self.latent_d].view(-1, self.latent_d) # (bs*7) X latent_d
        logvar = distributions[:, :, self.latent_d:].view(-1, self.latent_d) # (bs*7) X latent_d

        return mu.view(bs, k_edges, -1) # bs X 7 X d 

    def encode_to_distribution(self, encoder_input):
        # encoder_input: bs X T X 24 X 6
        bs, timesteps, _, _ = encoder_input.size()
        input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*6)
        input = input.transpose(1, 2) # bs X (24*6) X T
        latent = self.enc(input, offset=None) # input: bs X (n_edges*input_dim) X T
        # latent: bs X (k_edges*d) X (T//2^n_layers)
        k_edges = latent.shape[1] // self.d_model
        encoder_map_input = latent.view(bs, k_edges, self.d_model*self.compressed_t) # bs X k_edges X (d*T//2^n_layers)
        distributions = self.mid_fc_encode_mapping(encoder_map_input) # bs X k_edges X (2*latent_d)
        mu = distributions[:, :, :self.latent_d].view(-1, self.latent_d) # (bs*7) X latent_d
        logvar = distributions[:, :, self.latent_d:].view(-1, self.latent_d) # (bs*7) X latent_d

        return mu, logvar

    def get_mean_rec_res(self, input_aa_rep): # Used for inserting into VIBE pipeline to get our model output
        # input_aa_rep: bs X T X 72
        batch_size, seqlen, _ = input_aa_rep.size()
        input_aa_rep = input_aa_rep.view(batch_size, seqlen, 24, 3) # bs X T X 24 X 3
        
        input_rot_mat = self.aa2matrot(input_aa_rep.view(-1, 1, 24, 3)) # (bs*T) X 24 X 3 X 3

        rotMatrices = input_rot_mat.view(-1, 24, 3, 3) # (bs*T) X 24 X 3 X 3
        # Convert rotation matrix to 6D representation
        gt_cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # (bs*T) X 24 X 2 X 3
        gt_cont6DRep = gt_cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # (bs*T) X 24 X 6

        input_6d_rep = gt_cont6DRep.view(batch_size, seqlen, 24, 6) # bs X T X 24 X 6 

        z_vec = self.encode_to_z_vector(input_6d_rep) # bs X 7 X d
        cont6d_rep, _, _, _, _, _, _ = self._decode(z_vec) # bs X T X 24 X 6
        
        # Our 6d has a mismatch with VIBE! Convert to consistent order!
        cont6d_rep = cont6d_rep.view(batch_size, seqlen, 24, 2, 3)
        cont6d_rep = cont6d_rep.transpose(3, 4) # bs X T X 24 X 3 X 2
        cont6d_rep = cont6d_rep.contiguous().view(batch_size, seqlen, 24, 6) # bs X T X 24 X 6
    
        return input_6d_rep, cont6d_rep
        # bs X T X 24 X 6(our 6d format), bs X T X 24 X 6(vibe 6d format)

    def get_mean_rec_res_w_6d_input(self, input_6d_rep):
        # input_6d_rep: bs X T X 24 X 6
        z_vec = self.encode_to_z_vector(input_6d_rep) # bs X 7 X d
        print("z vec size:{0}".format(z_vec.size()))
        cont6d_rep, _, _, _, _, _, _ = self._decode(z_vec) # bs X T X 24 X 6
    
        return cont6d_rep
        # bs X T X 24 X 6(our 6d format)

    def l2_masked_criterion(self, pred, gt, mask):
        # pred: bs X T X 24 X 6, bs X T X 24 X 3 X 3, bs X T X 24 X 3
        # mask: bs X T X 24
        if len(pred.shape) == 4:
            mask = mask[:, :, :, None] # bs X T X 24 X 1
        elif len(pred.shape) == 5:
            mask = mask[:, :, :, None, None] # bs X T X 24 X 1 X 1
        
        assert pred.size() == gt.size()
        loss = (pred-gt)**2
        loss = loss * mask # bs X T X 24 X 6/bs X T X 24 X 3 X 3/bs X T X 24 X 3
        bs = loss.size()[0]
        timesteps = loss.size()[1]
        # print("loss size:{0}".format(loss.size()))
        save_loss = loss.reshape(bs, timesteps, -1) # bs X T X d
        save_loss = save_loss.mean(dim=-1) # bs X T 
        # save_loss = None 

        return loss.mean(), save_loss
          
    def batch_complete_seq_partial_input_w_gt_target(self, dest_image_directory, input_gt=True, gen_vis=False): # For comparison with the partial-body model paper.  
        pass 

    def batch_complete_seq_amass(self, dest_image_directory, input_gt=True, gen_vis=False): # For comparison with the partial-body model paper.  
        pass 

    def rot_mat_to_6d(self, rotMatrices):
        # bs X 24 X 3 X 3
        # Convert rotation matrix to 6D representation
        cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # bs X 24 X 2 X 3
        cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # bs X 24 X 6

        return cont6DRep

    def get_j3d_from_smpl(self, rotmat, pred_shape):
        # rot_mat: T X 24 X 3 X 3
        # pred_shape: T x 10
        joints = self.smpl(
            pred_shape,
            rotmat,
        )

        print("joints:{0}".format(joints.shape))

        return joints 

    def human_dynamics_cal_metrics(self, joints_pred, joints_gt, vis):
        pass 

    def cmp_human_dynamics_eval_pose_estimation(self, hp, image_directory): # For Comparison with VIBE
        pass 

    def sample_long_seq(self, hp, image_directory):
        pass 