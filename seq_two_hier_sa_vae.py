import copy
import numpy as np
import json 
import os 
import pickle as pkl 
import datetime 
import random 

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import lr_scheduler

import torchgeometry as tgm

import my_tools
from fk_layer import ForwardKinematicsLayer
from skeleton import SkeletonUnpool, SkeletonPool, SkeletonConv, find_neighbor, SkeletonLinear, get_edges

from torch.distributions import Categorical
import torch.distributions.multivariate_normal as dist_mn

from utils_common import show3Dpose_animation, show3Dpose_animation_multiple, show3Dpose_animation_with_mask
from torch.utils.tensorboard import SummaryWriter

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

def get_opt_scheduler(optimizer, hp, it=-1):
    if 'opt_lr_policy' not in hp or hp['opt_lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hp['opt_lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hp['opt_step_size'],
                                        gamma=hp['opt_gamma'], last_epoch=it)
    elif hp['opt_lr_policy'] == 'mstep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=hp['opt_step_size'],
                                        gamma=hp['opt_gamma'], last_epoch=it)
    else:
        return NotImplementedError('%s not implemented', hp['opt_lr_policy'])
    return scheduler

class Encoder(nn.Module):
    def __init__(self, args, topology):
        super(Encoder, self).__init__()
        self.topologies = [topology]
      
        self.latent_d = args['latent_d']
        self.shallow_latent_d = args['shallow_latent_d']

        self.channel_base = [6] # 6D representation 
        self.timestep_list = [args['train_seq_len']]
    
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.latent_enc_layers = nn.ModuleList() # Hierarchical latent vectors from different depth of layers 
        self.args = args
        self.convs = []

        kernel_size = args['kernel_size']
        padding = (kernel_size - 1) // 2
        bias = True

        for i in range(args['num_layers']):
            self.channel_base.append(self.channel_base[-1] * 2) # 6, 12, 24,(48, 96)
          
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    self.timestep_list.append(self.timestep_list[-1]) # 8, 8, 4, 2, 2
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2)
            elif args['train_seq_len'] == 16:
                if i == 0: # For len = 16
                    self.timestep_list.append(self.timestep_list[-1]) # 16, 16, 8, 4, 2
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2) 
            else:
                self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4
            # print("timestep list:{0}".format(self.timestep_list))

        for i in range(args['num_layers']):
            seq = []
            neighbor_list = find_neighbor(self.topologies[i], args['skeleton_dist']) # 24, 14, 9, 7
            in_channels = self.channel_base[i] * self.edge_num[i] # 6 * 24, 12 * 14, 24 * 9, 48 * 7,
            out_channels = self.channel_base[i+1] * self.edge_num[i] # 12 * 24, 24 * 14, 48 * 9,  96 * 7
         
            if i == 0: self.channel_list.append(in_channels)
            self.channel_list.append(out_channels) # 6*24, 12*14, 24*9, 48*7, 96*7

            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
          
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    curr_stride = 1
                else:
                    curr_stride = 2
            elif args['train_seq_len'] == 16:
                if i == 0:
                    curr_stride = 1
                else:
                    curr_stride = 2 
            else:
                curr_stride = 2

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
    
            if i == 0:
                latent_encode_linear = nn.Linear(self.channel_base[i+1]*self.timestep_list[i+1], self.shallow_latent_d*2)
            else:
                latent_encode_linear = nn.Linear(self.channel_base[i+1]*self.timestep_list[i+1], self.latent_d*2)
            self.latent_enc_layers.append(latent_encode_linear)

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

    def forward(self, input, offset=None):
        # train_hier_level: 1, 2, 3, 4 (deep to shallow)
        z_vector_list = []
        for i, layer in enumerate(self.layers):
            # print("i:{0}".format(i))
            # print("layer:{0}".format(layer))
            # print("layer input:{0}".format(input.size()))
            input = layer(input)
            # print("layer output:{0}".format(input.size()))
         
            # latent: bs X (k_edges*d) X (T//2^n_layers)
            bs, _, compressed_t = input.size()
            # print("input shape[1]:{0}".format(input.shape[1]))
            # print("channel:{0}".format(self.channel_base[i+1]))
            k_edges = input.shape[1] // self.channel_base[i+1]
            # print("k_edges:{0}".format(k_edges))
            
            encoder_map_input = input.view(bs, k_edges, -1)
            # print("encoder_map_input:{0}".format(encoder_map_input.size()))

            curr_z_vector = self.latent_enc_layers[i](encoder_map_input)
            # print("curr_z_vector:{0}".format(curr_z_vector.size()))
            z_vector_list.append(curr_z_vector)
           
        return input, z_vector_list 
        # each z_vector is bs X k_edges X (2*latent_d)

class Decoder(nn.Module):
    def __init__(self, args, enc: Encoder):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.latent_dec_layers = nn.ModuleList() # Hierarchical latent vectors from different depth of layers 

        self.latent_d = args['latent_d']
        self.shallow_latent_d = args['shallow_latent_d']

        self.args = args
        self.enc = enc
        self.convs = []

        self.hp = args

        self.timestep_list = [args['train_seq_len']]
        for i in range(args['num_layers']):
            if args['train_seq_len'] == 8:
                if i == 0 or i == args['num_layers'] - 1:
                    self.timestep_list.append(self.timestep_list[-1]) # 16, 16, 8, 4, 2(8, 8, 4, 2, 2)
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4
            elif args['train_seq_len'] == 16:
                if i == 0:
                    self.timestep_list.append(self.timestep_list[-1]) # 16, 16, 8, 4, 2(8, 8, 4, 2)
                else:
                    self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4
            else:
                self.timestep_list.append(self.timestep_list[-1]//2) # 64, 32, 16, 8, 4

        self.timestep_list.reverse() # 4, 8, 16, 32, 64 ( 2, 2, 4, 8, 8; 2, 4, 8, 16, 16)

        kernel_size = args['kernel_size']
        padding = (kernel_size - 1) // 2

        for i in range(args['num_layers']):
            seq = []
            if i == args['num_layers'] - 1:
                in_channels = enc.channel_list[args['num_layers'] - i]*2
            else:
                in_channels = enc.channel_list[args['num_layers'] - i] # 7*96, 9*48, 14*24, 24*12
            # print("in channels:{0}".format(in_channels))
            if i == args['num_layers'] - 1:
                out_channels = in_channels // 4
            else:   
                out_channels = in_channels // 2 # 7*96->7*24, 9*48->9*12, 14*24->14*6, 24*12->24*6
            
            neighbor_list = find_neighbor(enc.topologies[args['num_layers'] - i - 1], args['skeleton_dist']) # 7, 9, 14, 24
            # neighbor_list = find_neighbor(enc.topologies[args['num_layers'] - i], args['skeleton_dist']) # 7, 7, 9, 14

            if i != 0 and i != args['num_layers'] - 1:
                bias = False
            else:
                bias = True
          
            if i == args['num_layers'] - 1:
                latent_decode_linear = nn.Linear(self.shallow_latent_d, enc.channel_base[args['num_layers']-i]*self.timestep_list[i]) # 96*4
            else:
                latent_decode_linear = nn.Linear(self.latent_d, enc.channel_base[args['num_layers']-i]*self.timestep_list[i]) # 24*8, 12*16, 6*32
            self.latent_dec_layers.append(latent_decode_linear)

            self.unpools.append(SkeletonUnpool(enc.pooling_list[args['num_layers'] - i - 1], in_channels // len(neighbor_list)))

            if args['train_seq_len'] == 8:
                if i != args['num_layers'] - 1 and i != 0:
                    seq.append(nn.Upsample(scale_factor=2, mode=args['upsampling'], align_corners=False))
            elif args['train_seq_len'] == 16:
                if i != args['num_layers'] - 1:
                    seq.append(nn.Upsample(scale_factor=2, mode=args['upsampling'], align_corners=False))
            else:
                seq.append(nn.Upsample(scale_factor=2, mode=args['upsampling'], align_corners=False))

            seq.append(self.unpools[-1])
            for _ in range(args['extra_conv']):
                seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=in_channels,
                                        joint_num=enc.edge_num[args['num_layers'] - i - 1], kernel_size=kernel_size,
                                        stride=1,
                                        padding=padding, padding_mode=args['padding_mode'], bias=bias))
      
            curr_stride = 1 
           
            seq.append(SkeletonConv(neighbor_list, in_channels=in_channels, out_channels=out_channels,
                                    joint_num=enc.edge_num[args['num_layers'] - i - 1], kernel_size=kernel_size, stride=curr_stride,
                                    padding=padding, padding_mode=args['padding_mode'], bias=bias, add_offset=False,
                                    in_offset_channel=3 * enc.channel_base[args['num_layers'] - i - 1] // enc.channel_base[0]))
            self.convs.append(seq[-1])
            if i != args['num_layers'] - 1: seq.append(nn.LeakyReLU(negative_slope=0.2))

            self.layers.append(nn.Sequential(*seq))

    def forward(self, z_vec_list, offset=None):
        # train_hier_level: 1, 2, 3, 4 (deep to shallow)
        hier_feats = []
        num_z_vec = len(z_vec_list)
        for z_idx in range(len(z_vec_list)):
            curr_z_vector = z_vec_list[num_z_vec - z_idx - 1] # bs X k_edges X latent_d
            # print("curr_z_vec:{0}".format(curr_z_vector.size()))
            curr_feats = self.latent_dec_layers[z_idx](curr_z_vector) # each is bs X k_edges X (d*timesteps)
            # print("before view curr feats:{0}".format(curr_feats.size()))
            bs = curr_z_vector.size()[0]
            curr_feats = curr_feats.view(bs, -1, self.timestep_list[z_idx]) # bs X (k_edges*d_feats) X T'
            # print("curr_feats:{0}".format(curr_feats.size()))
            hier_feats.append(curr_feats) # from deep to shallow layer feats

        # hier feats: bs X (7*96) X 4, bs X (7*24) X 8, bs X (9*12) X 16, bs X (14*6) X 32
        for i, layer in enumerate(self.layers): # From deep to shallow layers
            # print("Decoder i:{0}".format(i))
            # print("Decoder layer:{0}".format(layer))
            if i == 0:
                input = hier_feats[i] # bs X (k_edges*d) X T'
            elif i == self.hp['num_layers'] - 1:
                bs, k_d, compressed_t = input.size()
                k_edges = self.enc.edge_num[self.hp['num_layers']-i]
                # print("decoder forward k edges:{0}".format(k_edges))
                tmp_input = input.view(bs, k_edges, -1, compressed_t)
                tmp_hier_feats = hier_feats[i].view(bs, k_edges, -1, compressed_t)
                
                tmp_cat_feats = torch.cat((tmp_input, tmp_hier_feats), dim=2) # bs X k_edges X (d+d') X T'
                input = tmp_cat_feats.view(bs, -1, compressed_t) # bs X (k_edges*2d) X T'
               
            # print("Decoder layer input:{0}".format(input.size()))
            input = layer(input)
            # print("Decoder layer output:{0}".format(input.size()))
            
        return input

class TwoHierSAVAEModel(nn.Module):
    def __init__(self, hp):
        super(TwoHierSAVAEModel, self).__init__()

        self.latent_d = hp['latent_d']
        self.shallow_latent_d = hp['shallow_latent_d']
        self.n_joints = hp['n_joints']
        self.input_dim = hp['input_dim'] 
        self.output_dim = hp['output_dim']
        self.max_timesteps = hp['train_seq_len']

        parent_json = "./utils/data/joint24_parents.json"
        edges = get_edges(parent_json) # a list with 23(n_joints-1) elements, each elements represents a pair (parent_idx, idx)

        self.fk_layer = ForwardKinematicsLayer()

        self.hp = hp 

        self.enc = Encoder(hp, edges)
        self.dec = Decoder(hp, self.enc)
      
        self.iteration_interval = hp['iteration_interval']

        mean_std_npy = "./utils/data/for_all_data_motion_model/all_amass_data_mean_std.npy"
              
        mean_std_data = np.load(mean_std_npy) # 2 X n_dim
        mean_std_data[1, mean_std_data[1, :]==0] = 1.0

        self.mean_vals = torch.from_numpy(mean_std_data[0, :]).float()[None, :].cuda() # 1 X n_dim
        self.std_vals = torch.from_numpy(mean_std_data[1, :]).float()[None, :].cuda() # 1 X n_dim

    def get_hier_level(self, step):
        if step < self.iteration_interval:
            level = 1
        else:
            level = 4

        return level

    def forward(self, data, hp, iterations, multigpus=False, validation_flag=False):
        offset = None 
        # input = torch.zeros(8, 24*6, 30).cuda() # For debug
        seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v = data
        seq_rot_6d = seq_rot_6d.float().cuda() # bs X T X (24*6), not normalized
        seq_rot_mat = seq_rot_mat.float().cuda() # bs X T X (24*3*3), not normalized
        # seq_rot_pos = seq_rot_pos.float().cuda() # bs X T X (24*3), not normalized
        bs, timesteps, _ = seq_rot_6d.size()
        seq_rot_pos = self.fk_layer(seq_rot_mat.view(bs*timesteps, self.n_joints, 3, 3)).detach() # (bs*T) X 24 X 3
        seq_rot_pos = seq_rot_pos.view(bs, timesteps, self.n_joints, 3) # bs X T X 24 X 3
        seq_rot_pos = seq_rot_pos.contiguous().view(bs, timesteps, -1) # bs X T X (24*3)
        seq_root_v = seq_root_v.float().cuda() # bs X T X 3, normalized

        encoder_input = seq_rot_6d.view(bs, timesteps, self.n_joints, -1) # bs X T X 24 X 6
       
        input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*9)
        input = input.transpose(1, 2) # bs X (24*9) X T
       
        latent, z_vec_list = self.enc(input, offset) # input: bs X (n_edges*input_dim) X T
        # latent: bs X (k_edges*d) X (T//2^n_layers)
        # list, each is bs X k_edges X (2*latent_d)
       
        z_list = []
        l_kl_list = []
        for z_idx in range(len(z_vec_list)):
            distributions = z_vec_list[z_idx] # bs X k_edges X (2*latent_d)
            bs, k_edges, _ = distributions.size()
            if z_idx == 0:
                mu = distributions[:, :, :self.shallow_latent_d].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
                logvar = distributions[:, :, self.shallow_latent_d:].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
            else:
                mu = distributions[:, :, :self.latent_d].view(-1, self.latent_d) # (bs*k_edges) X latent_d
                logvar = distributions[:, :, self.latent_d:].view(-1, self.latent_d) # (bs*k_edges) X latent_d

            if hp['kl_w'] != 0:
                z = self.reparametrize(mu, logvar) # (bs*7) X latent_d
            else:
                z = mu # (bs*7) X latent_d

            z = z.view(bs, k_edges, -1) # bs X 7 X latent_d

            iteration_interval = hp['iteration_interval']
   
            if z_idx == len(z_vec_list)-1: # The final deepest layer level
                l_kl_curr = self.kl_loss(logvar, mu)
            elif z_idx == 0:
                if iterations < iteration_interval:
                    l_kl_curr = self.kl_loss(logvar.detach(), mu.detach())
                    z = z.detach()
                else:
                    l_kl_curr = self.kl_loss(logvar, mu)
            else:
                l_kl_curr = torch.zeros(1).cuda()

            z_list.append(z)

            l_kl_list.append(l_kl_curr) # From shallow to deep layers

        out_cont6d, out_rotation_matrix, out_pose_pos, root_v_out, _, _, _ = self._decode(z_list)

        l_rec_6d = self.l2_criterion(out_cont6d.contiguous().view(bs, self.max_timesteps, -1), seq_rot_6d)
        l_rec_rot_mat = self.l2_criterion(out_rotation_matrix.view(bs, self.max_timesteps, -1), seq_rot_mat)
        l_rec_pose = self.l2_criterion(out_pose_pos.contiguous().view(bs, self.max_timesteps, -1), seq_rot_pos)
   
        l_rec_root_v = torch.zeros(1).cuda()

        l_rec_joint_pos = torch.zeros(1).cuda()
        l_rec_linear_v = torch.zeros(1).cuda()
        l_rec_angular_v = torch.zeros(1).cuda()

        l_kl = torch.zeros(1).cuda()
        l_pre_kl = torch.zeros(1).cuda() 

        l_kl = hp['kl_w'] * l_kl_list[3] + hp['shallow_kl_w'] * l_kl_list[0]

        l_total = hp['rec_6d_w'] * l_rec_6d + hp['rec_rot_w'] * l_rec_rot_mat + hp['rec_pose_w'] * l_rec_pose +  hp['kl_w'] * l_kl_list[3] + \
            hp['shallow_kl_w'] * l_kl_list[0]
        
        if not validation_flag:
            l_total.backward()

        return l_total, l_kl, l_rec_6d, l_rec_rot_mat, l_rec_pose, l_rec_joint_pos, l_rec_root_v, \
        l_rec_linear_v, l_rec_angular_v, l_kl_list

    def reparametrize(self, pred_mean, pred_logvar):
        random_z = torch.randn_like(pred_mean)
        vae_z = random_z * torch.exp(0.5 * pred_logvar)
        vae_z = vae_z + pred_mean
        return vae_z

    def kl_loss(self, logvar, mu):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        return loss.mean()

    def l2_criterion(self, pred, gt):
        assert pred.size() == gt.size() # If the two have different dimensions, there would be weird issues!
        loss = (pred-gt)**2

        return loss.mean()

    def _decode(self, z_list, adjust_root_rot_flag=False, relative_root_rot=None):
        # list of z: bs X 7 X latent_d
        result = self.dec(z_list) # bs X (24*out_dim) X T
        bs = result.size()[0]
        # print("output size:{0}".format(result.size()))
        result = result.transpose(1, 2) # bs X T X (24*out_dim)
        # print("result size:{0}".format(result.size()))
        decoder_out = result.contiguous().view(bs*self.max_timesteps, self.n_joints, -1) # (bs*T) X n_edges X out_dim

        out_dim = self.output_dim
        cont6d_rep = decoder_out[:, :, :out_dim] # (bs*T) X n_edges X 6
      
        root_v_out = None 

        joint_pos_out = None 
        linear_v_out = None 
        angular_v_out = None 

        # Convert 6d representation to rotation matrix
        out_rotation_matrix = my_tools.rotation_matrix_from_ortho6d(cont6d_rep) # (bs*T) X 24 X 3 X 3

        if adjust_root_rot_flag: # Only used in testing for Visualization! 
            if relative_root_rot is not None:
                # relative_root_rot: bs X T X 3 X 3
                out_rotation_matrix = out_rotation_matrix.view(bs, self.max_timesteps, 24, 3, 3)
                out_rotation_matrix[:, :, 0, :, :] = torch.matmul(relative_root_rot, out_rotation_matrix[:, :, 0, :, :])
            else:
                out_rotation_matrix, relative_root_rot = self.adjust_root_rot(out_rotation_matrix.view(bs, self.max_timesteps, 24, 3, 3)) # bs X T X 24 X 3 X 3, bs X T X 3 X 3
          
            out_rotation_matrix = out_rotation_matrix.view(bs*self.max_timesteps, 24, 3, 3)

        # Convert 6d representation to pose
        out_pose_pos = self.fk_layer(out_rotation_matrix) # (bs*T) X 24 X 3

        out_cont6d = cont6d_rep.view(bs, self.max_timesteps, self.n_joints, -1) # bs X T X 24 X 6
        out_rotation_matrix = out_rotation_matrix.view(bs, self.max_timesteps, self.n_joints, 3, 3) # bs X T X 24 X 3 X 3
        out_pose_pos = out_pose_pos.view(bs, self.max_timesteps, self.n_joints, 3) # bs X T X 24 X 3

        return out_cont6d, out_rotation_matrix, out_pose_pos, root_v_out, joint_pos_out, linear_v_out, angular_v_out
    
    def de_standardize(self, output_data, start_idx, end_idx):
        # output_data: bs X n_dim 
        # output_data: T X bs X n_dim
        if len(output_data.size()) == 2:
            de_data = self.mean_vals[:, start_idx:end_idx] + self.std_vals[:, start_idx:end_idx] * output_data
        elif len(output_data.size()) == 3:
            de_data = self.mean_vals[None, :, :][:, :, start_idx:end_idx] + self.std_vals[None, :, :][:, :, start_idx:end_idx] * output_data

        return de_data

    def gen_motion_w_trajectory(self, pose_data, root_v_data):
        # pose_data: T X bs X 24 X 3, root are origin
        # root_v_data: T X bs X 3, each timestep t for root represents the relative translation with respect to previous timestep
        tmp_root_v_data = root_v_data.clone() # T X bs X 3    
        tmp_root_v_data = self.de_standardize(tmp_root_v_data, 576, 579)

        timesteps, bs, _, _ = pose_data.size()
        absolute_pose_data = pose_data.clone()
        root_trans = torch.zeros(bs, 3).cuda() # bs X 3
        for t_idx in range(1, timesteps):
            root_trans += tmp_root_v_data[t_idx, :, :] # bs X 3
            absolute_pose_data[t_idx, :, :, :] += root_trans[:, None, :] # bs X 24 X 3  

        return absolute_pose_data # T X bs X 24 X 3

    def _decode_w_given_decoder(self, z_list, curr_decoder):
        # list of z: bs X 7 X latent_d
        result = curr_decoder(z_list, 1, 4) # bs X (24*out_dim) X T
        bs = result.size()[0]
        # print("output size:{0}".format(result.size()))
        result = result.transpose(1, 2) # bs X T X (24*out_dim)
        # print("result size:{0}".format(result.size()))
        decoder_out = result.contiguous().view(bs*self.max_timesteps, self.n_joints, -1) # (bs*T) X n_edges X out_dim

        out_dim = self.output_dim
        cont6d_rep = decoder_out[:, :, :out_dim] # (bs*T) X n_edges X 6
        
        root_v_out = None 

        joint_pos_out = None 
        linear_v_out = None 
        angular_v_out = None 

        # Convert 6d representation to rotation matrix
        out_rotation_matrix = my_tools.rotation_matrix_from_ortho6d(cont6d_rep) # (bs*T) X 24 X 3 X 3

        # Convert 6d representation to pose
        out_pose_pos = self.fk_layer(out_rotation_matrix) # (bs*T) X 24 X 3

        out_cont6d = cont6d_rep.view(bs, self.max_timesteps, self.n_joints, -1) # bs X T X 24 X 6
        out_rotation_matrix = out_rotation_matrix.view(bs, self.max_timesteps, self.n_joints, 3, 3) # bs X T X 24 X 3 X 3
        out_pose_pos = out_pose_pos.view(bs, self.max_timesteps, self.n_joints, 3) # bs X T X 24 X 3

        return out_cont6d, out_rotation_matrix, out_pose_pos, root_v_out, joint_pos_out, linear_v_out, angular_v_out

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

    def test(self, data, hp, iterations, gen_seq_len=None):
        self.eval()
        self.enc.eval()
        self.dec.eval()

        with torch.no_grad():
            offset = None 
            seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v = data
            seq_rot_6d = seq_rot_6d.float().cuda() # bs X T X (24*6), not normalized
            seq_rot_mat = seq_rot_mat.float().cuda() # bs X T X (24*3*3), not normalized
            # seq_rot_pos = seq_rot_pos.float().cuda() # bs X T X (24*3), not normalized
            bs, timesteps, _ = seq_rot_6d.size()
            seq_root_v = seq_root_v.float().cuda() # bs X T X 3, not normalized

            # Convert rot mat to a visualization used global rotation
            if hp['random_root_rot_flag']:
                seq_rot_mat, relative_rot = self.adjust_root_rot(seq_rot_mat.view(bs, timesteps, 24, 3, 3)) # bs X T X 24 X 3 X 3, bs X T X 3 X 3
            seq_rot_pos = self.fk_layer(seq_rot_mat.view(bs*timesteps, self.n_joints, 3, 3)).detach() # (bs*T) X 24 X 3
            seq_rot_pos = seq_rot_pos.view(bs, timesteps, self.n_joints, 3) # bs X T X 24 X 3
            seq_rot_pos = seq_rot_pos.contiguous().view(bs, timesteps, -1) # bs X T X (24*3)
            # seq_rot_pos = seq_rot_pos.float().cuda() # bs X T X (24*3), not normalized
            seq_root_v = seq_root_v.float().cuda() # bs X T X 3, not normalized
            # seq_root_v_for_vis = self.apply_root_rot_to_translation(relative_rot, seq_root_v) # bs X T X 3
            gt_seq_res = seq_rot_pos.transpose(0, 1).contiguous().view(timesteps, bs, 24, 3) # T X bs X 24 X 3

            encoder_input = seq_rot_6d.view(bs, timesteps, self.n_joints, -1) # bs X T X 24 X 6

            input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*9)
            input = input.transpose(1, 2) # bs X (24*9) X T
            latent, z_vec_list = self.enc(input, offset) # input: bs X (n_edges*input_dim) X T
            # latent: bs X (k_edges*d) X (T//2^n_layers)
            # list, each is bs X k_edges X (2*latent_d)
        
            mean_z_list = []
            sampled_z_list = []
            for z_idx in range(len(z_vec_list)):
                distributions = z_vec_list[z_idx] # bs X k_edges X (2*latent_d)
                bs, k_edges, _ = distributions.size()
                if z_idx == 0:
                    mu = distributions[:, :, :self.shallow_latent_d].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
                    logvar = distributions[:, :, self.shallow_latent_d:].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
                else:
                    mu = distributions[:, :, :self.latent_d].view(-1, self.latent_d) # (bs*k_edges) X latent_d
                    logvar = distributions[:, :, self.latent_d:].view(-1, self.latent_d) # (bs*k_edges) X latent_d

                mean_z = mu # (bs*7) X latent_d
                mean_z = mean_z.view(bs, k_edges, -1) # bs X 7 X latent_d
                mean_z_list.append(mean_z)

                sampled_z = torch.randn_like(mean_z).cuda()
                sampled_z_list.append(sampled_z)

            if hp['random_root_rot_flag']:
                mean_out_cont6d, mean_out_rotation_matrix, mean_out_pose_pos, \
                mean_out_root_v, _, _, _ = self._decode(mean_z_list,  \
                    adjust_root_rot_flag=True, relative_root_rot=relative_rot)
            else:
                mean_out_cont6d, mean_out_rotation_matrix, mean_out_pose_pos, \
                mean_out_root_v, _, _, _ = self._decode(mean_z_list)

            # bs X T X 24 X 6, bs X T X 24 X 3 X 3, bs X T X 24 X 3
            use_mean_seq_res = mean_out_pose_pos.transpose(0, 1) # T X bs X 24 X 3
           
            # Get sampled vector motion
            if hp['random_root_rot_flag']:
                sampled_out_cont6d, sampled_out_rotation_matrix, sampled_out_pose_pos, \
                sampled_out_root_v, _, _, _ = self._decode(sampled_z_list, \
                     adjust_root_rot_flag=True)
            else:
                sampled_out_cont6d, sampled_out_rotation_matrix, sampled_out_pose_pos, \
                sampled_out_root_v, _, _, _ = self._decode(sampled_z_list)
            # bs X T X 24 X 6, bs X T X 24 X 3 X 3, bs X T X 24 X 3
            use_sampled_seq_res = sampled_out_pose_pos.transpose(0, 1) # T X bs X 24 X 3
           
        self.train()
        self.enc.train()
        self.dec.train()

        return gt_seq_res, use_mean_seq_res, use_sampled_seq_res, None
        # T X bs X 24 X 3, T X bs X 24 X 3, T X bs X 24 X 3

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

    def encode_to_z_vector(self, encoder_input):
        # encoder_input: bs X T X 24 X 6
        mu_list, logvar_list = self.encode_to_distribution(encoder_input)

        return mu_list # bs X 7 X d 

    def encode_to_distribution(self, encoder_input):
        # encoder_input: bs X T X 24 X 6
        bs, timesteps, _, _ = encoder_input.size()
        input = encoder_input.view(bs, timesteps, -1) # bs X T X (24*6)
        input = input.transpose(1, 2) # bs X (24*6) X T

        fade_in_alpha = 1
        train_hier_level = 4
        offset = None
        latent, z_vec_list = self.enc(input, fade_in_alpha, train_hier_level, offset) # input: bs X (n_edges*input_dim) X T
        # latent: bs X (k_edges*d) X (T//2^n_layers)
        # list, each is bs X k_edges X (2*latent_d)
    
        mean_z_list = []
        logvar_list = []
        for z_idx in range(len(z_vec_list)):
            distributions = z_vec_list[z_idx] # bs X k_edges X (2*latent_d)
            bs, k_edges, _ = distributions.size()
            if z_idx == 0:
                mu = distributions[:, :, :self.shallow_latent_d].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
                logvar = distributions[:, :, self.shallow_latent_d:].view(-1, self.shallow_latent_d) # (bs*k_edges) X latent_d
            else:
                mu = distributions[:, :, :self.latent_d].view(-1, self.latent_d) # (bs*k_edges) X latent_d
                logvar = distributions[:, :, self.latent_d:].view(-1, self.latent_d) # (bs*k_edges) X latent_d

            mean_z = mu # (bs*7) X latent_d
            mean_z = mean_z.view(bs, k_edges, -1) # bs X 7 X latent_d
            mean_z_list.append(mean_z)

            logvar = logvar.view(bs, k_edges, -1)
            logvar_list.append(logvar)

        return mean_z_list, logvar_list 

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

    def load_amass_test_data(self, max_num_seq=None):
        root_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/utils/data"

        data_folder = os.path.join(root_folder, "for_all_data_motion_model")
        rot_npy_folder = os.path.join(root_folder, "processed_all_amass_data")
       
        test_json = os.path.join(data_folder, "test_all_amass_motion_data.json")
        test_json_data = json.load(open(test_json, 'r'))
        seq_cnt = 0
        batch_num = max_num_seq

        ori_pose_seq_data_list = []
        v_name_list = []
        for k in test_json_data:
            v_name = test_json_data[k]
            rot_npy_path = os.path.join(rot_npy_folder, v_name)
            ori_pose_seq_data = np.load(rot_npy_path) # T X n_dim
            
            # if ori_pose_seq_data.shape[0] >= self.max_timesteps and ("Jog" in v_name or "Walk" in v_name):
            if ori_pose_seq_data.shape[0] >= self.max_timesteps:
                ori_pose_seq_data = torch.from_numpy(ori_pose_seq_data).float()
                rot_mat_data = ori_pose_seq_data[:, 24*6:24*6+24*3*3] # T X (24*3*3)
                timesteps = rot_mat_data.size()[0] 
                ori_pose_seq_data_list.append(rot_mat_data.view(timesteps, 24, 3, 3))
                seq_cnt += 1

                v_name_list.append(v_name)     

            if max_num_seq is not None:
                if seq_cnt >= batch_num:
                    break;

        print("Total seq:{0}".format(seq_cnt))

        return v_name_list, ori_pose_seq_data_list

    def rot_mat_to_6d(self, rotMatrices):
        # bs X 24 X 3 X 3
        # Convert rotation matrix to 6D representation
        cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # bs X 24 X 2 X 3
        cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # bs X 24 X 6

        return cont6DRep

    def sample_single_seq(self):
        # Get sampled vector motion
        bs = 256
        k_edges = 7
        sampled_z = torch.randn(bs, k_edges, self.latent_d).cuda() # bs X 7 X latent_d
        decoder_input = sampled_z # bs X 7 X latent_d
        sampled_out_cont6d, sampled_out_rotation_matrix, sampled_out_pose_pos, \
        sampled_out_root_v, _, _, _ = self._decode(decoder_input)
        # bs X T X 24 X 6, bs X T X 24 X 3 X 3, bs X T X 24 X 3

        # Need to do a processing for 6d output? (since the direct output might not be normalized one, will introduce noise for trajectory prediction model?)
        clean_cont6d = my_tools.rotation_matrix_from_ortho6d(sampled_out_cont6d.view(-1, 24, 6))
        clean_cont6d = self.rot_mat_to_6d(clean_cont6d)
        clean_cont6d = clean_cont6d.view(bs, -1, 24, 6)

        return clean_cont6d 

    def sample_single_seq_w_trajectory(self, trajectory_trainer, dest_image_directory):
        shallow_k_edges = 14
        deep_k_edges = 7

        bs = 300
        
        useless_z_2 = torch.zeros(bs, 9, self.latent_d).cuda()
        useless_z_3 = torch.zeros(bs, 7, self.latent_d).cuda()

        deep_sampled_z = torch.randn(bs, deep_k_edges, self.latent_d).cuda()
        
        sampled_z_list = []

        # shallow_sampled_z = torch.randn(bs, shallow_k_edges, self.shallow_latent_d).cuda()
        shallow_sampled_z = torch.zeros(bs, shallow_k_edges, self.shallow_latent_d).cuda()

        sampled_z_list.append(shallow_sampled_z)
        sampled_z_list.append(useless_z_2)
        sampled_z_list.append(useless_z_3)
        sampled_z_list.append(deep_sampled_z)

        # Get sampled vector motion
        sampled_out_cont6d, sampled_out_rotation_matrix, sampled_out_pose_pos, \
        sampled_out_root_v, _, _, _ = self._decode(sampled_z_list, 1, 4)
        # bs X T X 24 X 6, bs X T X 24 X 3 X 3, bs X T X 24 X 3

        # Need to do a processing for 6d output? (since the direct output might not be normalized one, will introduce noise for trajectory prediction model?)
        clean_rot_mat = my_tools.rotation_matrix_from_ortho6d(sampled_out_cont6d.view(-1, 24, 6))
        clean_cont6d = self.rot_mat_to_6d(clean_rot_mat)
        clean_cont6d = clean_cont6d.view(bs, -1, 24, 6)
        clean_rot_mat = clean_rot_mat.view(bs, -1, 24, 3, 3)

        encoder_input = clean_cont6d # bs X T X 24 X 6
        pred_seq_w_root_trans_res = trajectory_trainer.sampled_seq_test(encoder_input) # T X bs X 24 X 3

        for bs_idx in range(bs):
            seq_for_vis = pred_seq_w_root_trans_res[:, bs_idx, :, :][None] # 1 X T X 24 X 3
            show3Dpose_animation(seq_for_vis.data.cpu().numpy(), dest_image_directory, \
                    0, "sampled_single_window", str(bs_idx), use_amass=True)

            # Save corresponding rotation information
            dest_rot_npy_path = os.path.join(dest_image_directory, str(0), \
                "sampled_single_window", str(bs_idx)+"_rot.npy")
            np.save(dest_rot_npy_path, clean_rot_mat[bs_idx].data.cpu().numpy()) # T X 24 X 3 X 3

            # Save corresponding translation information
            dest_trans_npy_path = os.path.join(dest_image_directory, str(0), \
                "sampled_single_window", str(bs_idx)+"_trans.npy")
            np.save(dest_trans_npy_path, pred_seq_w_root_trans_res[:, bs_idx, 0, :].data.cpu().numpy()) # T X 3

    def refine_dance_motions(self, hp, image_directory): # For Comparison with VIBE
        self.eval()
        self.enc.eval()
        self.dec.eval()

        # npy_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/output/tmp_test"
        # npy_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/dance_test_outputs/urban_dance_camp_00005_4/urban_dance_camp_00005_4"
        # npy_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/for_supp_outputs/final_walk/final_walk"
        npy_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/talk_test_output/test_tmp/test_tmp"
        pred_theta_path = os.path.join(npy_folder, "vibe_output.pkl")

        vibe_pred_pkl = joblib.load(pred_theta_path)

        with torch.no_grad():
            for p_idx in vibe_pred_pkl:
                # pred_theta_data = vibe_pred_pkl[p_idx]['pose'][:600] # T X 72
                pred_theta_data = vibe_pred_pkl[p_idx]['pose'] # T X 72
        
                timesteps, _ = pred_theta_data.shape
        
                bs = 1

                # Process predicted results from other methods as input to encoder
                pred_aa_data = torch.from_numpy(pred_theta_data).float().cuda() # T X 72
                pred_aa_data = pred_aa_data[None, :, :] # 1 X T X 72
                pred_cont6DRep, pred_rot_mat, pred_pose_pos = self.aa2others(pred_aa_data) # 
                # 1 X T X (24*6), 1 X T X (24*3*3), 1 X T X (24*3)

                # Process sequence with our model in sliding window fashion, use the centering frame strategy
                window_size = self.max_timesteps
                center_frame_start_idx = self.max_timesteps // 2 - 1
                center_frame_end_idx = self.max_timesteps // 2 - 1
                # Options: 7, 7; 
                overlap_len = window_size - (center_frame_end_idx-center_frame_start_idx+1)
                stride = window_size - overlap_len
                our_pred_6d_out_seq = None # T X 24 X 6
                
                pred_6d_rot = pred_cont6DRep.view(bs, timesteps, 24, 6)
                for t_idx in range(0, timesteps-window_size+1, stride):
                    curr_encoder_input = pred_6d_rot[:, t_idx:t_idx+window_size, :, :].cuda() # bs(1) X 16 X 24 X 6
                    our_rec_6d_out = self.get_mean_rec_res_w_6d_input(curr_encoder_input) # bs(1) X T(16) X 24 X 6(our 6d format)
                    if t_idx == 0:
                        # The beginning part, we take all the frames before center
                        our_pred_6d_out_seq = our_rec_6d_out.squeeze(0)[:center_frame_end_idx+1, :, :] 
                    elif t_idx == timesteps-window_size:
                        # Handle the last window in the end, take all the frames after center_start to make the videl length same as input
                        our_pred_6d_out_seq = torch.cat((our_pred_6d_out_seq, \
                            our_rec_6d_out[0, center_frame_start_idx:, :, :]), dim=0)
                    else:
                        our_pred_6d_out_seq = torch.cat((our_pred_6d_out_seq, \
                            our_rec_6d_out[0, center_frame_start_idx:center_frame_end_idx+1, :, :]), dim=0)
            
                # Use same skeleton for visualization
                pred_fk_pose = self.fk_layer(pred_6d_rot.squeeze(0)) # T X 24 X 3
                our_fk_pose = self.fk_layer(our_pred_6d_out_seq) # T X 24 X 3

                our_fk_pose[:, :, 0] += 1

                concat_seq_cmp = torch.cat((pred_fk_pose[None, :, :, :], our_fk_pose[None, :, :, :]), dim=0) # 2 X T X 24 X 3
                # Visualize single seq           
                show3Dpose_animation_multiple(concat_seq_cmp.data.cpu().numpy(), image_directory, \
                0, "cmp_vibe_ours_dance_vis", str(p_idx))

                # Save rotation matrix to numpy
                our_pred_rot_mat = my_tools.rotation_matrix_from_ortho6d(our_pred_6d_out_seq.view(-1, 6)) # (T*24) X 3 X 3(our_pred_6d_out_seq)
                our_pred_rot_mat = our_pred_rot_mat.view(-1, 24, 3, 3) # T X 24 X 3 X 3

                # T X 24 X 3 X 3
                dest_our_rot_npy_path = os.path.join(image_directory, str(p_idx)+"_our_rot_mat.npy")
                np.save(dest_our_rot_npy_path, our_pred_rot_mat.data.cpu().numpy())
            
                vibe_pred_rot_mat = pred_rot_mat.squeeze(0).view(-1, 24, 3, 3)
                # T X 24 X 3 X 3
                dest_vibe_rot_npy_path = os.path.join(image_directory, str(p_idx)+"_vibe_rot_mat.npy")
                np.save(dest_vibe_rot_npy_path, vibe_pred_rot_mat.data.cpu().numpy())

    def cal_l2_dist(self, pred, gt):
        loss = (pred-gt)**2

        return loss.mean() 

    def load_amass_test_data_singe_window(self, num_seq):
        root_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/utils/data"

        data_folder = os.path.join(root_folder, "for_all_data_motion_model")
        rot_npy_folder = os.path.join(root_folder, "processed_all_amass_data")
       
        test_json = os.path.join(data_folder, "val_all_amass_motion_data.json")
        test_json_data = json.load(open(test_json, 'r'))
        
        ori_pose_seq_data_list = []

        k_list = list(test_json_data.keys())
        selected_k_list = k_list[:num_seq]
        seq_cnt = 0
        for k in selected_k_list:
            v_name = test_json_data[k]
            rot_npy_path = os.path.join(rot_npy_folder, v_name)
            ori_pose_seq_data = np.load(rot_npy_path) # T X n_dim
            
            timesteps = ori_pose_seq_data.shape[0]

            if timesteps >= self.max_timesteps and "HumanEva" in v_name:
                # random_t_idx = random.sample(list(range(timesteps-self.max_timesteps+1)), 1)[0]
                random_t_idx = 0
                end_t_idx = random_t_idx + self.max_timesteps - 1

                ori_pose_seq_data = torch.from_numpy(ori_pose_seq_data).float()

                seq_6d_data = ori_pose_seq_data[random_t_idx:end_t_idx+1, :24*6] # T X (24*6)
                # rot_mat_data = ori_pose_seq_data[random_t_idx:end_t_idx, 24*6:24*6+24*3*3] # T X (24*3*3)

                # ori_pose_seq_data_list.append(rot_mat_data.view(self.train_seq_len, 24, 3, 3))
                ori_pose_seq_data_list.append(seq_6d_data.view(self.max_timesteps, 24, 6)) # T X 24 X 6

                seq_cnt += 1

            if seq_cnt >= num_seq:
                break; 

        ori_pose_seq_data_list = torch.stack(ori_pose_seq_data_list)

        return ori_pose_seq_data_list # K X w X 24 X 6

    def slerp_baseline_for_interpolation(self, rot_data, temporal_mask):
        # rot_data: T X 24 X 3 X 3
        # temporal_mask: T
        ori_timeteps = rot_data.size()[0]
        num_joints = rot_data.size()[1]

        temporal_idx = torch.nonzero(temporal_mask).squeeze(-1) # K 
        # print("rot_data:{0}".format(rot_data.size()))
        # print("temporal idx:{0}".format(temporal_idx.size()))
        if temporal_idx[-1] != ori_timeteps - 1:
            tmp_end_val = torch.zeros(1)
            tmp_end_val[0] = ori_timeteps - 1
            temporal_idx = torch.cat((temporal_idx, tmp_end_val.cuda().long()), dim=0)

        selected_rot_data = torch.index_select(rot_data, 0, temporal_idx) # K X 24 X 3 X 3 

        from scipy.spatial.transform import Rotation as R
        from scipy.spatial.transform import Slerp

        rot_res_list = []
        for joint_idx in range(num_joints):
            rot_arr = selected_rot_data[:, joint_idx, :, :].data.cpu().numpy() # K X 3 X 3
            key_rots = R.from_dcm(rot_arr) 
            
            key_times = temporal_idx.data.cpu().numpy().tolist()

            slerp = Slerp(key_times, key_rots)

            times = list(range(ori_timeteps))

            interp_rots = slerp(times) 

            interp_rot_mat = interp_rots.as_dcm() # T X 3 X 3
            interp_rot_mat = torch.from_numpy(interp_rot_mat).float()
            rot_res_list.append(interp_rot_mat)

        rot_res_list = torch.stack(rot_res_list)
        rot_res_list = rot_res_list.transpose(0, 1) # T X 24 X 3 X 3

        return rot_res_list.cuda()

    def lerp_root_trajectory(self, root_trajectory, temporal_mask):
        # root_trajcetory: T X 3
        # temporal_mask: T
        timesteps = root_trajectory.size()[0]
        times = list(range(timesteps))

        temporal_idx = torch.nonzero(temporal_mask).squeeze(-1) # K 
        selected_root_data = torch.index_select(root_trajectory, 0, temporal_idx) # K X 3 

        num_dim = selected_root_data.size()[1]
        lerp_res = np.zeros((timesteps, 3)) # T X 3
        for dim_idx in range(num_dim):
            lerp_res[:, dim_idx] = np.interp(times, temporal_idx.data.cpu().numpy(), selected_root_data.cpu().numpy()[:, dim_idx])

        lerp_res = torch.from_numpy(lerp_res).float()

        return lerp_res #  T X 3

    def cal_key_frame_root_loss(self, pred_out_seq, target_seq, temporal_mask):
        # pred_out_seq: T X 3
        # target_seq: T X 3
        # temporal_mask: T X 24
        
        temporal_idx = torch.nonzero(temporal_mask[0]).squeeze(-1) # K 
        selected_pred_data = torch.index_select(pred_out_seq, 0, temporal_idx) # K X 3
        relative_pred_seq = selected_pred_data[1:] - selected_pred_data[:-1] # (K-1) X 3

        selected_gt_data = torch.index_select(target_seq, 0, temporal_idx) # K X 3
        relative_gt_seq = selected_gt_data[1:] - selected_gt_data[:-1] # (K-1) X 3

        loss = self.l2_criterion(relative_pred_seq, relative_gt_seq)

        return loss 

    def load_amass_test_data_w_trajectory(self, max_num_seq=None):
        root_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/utils/data"

        data_folder = os.path.join(root_folder, "for_all_data_motion_model")
        rot_npy_folder = os.path.join(root_folder, "processed_all_amass_data")
       
        test_json = os.path.join(data_folder, "test_all_amass_motion_data.json")
        test_json_data = json.load(open(test_json, 'r'))
        seq_cnt = 0
        batch_num = max_num_seq

        ori_pose_seq_data_list = []
        root_v_list = []
        root_trans_list = []
        v_name_list = []
        for k in test_json_data:
            v_name = test_json_data[k]
            rot_npy_path = os.path.join(rot_npy_folder, v_name)
            ori_pose_seq_data = np.load(rot_npy_path) # T X n_dim
            
            if ori_pose_seq_data.shape[0] >= self.max_timesteps and ("dance" in v_name):
                ori_pose_seq_data = torch.from_numpy(ori_pose_seq_data).float()
                rot_mat_data = ori_pose_seq_data[:, 24*6:24*6+24*3*3] # T X (24*3*3)
                root_v_data = ori_pose_seq_data[:, -3:] # T X 3

                timesteps = rot_mat_data.size()[0] 

                root_trans = torch.zeros(timesteps, 3) # T X 3
                for t_idx in range(1, timesteps):
                    root_trans[t_idx] = root_trans[t_idx-1] + root_v_data[t_idx] # T X 3

                ori_pose_seq_data_list.append(rot_mat_data.view(timesteps, 24, 3, 3))
                root_v_list.append(root_v_data)
                root_trans_list.append(root_trans)

                seq_cnt += 1

                v_name_list.append(v_name)     

            if max_num_seq is not None:
                if seq_cnt >= batch_num:
                    break;

        print("Total seq:{0}".format(seq_cnt))

        return v_name_list, ori_pose_seq_data_list, root_v_list, root_trans_list

    def try_final_long_seq_generation(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False, trajectory_trainer=None): # For comparison with the partial-body model paper.  
        # json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_single_window_w_trajectory(100)

        json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_single_window_w_trajectory(only_try_selected=True)

        try_cnt_each_seq = 1
        f_cnt = 0
        num_windows = 5
        overlap = 10
        for f_name in json_files:
            for try_idx in range(try_cnt_each_seq):  
                seq_rot_mat = amass_rot_mat_data_list[f_cnt].float().cuda() # T X 24 X 3 X 3
                rotMatrices = seq_rot_mat # T X 24 X 3 X 3
                # Convert rotation matrix to 6D representation
                input_cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # T X 24 X 2 X 3
                input_cont6DRep = input_cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # T X 24 X 6
                gt_fk_pose = self.fk_layer(rotMatrices) # T X 24 X 3

                timesteps = gt_fk_pose.size()[0]

                whole_6d_seq = input_cont6DRep
                whole_rot_mat_seq = rotMatrices
                whole_fk_pose = gt_fk_pose 
                for w_idx in range(num_windows):
                    # Generate temporal mask 
                    temporal_mask = torch.zeros(timesteps)
                    temporal_mask[:overlap] = 1
                    temporal_mask = temporal_mask.cuda() # T 

                    padding_steps = timesteps - overlap
                    curr_target_cont6d_data = torch.cat((whole_6d_seq[-overlap:], torch.zeros(padding_steps, 24, 6).cuda()), dim=0)[None] # 1 X T' X 24 X 6
                    curr_target_rot_data = torch.cat((whole_rot_mat_seq[-overlap:], torch.zeros(padding_steps, 24, 3, 3).cuda()), dim=0)[None] # 1 X T' X 24 X 3 X 3
                    curr_target_coord_data = torch.cat((whole_fk_pose[-overlap:], torch.zeros(padding_steps, 24, 3).cuda()), dim=0)[None] # 1 X T' X 24 X 3
                    curr_target_mask = temporal_mask[None, :, None].repeat(1, 1, 24)  # 1 X T' X 24

                    shallow_k_edges = 14
                    deep_k_edges = 7

                    bs = 1
                    
                    useless_z_2_data = torch.zeros(bs, 9, self.latent_d).cuda()
                    useless_z_3_data = torch.zeros(bs, 7, self.latent_d).cuda()

                    # Keep same deep latent vector, only change shallow latent vector
                    deep_sampled_z_data = torch.randn(bs, deep_k_edges, self.latent_d).cuda()
                    shallow_sampled_z_data = torch.randn(bs, shallow_k_edges, self.shallow_latent_d).cuda()

                    self.z_vec_list = []
                    
                    self.z_vec_list.append(nn.Parameter(shallow_sampled_z_data.data))
                    self.z_vec_list.append(nn.Parameter(useless_z_2_data.data))
                    self.z_vec_list.append(nn.Parameter(useless_z_3_data.data))
                    self.z_vec_list.append(nn.Parameter(deep_sampled_z_data.data))

                    target_z_reg_list = []
                    target_z_reg_list.append(torch.from_numpy(shallow_sampled_z_data.data.cpu().numpy()).float().cuda())
                    target_z_reg_list.append(torch.from_numpy(useless_z_2_data.data.cpu().numpy()).float().cuda())
                    target_z_reg_list.append(torch.from_numpy(useless_z_3_data.data.cpu().numpy()).float().cuda())
                    target_z_reg_list.append(torch.from_numpy(deep_sampled_z_data.data.cpu().numpy()).float().cuda())

                    if self.hp['optimize_decoder']:
                        curr_decoder = copy.deepcopy(self.dec)
                        self.gen_opt_for_decoder = torch.optim.Adam(list(curr_decoder.parameters()), lr=self.hp['opt_lr']*0.001, weight_decay=self.hp["weight_decay"])
                        self.gen_scheduler_for_decoder = get_opt_scheduler(self.gen_opt_for_decoder, self.hp)
                    # else:
                    # self.gen_opt = torch.optim.Adam([self.z_vector]+list(self.dec.parameters()), lr=self.hp['opt_lr'], weight_decay=self.hp["weight_decay"])
                    self.gen_opt = torch.optim.Adam(self.z_vec_list, lr=self.hp['opt_lr'], weight_decay=self.hp["weight_decay"])
                    self.gen_scheduler = get_opt_scheduler(self.gen_opt, self.hp)

                    min_loss = 9999999
                    min_loss_out_6d = None 
                    min_loss_out_rot_mat = None 
                    min_loss_out_pose_pos = None 
                    
                    for i in range(self.hp["opt_it"]):
                        if self.hp['optimize_decoder']:
                            opt_out_6d, opt_out_rot_mat, opt_out_pose_pos, \
                            _, _, _, _ = self._decode_w_given_decoder(self.z_vec_list, curr_decoder)
                        else:
                            opt_out_6d, opt_out_rot_mat, opt_out_pose_pos, \
                            _, _, _, _ = self._decode(self.z_vec_list, 1, 4)
                            # m X w X 24 X 6
                        
                        l_rec_6d, saved_6d_loss = self.l2_masked_criterion(opt_out_6d, curr_target_cont6d_data, curr_target_mask) # 1 X w X 24 X 6
                        l_rec_rot_mat, saved_mat_loss = self.l2_masked_criterion(opt_out_rot_mat, curr_target_rot_data, curr_target_mask) # 1 X w X 24 X 3 X 3
                        l_rec_pose, saved_pose_loss = self.l2_masked_criterion(opt_out_pose_pos, curr_target_coord_data, curr_target_mask) # 1 X w X 24 X 3

                        l_reg = self.l2_criterion(self.z_vec_list[0], target_z_reg_list[0]) + \
                            self.l2_criterion(self.z_vec_list[3], target_z_reg_list[3])

                        if self.hp['optimize_decoder']:
                            l_reg_decoder = torch.zeros(1).cuda()
                            for name, params in curr_decoder.named_parameters():
                                l_reg_decoder += self.l2_criterion(params, self.dec.state_dict()[name])
                        else:
                            l_reg_decoder = torch.zeros(1).cuda() 

                    
                        l_rec_trajectory = torch.zeros(1).cuda()

                        l_total = self.hp['rec_6d_w'] * l_rec_6d + self.hp['rec_rot_w'] * l_rec_rot_mat + self.hp['rec_pose_w'] * l_rec_pose +  \
                            self.hp['reg_w'] * l_reg + self.hp['reg_w_decoder'] * l_reg_decoder
                    
                        if i % 1 == 0:
                            print("Optimizing %d iter... Rec6D loss %f, RecRot loss %f, RecPose loss %f, Reg loss %f, Decoder Reg loss %f, \
                                Trajectory loss %f, Total loss %f."%(i, \
                                l_rec_6d, l_rec_rot_mat, l_rec_pose, l_reg, l_reg_decoder, l_rec_trajectory, l_total))
                        # import pdb
                        # pdb.set_trace()
                    
                        prev_epochs = 50
                        if self.hp['optimize_decoder']:
                            if i > prev_epochs:
                                self.gen_opt_for_decoder.zero_grad()
                            else:
                                self.gen_opt.zero_grad()
                        else:
                            self.gen_opt.zero_grad()

                        l_total.backward() 
                        
                        if self.hp['optimize_decoder']:
                            if i > prev_epochs:
                                self.gen_opt_for_decoder.step()
                                self.gen_scheduler_for_decoder.step()
                            else:
                                self.gen_opt.step()
                                self.gen_scheduler.step()
                        else:
                            self.gen_opt.step()
                            self.gen_scheduler.step()

                        if l_total < min_loss:
                            min_loss = l_total
                            min_loss_out_6d = opt_out_6d
                            min_loss_out_rot_mat = opt_out_rot_mat
                            min_loss_out_pose_pos = opt_out_pose_pos

                    opt_our_pred_6d_seq = opt_out_6d[0, overlap:] # t' X 24 X 6
                    opt_our_pred_rot_mat_seq = opt_out_rot_mat[0, overlap:] # t' X 24 X 3 X 3
                    opt_our_pred_pose_seq = opt_out_pose_pos[0, overlap:] # t' X 24 X 3

                    whole_6d_seq = torch.cat((whole_6d_seq.detach(), opt_our_pred_6d_seq.detach()), dim=0)
                    whole_rot_mat_seq = torch.cat((whole_rot_mat_seq.detach(), opt_our_pred_rot_mat_seq.detach()), dim=0)
                    whoel_fk_pose = torch.cat((whole_fk_pose.detach(), opt_our_pred_pose_seq.detach()), dim=0) 

                # Save all the results 
                dest_directory = os.path.join(dest_image_directory, f_name, str(datetime.datetime.now()))
                if not os.path.exists(dest_directory):
                    os.makedirs(dest_directory)

                if trajectory_trainer is not None:
                    encoder_input = whole_6d_seq[None] # 1 X T X 24 X 6
                    pred_seq_w_root_trans_res = trajectory_trainer.sampled_seq_test(encoder_input) # T X 1 X 24 X 3

                    seq_for_vis = pred_seq_w_root_trans_res.transpose(0, 1) # 1 X T X 24 X 3
                    show3Dpose_animation(seq_for_vis.data.cpu().numpy(), dest_directory, \
                                0, "0", f_name.replace(".json", ""), use_amass=use_amass_data)

                    # Save translation info to npy 
                    dest_trans_opt_npy_path = os.path.join(dest_directory, str(0), "0", \
                        f_name.replace(".json", "")+"_root_trans_opt_res.npy")

                    np.save(dest_trans_opt_npy_path, pred_seq_w_root_trans_res.squeeze(1).data.cpu().numpy()) # T X 24 X 3
                   
                    # Save rotation info to npy 
                    dest_opt_npy_path = os.path.join(dest_directory, str(0), "0", \
                        f_name.replace(".json", "")+"_rot_opt_res.npy")
                    
                    opt_rot_mat_for_save = whole_rot_mat_seq.data.cpu().numpy() # T X 24 X 3 X 3
                    np.save(dest_opt_npy_path, opt_rot_mat_for_save)

            f_cnt += 1
        
    def final_long_seq_try_interpolation(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False, trajectory_trainer=None): # For comparison with the partial-body model paper.  
        # json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_single_window_w_trajectory(100)
        json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_w_trajectory()

        # json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_single_window_w_trajectory(only_try_selected=True)

        f_cnt = 0
        for f_name in json_files:
            seq_rot_mat = amass_rot_mat_data_list[f_cnt].float().cuda() # T X 24 X 3 X 3
            rotMatrices = seq_rot_mat # T X 24 X 3 X 3

            # Convert rotation matrix to 6D representation
            input_cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # T X 24 X 2 X 3
            input_cont6DRep = input_cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # T X 24 X 6

            gt_fk_pose = self.fk_layer(rotMatrices) # T X 24 X 3

            timesteps = gt_fk_pose.size()[0]

            num_windows = timesteps // self.max_timesteps

            whole_our_6d_seq = None
            whole_our_rot_mat_seq = None 
            whole_our_fk_pose_seq = None

            whole_slerp_6d_seq = None
            whole_slerp_rot_mat_seq = None
            whole_slerp_fk_pose_seq = None 
            for t_idx in range(num_windows):
                # Generate temporal mask 
                temporal_mask = np.zeros(self.max_timesteps)
                temporal_mask[::self.hp['interpolation_window']] = 1
                # Make sure the final timestep is 1 for slerp baseline
                temporal_mask[-1] = 1
                temporal_mask = torch.from_numpy(temporal_mask)
                temporal_mask = temporal_mask.cuda() # T' (the window size) 

                # Get slerped results for baseline
                slerped_rot_mat = self.slerp_baseline_for_interpolation(rotMatrices[t_idx*self.max_timesteps:(t_idx+1)*self.max_timesteps], \
                    temporal_mask) # T X 24 X 3 X 3

                curr_target_cont6d_data = input_cont6DRep[None][:, t_idx*self.max_timesteps:(t_idx+1)*self.max_timesteps, :, :] # 1 X T' X 24 X 6
                curr_target_rot_data = rotMatrices[None][:, t_idx*self.max_timesteps:(t_idx+1)*self.max_timesteps, :, :, :] # 1 X T' X 24 X 3 X 3
                curr_target_coord_data = gt_fk_pose[None][:, t_idx*self.max_timesteps:(t_idx+1)*self.max_timesteps, :, :] # 1 X T' X 24 X 3
                curr_target_mask = temporal_mask[None, :, None].repeat(1, 1, 24)  # 1 X T' X 24
                target_fk_pose = gt_fk_pose[t_idx*self.max_timesteps:(t_idx+1)*self.max_timesteps, :, :] # T' X 24 X 3

                shallow_k_edges = 14
                deep_k_edges = 7

                bs = 1
                
                useless_z_2_data = torch.zeros(bs, 9, self.latent_d).cuda()
                useless_z_3_data = torch.zeros(bs, 7, self.latent_d).cuda()

                # Keep same deep latent vector, only change shallow latent vector
                deep_sampled_z_data = torch.randn(bs, deep_k_edges, self.latent_d).cuda()
                shallow_sampled_z_data = torch.randn(bs, shallow_k_edges, self.shallow_latent_d).cuda()

                self.z_vec_list = []
                
                self.z_vec_list.append(nn.Parameter(shallow_sampled_z_data.data))
                self.z_vec_list.append(nn.Parameter(useless_z_2_data.data))
                self.z_vec_list.append(nn.Parameter(useless_z_3_data.data))
                self.z_vec_list.append(nn.Parameter(deep_sampled_z_data.data))

                target_z_reg_list = []
                target_z_reg_list.append(torch.zeros_like(shallow_sampled_z_data).float().cuda())
                target_z_reg_list.append(torch.zeros_like(useless_z_2_data).float().cuda())
                target_z_reg_list.append(torch.zeros_like(useless_z_3_data).float().cuda())
                target_z_reg_list.append(torch.zeros_like(deep_sampled_z_data).float().cuda())

                if self.hp['optimize_decoder']:
                    curr_decoder = copy.deepcopy(self.dec)
                    self.gen_opt_for_decoder = torch.optim.Adam(list(curr_decoder.parameters()), lr=self.hp['opt_lr']*0.001, weight_decay=self.hp["weight_decay"])
                    self.gen_scheduler_for_decoder = get_opt_scheduler(self.gen_opt_for_decoder, self.hp)
                # else:
                # self.gen_opt = torch.optim.Adam([self.z_vector]+list(self.dec.parameters()), lr=self.hp['opt_lr'], weight_decay=self.hp["weight_decay"])
                self.gen_opt = torch.optim.Adam(self.z_vec_list, lr=self.hp['opt_lr'], weight_decay=self.hp["weight_decay"])
                self.gen_scheduler = get_opt_scheduler(self.gen_opt, self.hp)

                min_loss = 9999999
                min_loss_out_6d = None 
                min_loss_out_rot_mat = None 
                min_loss_out_pose_pos = None 
                
                for i in range(self.hp["opt_it"]):
                    if self.hp['optimize_decoder']:
                        opt_out_6d, opt_out_rot_mat, opt_out_pose_pos, \
                        _, _, _, _ = self._decode_w_given_decoder(self.z_vec_list, curr_decoder)
                    else:
                        opt_out_6d, opt_out_rot_mat, opt_out_pose_pos, \
                        _, _, _, _ = self._decode(self.z_vec_list, 1, 4)
                        # m X w X 24 X 6
                    
                    l_rec_6d, saved_6d_loss = self.l2_masked_criterion(opt_out_6d, curr_target_cont6d_data, curr_target_mask) # 1 X w X 24 X 6
                    l_rec_rot_mat, saved_mat_loss = self.l2_masked_criterion(opt_out_rot_mat, curr_target_rot_data, curr_target_mask) # 1 X w X 24 X 3 X 3
                    l_rec_pose, saved_pose_loss = self.l2_masked_criterion(opt_out_pose_pos, curr_target_coord_data, curr_target_mask) # 1 X w X 24 X 3

                    l_reg = self.l2_criterion(self.z_vec_list[0], target_z_reg_list[0]) + \
                        self.l2_criterion(self.z_vec_list[3], target_z_reg_list[3])

                    if self.hp['optimize_decoder']:
                        l_reg_decoder = torch.zeros(1).cuda()
                        for name, params in curr_decoder.named_parameters():
                            l_reg_decoder += self.l2_criterion(params, self.dec.state_dict()[name])
                    else:
                        l_reg_decoder = torch.zeros(1).cuda() 

                    # Introduce trajectory constraint for key frames 
                    if self.hp['optimize_trajectory']:
                        encoder_input = opt_out_6d # m(1) X w X 24 X 6
                        pred_seq_w_root_trans_res = trajectory_trainer.sampled_seq_test(encoder_input) # w X m(1) X 24 X 3
                        pred_root_trans = pred_seq_w_root_trans_res.squeeze(1)[:, 0, :] # T X 3

                        gt_trans = root_trans_list[f_cnt].cuda() # T X 3

                        # Keep relative distance the same for key frames 
                        l_rec_trajectory = self.cal_key_frame_root_loss(pred_root_trans, \
                            gt_trans, curr_target_mask.squeeze(0))
                    else:
                        l_rec_trajectory = torch.zeros(1).cuda()

                    l_total = self.hp['rec_6d_w'] * l_rec_6d + self.hp['rec_rot_w'] * l_rec_rot_mat + self.hp['rec_pose_w'] * l_rec_pose +  \
                        self.hp['reg_w'] * l_reg + self.hp['reg_w_decoder'] * l_reg_decoder + self.hp['reg_w_trajectory'] * l_rec_trajectory 
                
                    if i % 1 == 0:
                        print("Optimizing %d iter... Rec6D loss %f, RecRot loss %f, RecPose loss %f, Reg loss %f, Decoder Reg loss %f, \
                            Trajectory loss %f, Total loss %f."%(i, \
                            l_rec_6d, l_rec_rot_mat, l_rec_pose, l_reg, l_reg_decoder, l_rec_trajectory, l_total))
                    # import pdb
                    # pdb.set_trace()
                    
                    prev_epochs = 50
                    if self.hp['optimize_decoder']:
                        if i > prev_epochs:
                            self.gen_opt_for_decoder.zero_grad()
                        else:
                            self.gen_opt.zero_grad()
                    else:
                        self.gen_opt.zero_grad()

                    l_total.backward() 
                    
                    if self.hp['optimize_decoder']:
                        if i > prev_epochs:
                            self.gen_opt_for_decoder.step()
                            self.gen_scheduler_for_decoder.step()
                        else:
                            self.gen_opt.step()
                            self.gen_scheduler.step()
                    else:
                        self.gen_opt.step()
                        self.gen_scheduler.step()

                    if l_total < min_loss:
                        min_loss = l_total
                        min_loss_out_6d = opt_out_6d
                        min_loss_out_rot_mat = opt_out_rot_mat
                        min_loss_out_pose_pos = opt_out_pose_pos

                opt_our_pred_6d_seq = opt_out_6d # 1 X window_size X 24 X 6
                opt_our_pred_rot_mat_seq = opt_out_rot_mat # 1 X window_size X 24 X 3 X 3
                opt_our_pred_pose_seq = opt_out_pose_pos # 1 X window_size X 24 X 3

                opt_our_fk_pose = opt_our_pred_pose_seq # K X T X 24 X 3

                slerped_fk_pose = self.fk_layer(slerped_rot_mat.view(-1, 24, 3, 3)).view(-1, 24, 3) # T' X 24 X 3
                slerped_fk_pose = slerped_fk_pose[None] # 1 X T' X 24 X 3
                slerped_rot_mat = slerped_rot_mat[None] # 1 X T' X 24 X 3 X 3

                if self.hp['replace_frame_with_gt']:
                    replace_mask_6d = temporal_mask[None, :, None, None].float()
                    replace_mask_rot_mat = temporal_mask[None, :, None, None, None].float()
                    replace_mask_fk_pose = temporal_mask[None, :, None, None].float()

                    opt_our_pred_6d_seq = replace_mask_6d * curr_target_cont6d_data + (1-replace_mask_6d) * opt_our_pred_6d_seq 
                    opt_our_pred_rot_mat_seq = replace_mask_rot_mat * curr_target_rot_data + (1-replace_mask_rot_mat) * opt_our_pred_rot_mat_seq
                    opt_our_fk_pose = replace_mask_fk_pose * curr_target_coord_data + (1-replace_mask_fk_pose) * opt_our_fk_pose

                # Concat result to existing sequence 
                if t_idx == 0:
                    whole_our_6d_seq = opt_our_pred_6d_seq.squeeze(0).detach()
                    whole_our_rot_mat_seq = opt_our_pred_rot_mat_seq.squeeze(0).detach()
                    whole_our_fk_pose_seq = opt_our_fk_pose.squeeze(0).detach()

                    whole_slerp_rot_mat_seq = slerped_rot_mat.squeeze(0).detach()
                    whole_slerp_fk_pose_seq = slerped_fk_pose.squeeze(0).detach()

                else:
                    whole_our_6d_seq = torch.cat((whole_our_6d_seq, opt_our_pred_6d_seq.squeeze(0).detach()), dim=0)
                    whole_our_rot_mat_seq = torch.cat((whole_our_rot_mat_seq, opt_our_pred_rot_mat_seq.squeeze(0).detach()), dim=0)
                    whole_our_fk_pose_seq = torch.cat((whole_our_fk_pose_seq, opt_our_fk_pose.squeeze(0).detach()), dim=0)

                    whole_slerp_rot_mat_seq = torch.cat((whole_slerp_rot_mat_seq, slerped_rot_mat.squeeze(0).detach()), dim=0)
                    whole_slerp_fk_pose_seq = torch.cat((whole_slerp_fk_pose_seq, slerped_fk_pose.squeeze(0).detach()), dim=0)

            if trajectory_trainer is not None:
                encoder_input = whole_our_6d_seq[None] # 1 X T X 24 X 6
                # For debug 
                # encoder_input = self.rot_mat_to_6d(gt_rot_mat[k_idx])[None] # 1 X T X 24 X 6, debug finished, trajectory is good with gt local motion
                pred_seq_w_root_trans_res = trajectory_trainer.sampled_seq_test(encoder_input).squeeze(1) # T X 24 X 3

                actual_len = pred_seq_w_root_trans_res.size()[0]
                gt_root_trans_res = root_trans_list[f_cnt][:actual_len] # T X 3
                gt_seq_w_root_trans_res = torch.from_numpy(gt_fk_pose[:actual_len].data.cpu().numpy()) # T X 24 X 3
                gt_seq_w_root_trans_res += gt_root_trans_res[:, None, :]

                slerp_encoder_input = self.rot_mat_to_6d(whole_slerp_rot_mat_seq)[None] # 1 X T x 24 X 6
                lerp_seq_w_root_trans_res = trajectory_trainer.sampled_seq_test(slerp_encoder_input).squeeze(1) # T X 24 X 3
                   
                # Save all the results 
                dest_directory = os.path.join(dest_image_directory, f_name, str(datetime.datetime.now()))
                if not os.path.exists(dest_directory):
                    os.makedirs(dest_directory)

                slerped_pose_seq_for_vis = lerp_seq_w_root_trans_res[None].data.cpu().numpy() # 1 X T X 24 X 3
                slerped_pose_seq_for_vis[:, :, :, 0] += 1

                opt_our_rec_pose_seq_for_vis = pred_seq_w_root_trans_res[None].data.cpu().numpy() # 1 X T X 24 X 3
                opt_our_rec_pose_seq_for_vis[:, :, :, 0] += 2

                gt_fk_pose_for_vis = gt_seq_w_root_trans_res[None].data.cpu().numpy()
                # gt_fk_pose_for_vis[:, :, :, 0] += 0

                if trajectory_trainer is not None:
                    tag = "temporal_interp_w_trajectory"
                    pred_seq_w_root_trans_for_vis = pred_seq_w_root_trans_res[None].data.cpu().clone() # 1 X T X 24 X 3
                    gt_seq_w_root_trans_for_vis = gt_seq_w_root_trans_res[None].data.cpu().clone()
                    lerp_seq_w_root_trans_for_vis = lerp_seq_w_root_trans_res[None].data.cpu().clone()

                    pred_seq_w_root_trans_for_vis[:, :, :, 0] += 2
                    lerp_seq_w_root_trans_for_vis[:, :, :, 0] += 1

                    concat_seq_for_vis = torch.cat((pred_seq_w_root_trans_for_vis, lerp_seq_w_root_trans_for_vis, gt_seq_w_root_trans_for_vis), dim=0) # 3 X T X 24 X 3
    
                    show3Dpose_animation_multiple(concat_seq_for_vis.data.cpu().numpy(), dest_directory, \
                        0, tag, f_name.replace(".json", ""), use_amass=use_amass_data)
                    
                    target_dir = os.path.join(dest_directory, str(0), tag)
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)

                    # Save translation info to npy 
                    dest_trans_opt_npy_path = os.path.join(dest_directory, str(0), tag, \
                        f_name.replace(".json", "")+"_root_trans_opt_res.npy")
                    dest_trans_gt_npy_path = os.path.join(dest_directory, str(0), tag, \
                        f_name.replace(".json", "")+"_root_trans_gt_res.npy")
                    dest_trans_slerped_npy_path = os.path.join(dest_directory, str(0), tag, \
                        f_name.replace(".json", "")+"_root_trans_slerped_res.npy")

                    np.save(dest_trans_opt_npy_path, pred_seq_w_root_trans_res.data.cpu().numpy()) # T X 24 X 3
                    np.save(dest_trans_gt_npy_path, gt_seq_w_root_trans_res.data.cpu().numpy())
                    np.save(dest_trans_slerped_npy_path, lerp_seq_w_root_trans_res.data.cpu().numpy())

                    # Save rotation info to npy 
                    dest_opt_npy_path = os.path.join(dest_directory, str(0), tag, \
                        f_name.replace(".json", "")+"_rot_opt_res.npy")
                    dest_gt_npy_path = os.path.join(dest_directory, str(0), tag, \
                        f_name.replace(".json", "")+"_rot_gt_res.npy")
                    dest_slerped_npy_path = os.path.join(dest_directory, str(0), tag, \
                        f_name.replace(".json", "")+"_rot_slerped_res.npy")
                    
                    opt_rot_mat_for_save = whole_our_rot_mat_seq.data.cpu().numpy() # T X 24 X 3 X 3
                    gt_rot_mat_for_save = rotMatrices[:actual_len].data.cpu().numpy() # T X 24 X 3 X 3
                    slerped_rot_mat_for_save = whole_slerp_rot_mat_seq.data.cpu().numpy() # T X 24 X 3 X 3

                    np.save(dest_opt_npy_path, opt_rot_mat_for_save)
                    np.save(dest_gt_npy_path, gt_rot_mat_for_save)
                    np.save(dest_slerped_npy_path, slerped_rot_mat_for_save)

            f_cnt += 1

    def final_motion_completion_long_seq(self, dest_image_directory, input_gt=True, gen_vis=False, use_amass_data=False, trajectory_trainer=None): # For comparison with the partial-body model paper.  
        # json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_single_window_w_trajectory(200) # For validation200
        json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_w_trajectory() # For testing set

        if not use_amass_data:
            use_partial = False

            if use_partial:
                # Load partial human results
                json_folder = ""
                json_files = os.listdir(json_folder)
            else:
                # Load vibe results
                # npy_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/for_supp_outputs/final_walk/final_walk"
                npy_folder = "/glab2/data/Users/jiaman/adobe/github/VIBE/talk_test_output/test_tmp/test_tmp"
                pred_theta_path = os.path.join(npy_folder, "vibe_output.pkl")
                vibe_pred_pkl = joblib.load(pred_theta_path)
                p_idx = 1
                # pred_theta_data = vibe_pred_pkl[p_idx]['pose'][:600] # T X 72
                pred_theta_data = vibe_pred_pkl[p_idx]['pose'] # T X 72
        
                timesteps, _ = pred_theta_data.shape
                # Process predicted results from other methds as input to encoder
                pred_aa_data = torch.from_numpy(pred_theta_data).float().cuda() # T X 72
                pred_aa_data = pred_aa_data[None, :, :] # 1 X T X 72
                pred_cont6DRep, pred_rot_mat, pred_pose_pos = self.aa2others(pred_aa_data) # 
                # 1 X T X (24*6), 1 X T X (24*3*3), 1 X T X (24*3)

                pred_rot_mat = pred_rot_mat.squeeze(0).view(-1, 24, 3, 3) # T X 24 X 3 X 3

                rot_mat_npy_path = os.path.join(npy_folder, "1_our_rot_mat.npy")
                pred_rot_mat = torch.from_numpy(np.load(rot_mat_npy_path)).float()
                pred_rot_mat, _ = self.adjust_root_rot(pred_rot_mat[None])
                pred_rot_mat = pred_rot_mat.squeeze(0).cuda()
                json_files = ["vibe_res"]

        # json_files, amass_rot_mat_data_list, root_v_list, root_trans_list = self.load_amass_test_data_single_window_w_trajectory(only_try_selected=True)
        # upper_joint_list = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        # lower_joint_list = [1, 2, 4, 5, 7, 8, 10, 11]

        upper_joint_list = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        lower_joint_list = [0, 3, 6, 9, 1, 2, 4, 5, 7, 8, 10, 11]

        if self.hp['missing_upper_completion']:
            missing_lower_mask = torch.ones(24) # 24
            missing_lower_mask[upper_joint_list] = 0
            missing_lower_mask = missing_lower_mask.cuda() # 24
        elif self.hp['missing_lower_completion']:
            missing_lower_mask = torch.ones(24) # 24
            missing_lower_mask[lower_joint_list] = 0
            missing_lower_mask = missing_lower_mask.cuda() # 24

        f_cnt = 0

        for f_name in json_files:
            # seq_rot_mat = amass_rot_mat_data_list[f_cnt].float().cuda() # T X 24 X 3 X 3
            # rotMatrices = seq_rot_mat # T X 24 X 3 X 3

            if use_amass_data:
                # gt_rot_mat = amass_rot_mat_data_list[f_cnt].float().cuda() # T X 24 X 3 X 3
                # pred_rot_mat = amass_rot_mat_data_list[f_cnt].float().cuda() # T X 24 X 3 X 3
                seq_rot_mat = amass_rot_mat_data_list[f_cnt].float().cuda() # T X 24 X 3 X 3
            else:
                if use_partial:
                    json_path = os.path.join(json_folder, f_name)
                    json_data = json.load(open(json_path, 'r'))
                
                    # seq_rot_mat = torch.from_numpy(np.asarray(json_data['gt_rot_seq'])).float().cuda() # T X 24 X 3 X 3
                    
                    seq_rot_mat = torch.from_numpy(np.asarray(json_data['pred_rot_seq'])).float().squeeze(1).cuda() # T X 24 X 3 X 3
                else:
                    seq_rot_mat = pred_rot_mat

            rotMatrices = seq_rot_mat # T X 24 X 3 X 3

            # Convert rotation matrix to 6D representation
            input_cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # T X 24 X 2 X 3
            input_cont6DRep = input_cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # T X 24 X 6

            gt_fk_pose = self.fk_layer(rotMatrices) # T X 24 X 3

            timesteps = gt_fk_pose.size()[0]

            overlap = 1
            stride = self.max_timesteps - overlap 

            whole_our_6d_seq = None
            whole_our_rot_mat_seq = None 
            whole_our_fk_pose_seq = None
            for t_idx in range(0, timesteps, stride):
                if t_idx == 0:
                    temporal_mask = missing_lower_mask[None].repeat(self.max_timesteps, 1) # T' X 24
                else:
                    padding_single_mask = torch.ones(overlap, self.n_joints).float().cuda() # 1 X 24
                    temporal_mask = missing_lower_mask[None].repeat(self.max_timesteps-overlap, 1) # (T'-1) X 24
                    temporal_mask = torch.cat((padding_single_mask, temporal_mask), dim=0) # T' X 24 

                if t_idx == 0:
                    curr_target_cont6d_data = input_cont6DRep[None][:, t_idx:t_idx+self.max_timesteps, :, :] # 1 X T' X 24 X 6
                    curr_target_rot_data = rotMatrices[None][:, t_idx:t_idx+self.max_timesteps, :, :, :] # 1 X T' X 24 X 3 X 3
                    curr_target_coord_data = gt_fk_pose[None][:, t_idx:t_idx+self.max_timesteps, :, :] # 1 X T' X 24 X 3
                else:
                    curr_target_cont6d_data = input_cont6DRep[None][:, t_idx:t_idx+self.max_timesteps, :, :] # 1 X T' X 24 X 6
                    curr_target_rot_data = rotMatrices[None][:, t_idx:t_idx+self.max_timesteps, :, :, :] # 1 X T' X 24 X 3 X 3
                    curr_target_coord_data = gt_fk_pose[None][:, t_idx:t_idx+self.max_timesteps, :, :] # 1 X T' X 24 X 3

                    # Replace the half pose in first timestep with previous window's whole pose 
                    curr_target_cont6d_data[0, :overlap, :, :] = whole_our_6d_seq[-overlap:, :, :]
                    curr_target_rot_data[0, :overlap, :, :, :] = whole_our_rot_mat_seq[-overlap:, :, :, :]
                    curr_target_coord_data[0, :overlap, :, :] = whole_our_fk_pose_seq[-overlap, :, :]
                                
                if curr_target_coord_data.size()[1] != self.max_timesteps:
                    break # If the final window is not 64, quit 

                curr_target_mask = temporal_mask[None]  # 1 X T' X 24

                shallow_k_edges = 14
                deep_k_edges = 7

                bs = 1
                
                useless_z_2_data = torch.zeros(bs, 9, self.latent_d).cuda()
                useless_z_3_data = torch.zeros(bs, 7, self.latent_d).cuda()

                # Keep same deep latent vector, only change shallow latent vector
                deep_sampled_z_data = torch.randn(bs, deep_k_edges, self.latent_d).cuda()
                shallow_sampled_z_data = torch.randn(bs, shallow_k_edges, self.shallow_latent_d).cuda()

                self.z_vec_list = []
                
                self.z_vec_list.append(nn.Parameter(shallow_sampled_z_data.data))
                self.z_vec_list.append(nn.Parameter(useless_z_2_data.data))
                self.z_vec_list.append(nn.Parameter(useless_z_3_data.data))
                self.z_vec_list.append(nn.Parameter(deep_sampled_z_data.data))

                target_z_reg_list = []
                target_z_reg_list.append(torch.zeros_like(shallow_sampled_z_data).float().cuda())
                target_z_reg_list.append(torch.zeros_like(useless_z_2_data).float().cuda())
                target_z_reg_list.append(torch.zeros_like(useless_z_3_data).float().cuda())
                target_z_reg_list.append(torch.zeros_like(deep_sampled_z_data).float().cuda())
            
                if self.hp['optimize_decoder']:
                    curr_decoder = copy.deepcopy(self.dec)
                    self.gen_opt_for_decoder = torch.optim.Adam(list(curr_decoder.parameters()), lr=self.hp['opt_lr']*0.001, weight_decay=self.hp["weight_decay"])
                    self.gen_scheduler_for_decoder = get_opt_scheduler(self.gen_opt_for_decoder, self.hp)
                # else:
                # self.gen_opt = torch.optim.Adam([self.z_vector]+list(self.dec.parameters()), lr=self.hp['opt_lr'], weight_decay=self.hp["weight_decay"])
                self.gen_opt = torch.optim.Adam(self.z_vec_list, lr=self.hp['opt_lr'], weight_decay=self.hp["weight_decay"])
                self.gen_scheduler = get_opt_scheduler(self.gen_opt, self.hp)

                min_loss = 9999999
                min_loss_out_6d = None 
                min_loss_out_rot_mat = None 
                min_loss_out_pose_pos = None 
                
                for i in range(self.hp["opt_it"]):
                    if self.hp['optimize_decoder']:
                        opt_out_6d, opt_out_rot_mat, opt_out_pose_pos, \
                        _, _, _, _ = self._decode_w_given_decoder(self.z_vec_list, curr_decoder)
                    else:
                        opt_out_6d, opt_out_rot_mat, opt_out_pose_pos, \
                        _, _, _, _ = self._decode(self.z_vec_list, 1, 4)
                        # m X w X 24 X 6
                    
                    l_rec_6d, saved_6d_loss = self.l2_masked_criterion(opt_out_6d, curr_target_cont6d_data, curr_target_mask) # 1 X w X 24 X 6
                    l_rec_rot_mat, saved_mat_loss = self.l2_masked_criterion(opt_out_rot_mat, curr_target_rot_data, curr_target_mask) # 1 X w X 24 X 3 X 3
                    l_rec_pose, saved_pose_loss = self.l2_masked_criterion(opt_out_pose_pos, curr_target_coord_data, curr_target_mask) # 1 X w X 24 X 3

                    l_reg = self.l2_criterion(self.z_vec_list[0], target_z_reg_list[0]) + \
                        self.l2_criterion(self.z_vec_list[3], target_z_reg_list[3])

                    if self.hp['optimize_decoder']:
                        l_reg_decoder = torch.zeros(1).cuda()
                        for name, params in curr_decoder.named_parameters():
                            l_reg_decoder += self.l2_criterion(params, self.dec.state_dict()[name])
                    else:
                        l_reg_decoder = torch.zeros(1).cuda() 

                    l_total = self.hp['rec_6d_w'] * l_rec_6d + self.hp['rec_rot_w'] * l_rec_rot_mat + self.hp['rec_pose_w'] * l_rec_pose +  \
                        self.hp['reg_w'] * l_reg + self.hp['reg_w_decoder'] * l_reg_decoder 
                
                    if i % 1 == 0:
                        print("Optimizing %d iter... Rec6D loss %f, RecRot loss %f, RecPose loss %f, Reg loss %f, Decoder Reg loss %f, \
                            Total loss %f."%(i, \
                            l_rec_6d, l_rec_rot_mat, l_rec_pose, l_reg, l_reg_decoder, l_total))
                    # import pdb
                    # pdb.set_trace()
                    
                    prev_epochs = 100
                    if self.hp['optimize_decoder']:
                        if i > prev_epochs:
                            self.gen_opt_for_decoder.zero_grad()
                        else:
                            self.gen_opt.zero_grad()
                    else:
                        self.gen_opt.zero_grad()

                    l_total.backward() 
                    
                    if self.hp['optimize_decoder']:
                        if i > prev_epochs:
                            self.gen_opt_for_decoder.step()
                            self.gen_scheduler_for_decoder.step()
                        else:
                            self.gen_opt.step()
                            self.gen_scheduler.step()
                    else:
                        self.gen_opt.step()
                        self.gen_scheduler.step()

                    if l_total < min_loss:
                        min_loss = l_total
                        min_loss_out_6d = opt_out_6d
                        min_loss_out_rot_mat = opt_out_rot_mat
                        min_loss_out_pose_pos = opt_out_pose_pos

                opt_our_pred_6d_seq = opt_out_6d # 1 X window_size X 24 X 6
                opt_our_pred_rot_mat_seq = opt_out_rot_mat # 1 X window_size X 24 X 3 X 3
                opt_our_fk_pose = opt_out_pose_pos # 1 X window_size X 24 X 3

                if self.hp['replace_part_with_gt']:
                    replace_mask_6d = temporal_mask[None, :, :, None].float() # 1 X T X 24 X 1
                    replace_mask_rot_mat = temporal_mask[None, :, :, None, None].float() # 1 x T X 24 X 1 X 1
                    replace_mask_fk_pose = temporal_mask[None, :, :, None].float() # 1 X T X 24 X 1

                    opt_our_pred_6d_seq = replace_mask_6d * curr_target_cont6d_data + (1-replace_mask_6d) * opt_our_pred_6d_seq 
                    opt_our_pred_rot_mat_seq = replace_mask_rot_mat * curr_target_rot_data + (1-replace_mask_rot_mat) * opt_our_pred_rot_mat_seq
                    opt_our_fk_pose = replace_mask_fk_pose * curr_target_coord_data + (1-replace_mask_fk_pose) * opt_our_fk_pose

                # Concat result to existing sequence 
                if t_idx == 0:
                    whole_our_6d_seq = opt_our_pred_6d_seq.squeeze(0).detach()
                    whole_our_rot_mat_seq = opt_our_pred_rot_mat_seq.squeeze(0).detach()
                    whole_our_fk_pose_seq = opt_our_fk_pose.squeeze(0).detach()
                else:
                    whole_our_6d_seq = torch.cat((whole_our_6d_seq, opt_our_pred_6d_seq.squeeze(0).detach()[overlap:]), dim=0)
                    whole_our_rot_mat_seq = torch.cat((whole_our_rot_mat_seq, opt_our_pred_rot_mat_seq.squeeze(0).detach()[overlap:]), dim=0)
                    whole_our_fk_pose_seq = torch.cat((whole_our_fk_pose_seq, opt_our_fk_pose.squeeze(0).detach()[overlap:]), dim=0)

            if trajectory_trainer is not None:
                encoder_input = whole_our_6d_seq[None] # 1 X T X 24 X 6
                # For debug 
                # encoder_input = self.rot_mat_to_6d(gt_rot_mat[k_idx])[None] # 1 X T X 24 X 6, debug finished, trajectory is good with gt local motion
                pred_seq_w_root_trans_res = trajectory_trainer.sampled_seq_test(encoder_input).squeeze(1) # T X 24 X 3

                actual_len = pred_seq_w_root_trans_res.size()[0]
                if use_amass_data:
                    gt_root_trans_res = root_trans_list[f_cnt][:actual_len] # T X 3
                else:
                    gt_root_trans_res = pred_seq_w_root_trans_res.clone()[:, 0, :].cpu()
                gt_seq_w_root_trans_res = torch.from_numpy(gt_fk_pose[:actual_len].data.cpu().numpy()) # T X 24 X 3
                gt_seq_w_root_trans_res += gt_root_trans_res[:, None, :]

                # Save all the results 
                dest_directory = os.path.join(dest_image_directory, f_name, str(datetime.datetime.now()))
                if not os.path.exists(dest_directory):
                    os.makedirs(dest_directory)

                opt_our_rec_pose_seq_for_vis = pred_seq_w_root_trans_res[None].cpu().clone() # 1 X T X 24 X 3
                opt_our_rec_pose_seq_for_vis[:, :, :, 0] += 1

                gt_fk_pose_for_vis = gt_seq_w_root_trans_res[None].clone()
                # gt_fk_pose_for_vis[:, :, :, 0] += 0

                concat_seq_for_vis = torch.cat((gt_fk_pose_for_vis, opt_our_rec_pose_seq_for_vis), dim=0) # 2 X T X 24 X 3
                             
                tag = "temporal_interp_w_trajectory"

                show3Dpose_animation_multiple(concat_seq_for_vis.data.cpu().numpy(), dest_directory, \
                    0, tag, f_name.replace(".json", ""), use_amass=True)
                
                dest_dir = os.path.join(dest_directory, str(0), tag)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                # Save translation info to npy 
                dest_trans_opt_npy_path = os.path.join(dest_directory, str(0), tag, \
                    f_name.replace(".json", "")+"_root_trans_opt_res.npy")
                dest_trans_gt_npy_path = os.path.join(dest_directory, str(0), tag, \
                    f_name.replace(".json", "")+"_root_trans_gt_res.npy")


                np.save(dest_trans_opt_npy_path, pred_seq_w_root_trans_res.data.cpu().numpy()) # T X 24 X 3
                np.save(dest_trans_gt_npy_path, gt_seq_w_root_trans_res.data.cpu().numpy())

                # Save rotation info to npy 
                dest_opt_npy_path = os.path.join(dest_directory, str(0), tag, \
                    f_name.replace(".json", "")+"_rot_opt_res.npy")
                dest_gt_npy_path = os.path.join(dest_directory, str(0), tag, \
                    f_name.replace(".json", "")+"_rot_gt_res.npy")
                
                opt_rot_mat_for_save = whole_our_rot_mat_seq.data.cpu().numpy() # T X 24 X 3 X 3
                actual_len = whole_our_rot_mat_seq.size()[0]
                gt_rot_mat_for_save = rotMatrices.data.cpu().numpy()[:actual_len] # T X 24 X 3 X 3

                np.save(dest_opt_npy_path, opt_rot_mat_for_save)
                np.save(dest_gt_npy_path, gt_rot_mat_for_save)

            f_cnt += 1
