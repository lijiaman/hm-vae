import torch
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchgeometry as tgm

import os
import json
import numpy as np
import random 

from scipy.spatial.transform import Rotation as R

import my_tools
from fk_layer import ForwardKinematicsLayer

def rand_rotation_matrix(deflection=1.0, randnums=None):
    """
    Creates a random rotation matrix.
    
    deflection: the magnitude of the rotation. For 0, no rotation; for 1, competely random
    rotation. Small deflection => small perturbation.
    randnums: 3 random numbers in the range [0, 1]. If `None`, they will be auto-generated.
    """
    # from http://www.realtimerendering.com/resources/GraphicsGems/gemsiii/rand_rotation.c
    
    if randnums is None:
        randnums = np.random.uniform(size=(3,))
        
    theta, phi, z = randnums
    
    theta = theta * 2.0*deflection*np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0*np.pi  # For direction of pole deflection.
    z = z * 2.0*deflection  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    
    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

# 0: Pelvis, 1: L_Hip, 2: R_Hip, 3: Spine1, 4: L_Knee
# 5: R_Knee, 6: Spine2, 7: L_Ankle, 8: R_Ankle, 9: Spine3
# 10: L_Foot, 11: R_Foot, 12: Neck, 13: L_Collar, 14: R_Collar
# 15: Head, 16: L_Shoulder, 17: R_Shoulder, 18: L_Elbow, 19: R_Elbow
# 20: L_Wrist, 21: R_Wrist, 22(25): L_Index1, 23(40): R_Index1

def change_fps(ori_data, train_seq_len):
    # ori_data: T X n_dim
    try_cnt = 0
    res = ori_data
    while try_cnt < 10:
        sample_freq = random.sample([1, 2, 3, 4, 5, 6, 8, 10, 12], 1)[0] 
        # sample_freq = random.sample([1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 24, 30, 40, 60, 120], 1)[0] 
        # 120fps, 60fps, 40fps, 30fps, 24fps, 20fps, 15fps, 12fps, 10fps, 8fps, 6fps, 5fps, 4fps, 3fps, 2fps, 1fps    

        dest_data = ori_data[0::sample_freq, :]

        try_cnt += 1
        if dest_data.shape[0] >= train_seq_len:
            res = dest_data
            break 

    return res 

class MotionSeqData(data.Dataset):

    def __init__(self, rot_npy_folder, jsonfile, mean_std_path, cfg, test_flag=False, fps_aug_flag=False, random_root_rot_flag=False):

        self.ids_dic = json.load(open(jsonfile, 'r'))

        id_list = []
        for k in self.ids_dic.keys():
            id_list.append(int(k))
           
        self.ids = id_list

        self.rot_npy_folder = rot_npy_folder

        self.train_seq_len = cfg['train_seq_len']

        self.mean_std_data = np.load(mean_std_path) # 2 X n_dim
        self.mean_std_data[1, self.mean_std_data[1, :]==0] = 1.0

        self.hp = cfg 
        self.test_flag = test_flag

        self.fps_aug_flag = fps_aug_flag
        self.random_root_rot_flag = random_root_rot_flag

    def standardize_data(self, ori_data):
        # ori_data: T X n_dim
        mean_val = self.mean_std_data[0, :][np.newaxis, :] # 1 X n_dim
        std_val = self.mean_std_data[1, :][np.newaxis, :] # 1 X n_dim
        dest_data = (ori_data - mean_val)/std_val # T X n_dim

        return dest_data

    def standardize_data_specify_dim(self, ori_data, start_idx, end_idx):
        # ori_data: T X n_dim
        mean_val = self.mean_std_data[0, :][np.newaxis, start_idx:end_idx] # 1 X n_dim
        std_val = self.mean_std_data[1, :][np.newaxis, start_idx:end_idx] # 1 X n_dim
        dest_data = (ori_data - mean_val)/std_val # T X n_dim

        return dest_data

    def __getitem__(self, index):
        # index = 0 # For debug
        v_name = self.ids_dic[str(index)]      
        rot_npy_path = os.path.join(self.rot_npy_folder, v_name)
        ori_pose_seq_data = np.load(rot_npy_path) # T X n_dim

        if self.fps_aug_flag:
            ori_pose_seq_data = change_fps(ori_pose_seq_data, self.train_seq_len) # T' X n_dim

        timesteps = ori_pose_seq_data.shape[0]
        n_dim = ori_pose_seq_data.shape[1]
       
        if self.train_seq_len <= timesteps:
            random_t_idx = random.sample(list(range(timesteps-self.train_seq_len+1)), 1)[0]
            # random_t_idx = 0 # For debug
            end_t_idx = random_t_idx + self.train_seq_len - 1
        else:
            random_t_idx = 0
            end_t_idx = timesteps-1
           
        pose_seq_data = self.standardize_data(ori_pose_seq_data)

        # theta = torch.cat((rot6d_list.view(timesteps, -1), rot_list.view(timesteps, -1), coord_list.view(timesteps, -1), \
        #         linear_v.view(timesteps, -1), linear_v_list.view(timesteps, -1), root_v), dim=1) # T X n_dim
        # n_dim = 24*6(rot6d) + 24*3*3(rot matrix) + 24*3(joint coord) + 24*3(linear v) + 24*3(linear v instead, not used, angular v) + 3
        # = 144 + 216 + 72 + 72 + 72 + 3 = 579
        ori_seq_pose_data = torch.from_numpy(ori_pose_seq_data[random_t_idx:end_t_idx+1, :]).float() # T X n_dim
        seq_pose_data = torch.from_numpy(pose_seq_data[random_t_idx:end_t_idx+1, :]).float() # T X n_dim

        seq_rot_6d = ori_seq_pose_data[:, :24*6] # T X (24*6)
        seq_rot_mat = ori_seq_pose_data[:, 24*6:24*6+24*3*3] # T X (24*3*3), used for loss, no need for normalization
        seq_rot_pos = ori_seq_pose_data[:, 24*6+24*3*3:24*6+24*3*3+24*3] # T X (24*3), used for loss, no need for normalization
        seq_joint_pos = seq_pose_data[:, 24*6+24*3*3:24*6+24*3*3+24*3] # T X (24*3)
        seq_linear_v = seq_pose_data[:, 24*6+24*3*3+24*3:24*6+24*3*3+24*3+24*3] # T X (24*3) 
        seq_angular_v = seq_pose_data[:, 24*6+24*3*3+24*3+24*3:24*6+24*3*3+24*3+24*3+24*3] # T X (24*3)
        seq_root_v = seq_pose_data[:, 576:579] # T X 3 

        # Random rotate global orientation to make the model adapt to different global rotation 
        if self.random_root_rot_flag:
            random_root_rot = rand_rotation_matrix() # 3 X 3

            random_root_rot = torch.from_numpy(random_root_rot).float()[None, :, :] # 1 X 3 X 3
            random_root_rot = random_root_rot.repeat(self.train_seq_len, 1, 1) # T X 3 X 3
            ori_root_rot = seq_rot_mat[:, :3*3] # T X (3*3)
            ori_root_rot = ori_root_rot.contiguous().view(-1, 3, 3) 
            aug_root_rot = torch.matmul(random_root_rot, ori_root_rot) # T X 3 X 3

            # Process root_v
            aug_root_v = torch.matmul(random_root_rot, ori_seq_pose_data[:, 576:579][:, :, None]) # T X 3 X 1
            # aug_root_v = torch.matmul(noisy_random_root_rot, aug_root_v) 
            seq_root_v = aug_root_v.squeeze(-1) # T X 3 
            # Do normalization
            seq_root_v = self.standardize_data_specify_dim(seq_root_v, 576, 579)

            aug_root_rot = aug_root_rot.view(-1, 3*3) # T X 9
            seq_rot_mat[:, :3*3] = aug_root_rot

            rotMatrices = seq_rot_mat.view(-1, 24, 3, 3) # T X 24 X 3 X 3
            # Convert rotation matrix to 6D representation
            cont6DRep = torch.stack((rotMatrices[:, :, :, 0], rotMatrices[:, :, :, 1]), dim=-2) # T X 24 X 2 X 3
            cont6DRep = cont6DRep.view(rotMatrices.size()[0], rotMatrices.size()[1], 6) # T X 24 X 6

            seq_rot_6d = cont6DRep.view(-1, 24*6)
        
        return seq_rot_6d, seq_rot_mat, seq_rot_pos, seq_joint_pos, seq_linear_v, seq_angular_v, seq_root_v
        # When data_aug is True, only use seq_rot_6d, seq_rot_mat  

    def __len__(self):
        return len(self.ids)

def get_train_loaders_all_data_seq(cfg):
    root_folder = "/orion/u/jiamanli/github/hm-vae/utils/data"
  
    data_folder = os.path.join(root_folder, "for_all_data_motion_model")

    if cfg['use_30fps_data']:
        rot_npy_folder = os.path.join(root_folder, "/orion/u/jiamanli/datasets/amass_for_hm_vae_fps30")
    else:
        rot_npy_folder = "/orion/u/jiamanli/datasets/amass_for_hm_vae"
    
    mean_std_path = os.path.join(data_folder, "all_amass_data_mean_std.npy")

    train_json_file = os.path.join(data_folder, "train_all_amass_motion_data.json")
    val_json_file = os.path.join(data_folder, "val_all_amass_motion_data.json")
  
    batch_size = cfg['batch_size'] 

    workers = 1
    fps_aug_flag = cfg['fps_aug_flag']
    random_root_rot_flag = cfg['random_root_rot_flag']

    train_dataset = MotionSeqData(rot_npy_folder, train_json_file, mean_std_path, cfg, \
        fps_aug_flag=fps_aug_flag, random_root_rot_flag=random_root_rot_flag)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)

    val_dataset = MotionSeqData(rot_npy_folder, val_json_file, mean_std_path, cfg, \
        fps_aug_flag=fps_aug_flag, random_root_rot_flag=random_root_rot_flag)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)

    test_dataset = MotionSeqData(rot_npy_folder, val_json_file, mean_std_path, cfg, \
        fps_aug_flag=fps_aug_flag, random_root_rot_flag=random_root_rot_flag)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True, drop_last=True)

    return (train_loader, val_loader, test_loader)

class EvalMotionSeqData(data.Dataset):

    def __init__(self, rot_npy_folder, jsonfile, mask_npy_folder, test_flag, cfg):

        self.ids_dic = json.load(open(jsonfile, 'r'))

        id_list = []
        for k in self.ids_dic.keys():
            id_list.append(int(k))
           
        self.ids = id_list

        self.rot_npy_folder = rot_npy_folder
        self.mask_npy_folder = os.path.join(mask_npy_folder, str(cfg['missing_joint_prob'])) 

        self.hp = cfg 
        self.test_flag = test_flag

        self.train_seq_len = cfg['train_seq_len']

        self.upper_joint_list = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        self.lower_joint_list = [1, 2, 4, 5, 7, 8, 10, 11]

        self.missing_upper_mask = torch.ones(24) # 24
        self.missing_upper_mask[self.upper_joint_list] = 0

        self.missing_lower_mask = torch.ones(24)
        self.missing_lower_mask[self.lower_joint_list] = 0

    def __getitem__(self, index):
        v_name = self.ids_dic[str(index)]      
        rot_npy_path = os.path.join(self.rot_npy_folder, v_name)
        mask_npy_path = os.path.join(self.mask_npy_folder, v_name)

        ori_pose_seq_data = np.load(rot_npy_path) # T X n_dim
        timesteps = ori_pose_seq_data.shape[0]

        if self.hp['make_upper_joint_missing']:
            mask_data = self.missing_upper_mask[None, :].repeat(timesteps, 1) # T X 24
        elif self.hp['make_lower_joint_missing']:
            mask_data = self.missing_lower_mask[None, :].repeat(timesteps, 1) # T X 24
        else:
            mask_data = np.load(mask_npy_path) # T X 24
            mask_data = torch.from_numpy(mask_data).float()

        if self.test_flag:
            ori_seq_pose_data = ori_pose_seq_data
        else:
            # Random select one interval data seq for training, t~t+seq_len-1
            random_t_idx = random.sample(list(range(timesteps-self.train_seq_len+1)), 1)[0]
            end_t_idx = random_t_idx + self.train_seq_len - 1

            # theta = torch.cat((rot6d_list.view(timesteps, -1), rot_list.view(timesteps, -1), coord_list.view(timesteps, -1), \
            #         linear_v.view(timesteps, -1), angular_v_list.view(timesteps, -1), root_v), dim=1) # T X n_dim
            # n_dim = 24*6(rot6d) + 24*3*3(rot matrix) + 24*3(joint coord) + 24*3(linear v) + 24*3(angular v) + 3
            # = 144 + 216 + 72 + 72 + 72 + 3 = 579
            ori_seq_pose_data = ori_pose_seq_data[random_t_idx:end_t_idx+1, :] # T X n_dim
      
        seq_rot_6d = torch.from_numpy(ori_seq_pose_data[:, :24*6]).float() # T X (24*6)
        seq_rot_mat = torch.from_numpy(ori_seq_pose_data[:, 24*6:24*6+24*3*3]).float() # T X (24*3*3), used for loss, no need for normalization
        seq_rot_pos = torch.from_numpy(ori_seq_pose_data[:, 24*6+24*3*3:24*6+24*3*3+24*3]).float() # T X (24*3), used for loss, no need for normalization

        seq_rot_6d = seq_rot_6d.view(-1, 24, 6) # T X 24 X 6
        seq_rot_mat = seq_rot_mat.view(-1, 24, 3, 3) # T X 24 X 3 X 3
        seq_rot_pos = seq_rot_pos.view(-1, 24, 3) # T X 24 X 3

        if self.test_flag:
            masked_cont6DRep = seq_rot_6d.clone() # T X 24 X 6
            masked_rotRep = seq_rot_mat.clone() # T X 24 X 3 X 3
            masked_coordRep = seq_rot_pos.clone() # T X 24 X 3
            
            # mask_data = torch.from_numpy(mask_data).float() # T X 24
            masked_cont6DRep[mask_data==0] = 0.0
            masked_rotRep[mask_data==0] = 0.0
            masked_coordRep[mask_data==0] = 0.0
        else:
            masked_cont6DRep = seq_rot_6d.clone()
            masked_rotRep = seq_rot_mat.clone()
            masked_coordRep = seq_rot_pos.clone()
            mask_data = seq_rot_6d # not used!
        
        return seq_rot_6d, seq_rot_mat, seq_rot_pos, masked_cont6DRep, masked_rotRep, masked_coordRep, mask_data              

    def __len__(self):
        return len(self.ids)

def get_train_loaders_all_data_seq_for_eval(cfg):
    root_folder = "/mount/Users/jiaman/adobe/github/motion_prior/utils/data"

    if not os.path.exists(root_folder):
        root_folder = "/glab2/data/Users/jiaman/adobe/github/motion_prior/utils/data"
  
    data_folder = os.path.join(root_folder, "for_all_data_motion_model")
    rot_npy_folder = os.path.join(root_folder, "processed_all_amass_data")
    val_json_file = os.path.join(data_folder, "test_all_amass_motion_data.json")
    mask_npy_folder = os.path.join(root_folder, "all_amass_data_motion_noisy_data_for_eval")
   
    # data_folder = os.path.join(root_folder, "for_subset_motion_model")
    # rot_npy_folder = os.path.join(root_folder, "processed_walk_subset_data")
    # val_json_file = os.path.join(data_folder, "val_walk_subset_motion_data.json")
    # mask_npy_folder = os.path.join(root_folder, "walk_motion_subset_noisy_data_for_eval")
    
    workers = 1
    test_flag = True
    bs = 1
    
    # test_dataset = EvalMotionSeqData(rot_npy_folder, val_json_file, mask_npy_folder, cfg, test_flag=False)
    test_dataset = EvalMotionSeqData(rot_npy_folder, val_json_file, mask_npy_folder, test_flag, cfg)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=bs, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=True)

    return test_loader
