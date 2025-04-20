import os
import numpy as np
import json
import copy
import random

import lmdb
import msgpack
import msgpack_numpy
from typing import List, Dict, Tuple, Union, Iterator
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
msgpack_numpy.patch()
from genrobo3d.utils.rvt_util import instr_trans,unify_color,replace_color
import torch
from genrobo3d.configs.rlbench.constants import SCENE_BOUNDS,COLORS
from scipy.spatial.transform import Rotation as R

from genrobo3d.train.datasets.common import (
    pad_tensors, gen_seq_masks, random_rotate_z
)
from genrobo3d.configs.rlbench.constants import (
    get_rlbench_labels, get_robot_workspace
)
from genrobo3d.utils.rotation_transform import (
    RotationMatrixTransform, quaternion_to_discrete_euler
)
from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.utils.action_position_utils import get_disc_gt_pos_prob
from genrobo3d.train.datasets.simple_policy_dataset import SimplePolicyDataset


class MotionPlannerDataset(SimplePolicyDataset):
    def __init__(
            self, data_dir, action_embed_file, gt_act_obj_label_file,
            taskvar_file=None, num_points=10000, 
            xyz_shift='center', xyz_norm=False, use_height=False,
            max_traj_len=5, pc_label_type='coarse',
            pc_label_augment=False, pc_midstep_augment=False,
            rot_type='quat', instr_embed_type='last', all_step_in_batch=True,
            rm_table=True, rm_robot='none', include_last_step=False, augment_pc=False,
            same_npoints_per_example=False,
            rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
            pos_type='cont', pos_bins=50, pos_bin_size=0.01, 
            pos_heatmap_type='plain', pos_heatmap_no_robot=False, 
            aug_max_rot=45, use_color=False, instr_include_objects=False, 
            real_robot=False,transform_color=False,aug_color=False,crop_label=False, **kwargs
        ):

        assert instr_embed_type in ['last', 'all']
        assert xyz_shift in ['none', 'center', 'gripper']
        assert pos_type in ['cont', 'disc']
        assert rot_type in ['quat', 'rot6d', 'euler', 'euler_disc']
        assert rm_robot in ['none', 'gt', 'box', 'box_keep_gripper']
        assert pc_label_type in ['coarse', 'fine', 'mix']

        self.action_embeds = np.load(action_embed_file, allow_pickle=True).item()
        self.action_names = set()
        if instr_embed_type == 'last':
            self.action_embeds = {instr: embeds[-1:] for instr, embeds in self.action_embeds.items()}
        if gt_act_obj_label_file is not None:
            self.gt_act_obj_labels = json.load(open(gt_act_obj_label_file))
            
        self.aug_color = aug_color
        if aug_color:
            self.color_dict=json.load(open("assets/train_aug_color.json"))

        if taskvar_file is not None:
            self.taskvars = json.load(open(taskvar_file))
        else:
            self.taskvars = os.listdir(data_dir)

        self.lmdb_envs, self.lmdb_txns = {}, {}
        self.data_ids = []
        for taskvar in self.taskvars:
            if not os.path.exists(os.path.join(data_dir, taskvar)):
                continue
            self.lmdb_envs[taskvar] = lmdb.open(os.path.join(data_dir, taskvar), readonly=True)
            self.lmdb_txns[taskvar] = self.lmdb_envs[taskvar].begin()
            if all_step_in_batch:
                self.data_ids.extend(
                    [(taskvar, key) for key in self.lmdb_txns[taskvar].cursor().iternext(values=False)]
                )
            else:
                for key, value in self.lmdb_txns[taskvar].cursor():
                    value = msgpack.unpackb(value)
                    num_steps = len(value['xyz'])
                    if (not include_last_step):
                        num_steps -= 1
                    assert num_steps>0
                    if pc_midstep_augment:
                        self.data_ids.extend([(taskvar, key, t) for t in range(num_steps)])
                    else:
                        self.data_ids.extend([(taskvar, key, t) for t in range(num_steps) if (value['is_new_keystep'][t]) or (t == len(value['xyz'])-1)])

        self.num_points = num_points
        self.max_traj_len = max_traj_len
        self.pc_label_type = pc_label_type
        self.pc_label_augment = pc_label_augment
        self.pc_midstep_augment = pc_midstep_augment
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm
        self.use_height = use_height
        self.pos_type = pos_type
        self.rot_type = rot_type
        self.rm_table = rm_table
        self.rm_robot = rm_robot
        self.all_step_in_batch = all_step_in_batch
        self.include_last_step = include_last_step
        self.instr_include_objects = instr_include_objects
        self.use_color = use_color
        self.augment_pc = augment_pc
        self.aug_max_rot = np.deg2rad(aug_max_rot)
        self.rm_pc_outliers = rm_pc_outliers
        self.rm_pc_outliers_neighbors = rm_pc_outliers_neighbors
        self.same_npoints_per_example = same_npoints_per_example
        self.euler_resolution = euler_resolution
        self.pos_bins = pos_bins
        self.pos_bin_size = pos_bin_size
        self.pos_heatmap_type = pos_heatmap_type
        self.pos_heatmap_no_robot = pos_heatmap_no_robot
        self.real_robot = real_robot

        self.TABLE_HEIGHT = get_robot_workspace(real_robot=self.real_robot)['TABLE_HEIGHT']
        self.rotation_transform = RotationMatrixTransform()
        self.transform_color = transform_color
        self.crop_label = crop_label
        self.crop_with_idx = False
        self.data_cache={}

    def _aug_pc_color(self,instr_dict:dict,aug_label:str,exp_col_lst:list,pc,pc_labels):
        assert aug_label in ['object','target']
        color_names = [color[0] for color in COLORS]
        instr_dict = copy.deepcopy(instr_dict)
        opt_color_lst=[]
        for color in COLORS:
            if color[0] in exp_col_lst:
                continue
            opt_color_lst.append(color)
        target_color = opt_color_lst[random.randint(0,len(opt_color_lst)-1)]

        label_dict={'object':2, 'target':3}
        target_label = label_dict[aug_label]
        for i in range(len(pc_labels)):
            if pc_labels[i]==target_label:
                for idx in range(3):
                    pc[i,idx-3] = target_color[1][idx]*2-1
        assert aug_label in instr_dict.keys()
        
        aug_instr = replace_color(aug_label,instr_dict,target_color[0],exp_col_lst)
        # print(f"target color {target_color[0]}")
        return aug_instr,pc



    def _get_rotation_from_quat(self, gt_rot):
        if self.rot_type == 'euler':
            gt_rot = self.rotation_transform.quaternion_to_euler(
                torch.from_numpy(gt_rot[None, :]))[0].numpy() / 180.
        elif self.rot_type == 'euler_disc':
            gt_rot = quaternion_to_discrete_euler(gt_rot, self.euler_resolution)
        elif self.rot_type == 'rot6d':
            gt_rot = self.rotation_transform.quaternion_to_ortho6d(
                torch.from_numpy(gt_rot[None, :]))[0].numpy()
        return gt_rot
                
    def _augment_pc(self, xyz, ee_pose, gt_trajs, aug_max_rot):
        # rotate around z-axis
        angle = np.random.uniform(-1, 1) * aug_max_rot
        xyz = random_rotate_z(xyz, angle=angle)
        ee_pose[:3] = random_rotate_z(ee_pose[:3], angle=angle)
        ee_pose[3:-1] = self._rotate_gripper(ee_pose[3:-1], angle)
        for gt_action in gt_trajs:
            gt_action[:3] = random_rotate_z(gt_action[:3], angle=angle)
            gt_action[3:-1] = self._rotate_gripper(gt_action[3:-1], angle)

        # add small noises (+-2mm)
        pc_noises = np.random.uniform(0, 0.002, size=xyz.shape)
        xyz = pc_noises + xyz

        return xyz, ee_pose, gt_trajs
    
    def _get_distract_obj_pc(self,pc,gen_num=3):
        #all pc should be object label
        #pc (xxx,3)
        if gen_num==0:
            return None
        o = np.array([np.mean(pc[:, 0]),np.mean(pc[:, 1])])
        l_array = np.linalg.norm(pc[:, :2] - o, axis=1)
        sorted_array = np.sort(l_array)
        k = 30
        l = sorted_array[-k]
        assert l>sorted_array[0]
        x_min=SCENE_BOUNDS[0]
        y_min=SCENE_BOUNDS[1]
        x_max=SCENE_BOUNDS[3]
        y_max=SCENE_BOUNDS[4]
        x_min+=7*l
        x_max-=7*l
        y_min+=7*l
        y_max-=7*l
        o_p=None
        o_lst = [o]
        for _ in range(200):
            if len(o_lst)>=gen_num+1:
                break
            o_p=np.array([random.uniform(x_min, x_max),random.uniform(y_min,y_max)])
            can_add=True
            for cent in o_lst:
                if np.linalg.norm(cent - o_p)<5*l:
                    can_add=False
                    break
            if can_add:
                o_lst.append(o_p)
        ret_pc = []
        for i in range(1,len(o_lst),1):
            o_p=o_lst[i]
            raw_pc = pc - np.append(o_p,0)
            angle = np.random.uniform(-1, 1) * 180
            tar_pc = random_rotate_z(raw_pc, angle=angle)+np.append(o_p,0)
            ret_pc.append(tar_pc)
        if len(ret_pc)>0:
            return np.concatenate(ret_pc,0)
        else:
            return None
        
    def __getitem__(self, idx):
        if self.all_step_in_batch:
            taskvar, data_id = self.data_ids[idx]
        else:
            taskvar, data_id, data_step = self.data_ids[idx]

        task, variation = taskvar.split('+')

        gt_act_obj_labels = self.gt_act_obj_labels[taskvar]
        # id_str=data_id.decode('utf-8')
        # query_key=taskvar+id_str
        # if query_key in self.data_cache.keys():
        #     data = self.data_cache[query_key]
        # else:
        data = msgpack.unpackb(self.lmdb_txns[taskvar].get(data_id))
        # self.data_cache[query_key]=data
        self.crop_with_idx = self.crop_label and 'img_idx' in data.keys()
        outs = {
            'data_ids': [], 'pc_fts': [], 'pc_labels': [], 
            'pc_centroids': [], 'pc_radius': [], 'ee_poses': [], 
            'txt_embeds': [], 'gt_trajs': [], 'gt_trajs_stop': [],'instr_txt': []
        }
        if self.pos_type == 'disc':
            outs['gt_trajs_disc_pos_probs'] = []

        keystep = -1
        num_steps = len(data['xyz'])
        for t in range(num_steps):
            if data['is_new_keystep'][t]:
                keystep += 1

            if (not self.all_step_in_batch) and t != data_step:
                continue
            if (not self.pc_midstep_augment) and (not data['is_new_keystep'][t]) and (t != num_steps - 1):
                continue
            if (not self.include_last_step) and (t == (num_steps - 1)):
                continue

            xyz, rgb, gt_sem = data['xyz'][t], data['rgb'][t], data['sem'][t]
            if self.crop_with_idx:
                img_idx = data['img_idx'][t]
            arm_links_info = (
                {k: v[t] for k, v in data['bbox_info'].items()}, 
                {k: v[t] for k, v in data['pose_info'].items()}
            )

            if t < num_steps - 1:
                gt_traj_len = len(data['trajs'][t])
                gt_trajs = copy.deepcopy(data['trajs'][t])[:self.max_traj_len]
            else:
                gt_traj_len = 1
                gt_trajs = copy.deepcopy(data['trajs'][-2][-1:])
            ee_pose = copy.deepcopy(data['ee_pose'][t])

            action_name = gt_act_obj_labels[keystep]['action']
            if self.instr_include_objects:
                if 'object' in gt_act_obj_labels[keystep]:
                    action_name = f"{action_name} {gt_act_obj_labels[keystep]['object']['name']}"
                if 'target' in gt_act_obj_labels[keystep]:
                    action_name = f"{action_name} to {gt_act_obj_labels[keystep]['target']['name']}"
            if not self.transform_color:
                action_embed = self.action_embeds[action_name]

            # remove background points
            if self.rm_table:
                mask = xyz[..., 2] > self.TABLE_HEIGHT
                xyz = xyz[mask]
                rgb = rgb[mask]
                gt_sem = gt_sem[mask]
                if self.crop_with_idx:
                    img_idx = img_idx[mask]
                
            if self.rm_robot.startswith('box'):
                mask = self._get_mask_with_robot_box(xyz.copy(), arm_links_info, self.rm_robot)
                xyz = xyz[mask]
                rgb = rgb[mask]
                gt_sem = gt_sem[mask]
                if self.crop_with_idx:
                    img_idx = img_idx[mask]

            if self.rm_pc_outliers:
                xyz, rgb, point_idxs = self._rm_pc_outliers(xyz, rgb=rgb, return_idxs=True)
                gt_sem = gt_sem[point_idxs]
                if self.crop_with_idx:
                    img_idx = img_idx[point_idxs]

            # sampling points
            if len(xyz) > self.num_points:
                point_idxs = np.random.permutation(len(xyz))[:self.num_points]
            else:
                if self.same_npoints_per_example:
                    point_idxs = np.random.choice(xyz.shape[0], self.num_points, replace=True)
                else:
                    max_npoints = int(len(xyz) * np.random.uniform(0.95, 1))
                    point_idxs = np.random.permutation(len(xyz))[:max_npoints]

            xyz = xyz[point_idxs]
            rgb = rgb[point_idxs]
            gt_sem = gt_sem[point_idxs]
            if self.crop_with_idx:
                img_idx = img_idx[point_idxs]
            height = xyz[:, -1] - self.TABLE_HEIGHT

            # robot_mask = self._get_mask_with_label_ids(gt_sem, robot_label_ids)
            robot_box = RobotBox(arm_links_info, keep_gripper=False)
            robot_point_idxs = robot_box.get_pc_overlap_ratio(xyz=xyz, return_indices=True)[1]
            robot_point_idxs = np.array(list(robot_point_idxs))
            robot_mask = np.zeros((xyz.shape[0], ), dtype=bool)
            if len(robot_point_idxs) > 0:
                robot_mask[robot_point_idxs] = True

            pc_label = np.zeros((gt_sem.shape[0], ), dtype=np.int32)
            pc_label[robot_mask] = 1
            for oname in ['object', 'target']:
                if oname in gt_act_obj_labels[keystep]:
                    v = gt_act_obj_labels[keystep][oname]
                    if self.pc_label_type != 'mix':
                        obj_label_ids = v[self.pc_label_type]
                    else:
                        obj_label_ids = v[random.choice(['coarse', 'fine'])]
                    obj_mask = self._get_mask_with_label_ids(gt_sem, obj_label_ids)
                    if 'zrange' in v:
                        obj_mask = obj_mask & (xyz[:, 2] > v['zrange'][0]) & (xyz[:, 2] < v['zrange'][1])
                    if self.pc_label_augment > 0: # only keep part of the gt labels
                        rand_idxs = np.arange(obj_mask.shape[0])[obj_mask]
                        rm_num = int(np.random.uniform(low=0, high=self.pc_label_augment) * len(rand_idxs))
                        rand_idxs = np.random.permutation(rand_idxs)[:rm_num]
                        obj_mask[rand_idxs] = False
                    if oname == 'object':
                        pc_label[obj_mask] = 2
                    else:   # target
                        pc_label[obj_mask] = 3

            # point cloud augmentation
            if self.augment_pc:
                xyz, ee_pose, gt_trajs = self._augment_pc(xyz, ee_pose, gt_trajs, self.aug_max_rot)
            gt_rots = np.stack(
                [self._get_rotation_from_quat(gt_action[3:-1]) for gt_action in gt_trajs], 0
            )

            # normalize point cloud
            if self.xyz_shift == 'none':
                centroid = np.zeros((3, ))
            elif self.xyz_shift == 'center':
                centroid = np.mean(xyz, 0)
            elif self.xyz_shift == 'gripper':
                centroid = copy.deepcopy(ee_pose[:3])
            if self.xyz_norm:
                radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
            else:
                radius = 1

            xyz = (xyz - centroid) / radius
            height = height / radius
            gt_trajs[:, :3] = (gt_trajs[:, :3] - centroid) / radius
            ee_pose[:3] = (ee_pose[:3] - centroid) / radius
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)

            gt_trajs = np.concatenate([gt_trajs[:, :3], gt_rots, gt_trajs[:, -1:]], -1)
            # if self.transform_color and task=='push_button':
            #     #data sem error, training only top is object, test all is object
            #     obj_list_xyz = []
            #     for i in range(len(pc_label)):
            #         label = pc_label[i]
            #         pt = xyz[i]
            #         if label==2 or label==0:
            #             pc_label[i]=2
            #             obj_list_xyz.append(pt)
                
            #     gen_distractor_num = random.randint(0, 3)
            #     target_xyz = np.array(obj_list_xyz)
            #     aug_pc = self._get_distract_obj_pc(target_xyz,gen_distractor_num)
            #     if aug_pc is not None:
            #         pc_label = np.concatenate([pc_label,np.zeros(len(aug_pc), dtype=np.int32)])
            #         xyz = np.concatenate([xyz,aug_pc],0)
            #         rgb = np.concatenate([rgb,np.ones((len(aug_pc),3), dtype=np.uint8)*255],0)
            #         height = np.concatenate([height,np.zeros(len(aug_pc), dtype=np.float64)])
                    

            pc_ft = xyz
            if self.use_height:
                pc_ft = np.concatenate([pc_ft, height[:, None]], 1)
            if self.use_color:
                rgb = (rgb / 255.) * 2 - 1
                pc_ft = np.concatenate([pc_ft, rgb], 1)

            if self.crop_with_idx:#use img index to crop
                n_cam = 4
                n_save = random.randint(1,n_cam)# start from zero
                reserve_lst = np.random.permutation(n_cam)[:n_save].tolist()
                assert len(pc_label)==len(img_idx)
                for i in range(len(pc_label)):
                    label = pc_label[i]
                    if label==2 or label==3:
                        if img_idx[i] in reserve_lst:
                            continue
                        pc_label[i]=0
            elif self.crop_label: 
                crop_poss=60
                if random.randint(1,100)<crop_poss:
                    drop_percent = random.randint(25,75)
                    for i in range(len(pc_label)):
                        label = pc_label[i]
                        if label==2 or label==3:
                            if random.randint(1,100)>drop_percent:
                                continue
                            pc_label[i]=0


            if self.transform_color:
                action_name = instr_trans(action_name)
                action_embed = self.action_embeds[action_name]
                pc_ft = unify_color(pc_label,pc_ft)
            
            if self.aug_color and taskvar in self.color_dict.keys():
                aug_thresh = 95
                if random.randint(1,100)<aug_thresh:#进行增强
                    except_color_lst = self.color_dict[taskvar]['distractor_colors']
                    aug_label = self.color_dict[taskvar]['aug_label']
                    if task == "stack_blocks" and \
                        gt_act_obj_labels[keystep]['action']=="move grasped object" and \
                        gt_act_obj_labels[keystep]["target"]["name"]!="green square":
                            aug_label = "target"
                    if aug_label in gt_act_obj_labels[keystep].keys():
                        action_name,pc_ft = self._aug_pc_color(gt_act_obj_labels[keystep],aug_label,except_color_lst,pc_ft,pc_label)
                    # print(f"{aug_label} {taskvar} {action_name}")
                action_embed = self.action_embeds[action_name]

            if self.pos_type == 'disc':
                gt_trajs_disc_pos_probs = []
                for gt_action in gt_trajs:
                    # (npoints, 3, pos_bins*2)
                    disc_pos_prob = get_disc_gt_pos_prob(
                        xyz, gt_action[:3], pos_bins=self.pos_bins, 
                        pos_bin_size=self.pos_bin_size,
                        heatmap_type=self.pos_heatmap_type,
                        robot_point_idxs=robot_point_idxs if self.pos_heatmap_no_robot else None
                    )
                    gt_trajs_disc_pos_probs.append(disc_pos_prob)
                # (max_trajs, npoints, 3, pos_bins*2)
                gt_trajs_disc_pos_probs = np.stack(gt_trajs_disc_pos_probs, 0)
                outs['gt_trajs_disc_pos_probs'].append(torch.from_numpy(gt_trajs_disc_pos_probs))
            outs['instr_txt'].append(f'{action_name}')
            outs['data_ids'].append(f'{taskvar}-{data_id.decode("ascii")}-t{t}')
            outs['pc_fts'].append(torch.from_numpy(pc_ft).float())
            outs['pc_centroids'].append(centroid)
            outs['pc_radius'].append(radius)
            outs['pc_labels'].append(torch.from_numpy(pc_label).long())
            outs['txt_embeds'].append(torch.from_numpy(action_embed).float())
            outs['ee_poses'].append(torch.from_numpy(ee_pose).float())
            outs['gt_trajs'].append(torch.from_numpy(gt_trajs).float())
            outs['gt_trajs_stop'].append(torch.arange(self.max_traj_len) >= (gt_traj_len-1))
            
        return outs
    

def base_collate_fn_partial(max_traj_len, data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    
    for key in ['pc_fts', 'pc_labels', 'ee_poses']:   
        batch[key] = torch.stack(batch[key], 0)

    gt_trajs, traj_lens = [], []
    for traj in batch['gt_trajs']:
        traj_lens.append(traj.size(0))
        if traj.size(0) < max_traj_len:
            # repeat the last action
            gt_trajs.append(
                torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1)])
            )
        else:
            assert len(traj) == max_traj_len, len(traj)
            gt_trajs.append(traj)
    batch['gt_trajs'] = torch.stack(gt_trajs, 0)
    batch['traj_masks'] = torch.from_numpy(
        gen_seq_masks(traj_lens, max_len=max_traj_len)
    ).bool()

    txt_lens = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_masks'] = torch.from_numpy(
        gen_seq_masks(txt_lens, max_len=max(txt_lens))
    ).bool()
    batch['txt_embeds'] = pad_tensors(
        batch['txt_embeds'], lens=txt_lens, max_len=max(txt_lens)
    )

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)
        batch['pc_radius'] = np.array(batch['pc_radius'])
    
    return batch

def ptv3_collate_fn_partial(max_traj_len, data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    
    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)         # (#all points, 6)
    batch['pc_labels'] = torch.cat(batch['pc_labels'], 0)   # (#all points, )

    for key in ['ee_poses', 'gt_trajs_stop']:
        if key in batch:
            batch[key] = torch.stack(batch[key], 0)

    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

    gt_trajs, traj_lens = [], []
    for traj in batch['gt_trajs']:
        traj_lens.append(traj.size(0))
        if traj.size(0) < max_traj_len:
            # repeat the last action
            gt_trajs.append(
                torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1)])
            )
        else:
            assert len(traj) == max_traj_len, len(traj)
            gt_trajs.append(traj)
    batch['gt_trajs'] = torch.stack(gt_trajs, 0)    # (batch, traj_len, dim)
    batch['traj_lens'] = traj_lens
    batch['traj_masks'] = torch.from_numpy(
        gen_seq_masks(traj_lens, max_len=max_traj_len)
    ).bool()
    if 'gt_trajs_disc_pos_probs' in batch.keys():
        gt_trajs_disc_pos_probs = []
        for traj in batch['gt_trajs_disc_pos_probs']:
            if traj.size(0) < max_traj_len:
                # repeat the last action (max_traj_len, 3, npoints*bins)
                gt_trajs_disc_pos_probs.append(
                    torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1, 1)])
                )
            else:
                assert len(traj) == max_traj_len, len(traj)
                gt_trajs_disc_pos_probs.append(traj)
        batch['gt_trajs_disc_pos_probs'] = gt_trajs_disc_pos_probs
    
    return batch

def ptv3_collate_fn_partial4simple(max_traj_len, data):
    batch = {}
    for key in data[0].keys():
        batch[key] = sum([x[key] for x in data], [])
    
    npoints_in_batch = [x.size(0) for x in batch['pc_fts']]
    batch['npoints_in_batch'] = npoints_in_batch
    batch['offset'] = torch.cumsum(torch.LongTensor(npoints_in_batch), dim=0)
    batch['pc_fts'] = torch.cat(batch['pc_fts'], 0)         # (#all points, 6)
    batch['pc_labels'] = torch.cat(batch['pc_labels'], 0)   # (#all points, )

    for key in ['ee_poses', 'gt_trajs_stop']:
        if key in batch:
            batch[key] = torch.stack(batch[key], 0)

    batch['txt_lens'] = [x.size(0) for x in batch['txt_embeds']]
    batch['txt_embeds'] = torch.cat(batch['txt_embeds'], 0)

    if len(batch['pc_centroids']) > 0:
        batch['pc_centroids'] = np.stack(batch['pc_centroids'], 0)

    gt_trajs, traj_lens = [], []
    for traj in batch['gt_trajs']:
        traj_lens.append(traj.size(0))
        if traj.size(0) < max_traj_len:
            # repeat the last action
            gt_trajs.append(
                torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1)])
            )
        else:
            assert len(traj) == max_traj_len, len(traj)
            gt_trajs.append(traj)
    batch['gt_actions'] = torch.stack(gt_trajs, 0)    # (batch, traj_len, dim) change this
    batch['gt_trajs'] = batch['gt_actions']
    batch['traj_lens'] = traj_lens
    batch['traj_masks'] = torch.from_numpy(
        gen_seq_masks(traj_lens, max_len=max_traj_len)
    ).bool()
    if 'gt_trajs_disc_pos_probs' in batch.keys():
        gt_trajs_disc_pos_probs = []
        for traj in batch['gt_trajs_disc_pos_probs']:
            if traj.size(0) < max_traj_len:
                # repeat the last action (max_traj_len, 3, npoints*bins)
                gt_trajs_disc_pos_probs.append(
                    torch.cat([traj, traj[-1].repeat(max_traj_len - traj.size(0), 1, 1)])
                )
            else:
                assert len(traj) == max_traj_len, len(traj)
                gt_trajs_disc_pos_probs.append(traj)
        batch['gt_trajs_disc_pos_probs'] = gt_trajs_disc_pos_probs
    
    return batch

if __name__ == '__main__':
    from functools import partial

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    max_traj_len = 1
    data_file="data/gembench/train_dataset/motion_keysteps_bbox_pcd_sam2/seed0"
    dataset = MotionPlannerDataset(
        data_file,
        'data/gembench/train_dataset/instr_embed/action-object_embeds_clip.npy',
        'assets/taskvars_target_label_zrange.json',
        taskvar_file='assets/taskvars_train.json', 
        num_points=20480, xyz_shift='none', xyz_norm=False, use_height=True,
        max_traj_len=max_traj_len, pc_label_type='coarse',
        pc_label_augment=False, pc_midstep_augment=True, augment_pc=True,
        rot_type='quat', instr_embed_type='last', all_step_in_batch=False,
        rm_robot='gt', include_last_step=False, 
        same_npoints_per_example=False,instr_include_objects=True,
        rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
        pos_type='cont', pos_bins=15, pos_bin_size=0.01, pos_heatmap_type='dist',
        pos_heatmap_no_robot=True,transform_color=False,aug_color=True
    )
    # dataset = MotionPlannerRealRobotDataset(
    #     'data/real_robot_data/v3/keysteps_bbox_pcd_cam2_motionplanner_vlm',
    #     'data/gembench/train_dataset/motion_keysteps_bbox_pcd/action_embeds_clip.npy',
    #     taskvar_file='assets/taskvars_realrobotv1.json', 
    #     num_points=4096, xyz_shift='none', xyz_norm=False, use_height=True,
    #     max_traj_len=max_traj_len, pc_label_type='coarse',
    #     pc_label_augment=False, pc_midstep_augment=True, augment_pc=True,
    #     rot_type='euler_disc', instr_embed_type='last', all_step_in_batch=True,
    #     rm_robot='box_keep_gripper', include_last_step=False, 
    #     same_npoints_per_example=False,
    #     rm_pc_outliers=False, rm_pc_outliers_neighbors=25, euler_resolution=5,
    #     pos_type='disc', pos_bins=15, pos_bin_size=0.01, pos_heatmap_type='dist',
    #     pos_heatmap_no_robot=True, real_robot=True,
    # )
    # print('#data', len(dataset))

    collate_fn = partial(ptv3_collate_fn_partial, max_traj_len)
    size = 2
    sampler = DistributedSampler(
        dataset, num_replicas=size, rank=1, 
        shuffle=True
    )
    pre_epoch = sampler.set_epoch
    dataloader = loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=64,
        num_workers=1,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False,
        prefetch_factor=2 ,
    )
    print('#steps', len(dataloader))
    sm_bs=0
    sm=0
    from tqdm import tqdm
    for epoch in range(123):
        for batch in tqdm(dataloader):
            sm+=1
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.size())
        #     else:
        #         print(k)
        # print(batch['gt_trajs'])
        # print(batch['traj_masks'])
        # np.save('batch.npy', batch)
    print(sm)
