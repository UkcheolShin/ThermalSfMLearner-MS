import torch
import torch.utils.data as data
import numpy as np

import math
import random
from imageio import imread
from path import Path

def load_as_float(path):
    return imread(path).astype(np.float32)


class ValidationSet(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/Thermal/0000000.jpg
        root/scene_1/Depth/0000000.npy
        root/scene_1/Thermal/0000001.jpg
        root/scene_1/Depth/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, tf_thr=None, scene_type='indoor', inference_folder = '', sequence_length=3, interval=1):
        self.root = Path(root)

        if inference_folder == '' : 
            if scene_type == 'indoor': # indoor
                folder_list_path = self.root/'val_indoor.txt'
            elif scene_type == 'outdoor': # outdoor
                folder_list_path = self.root/'val_outdoor.txt'            
            self.folders = [self.root/folder[:-1] for folder in open(folder_list_path)]
        else:
            self.folders = [self.root/inference_folder]

        self.tf_thr       = tf_thr
        self.crawl_folders(sequence_length, interval)

    def crawl_folders(self, sequence_length=3, interval=1):
        sequence_set = []
        demi_length = (sequence_length-1)//2 + interval - 1
        shifts = list(range(-demi_length, demi_length + 1))
        for i in range(1, 2*demi_length):
            shifts.pop(1)

        for folder in self.folders:      
            imgs_thr = sorted((folder/"Thermal").files('*.png')) 
            for i in range(demi_length, len(imgs_thr)-demi_length):
                d_thr = folder/"Depth_T"/(imgs_thr[i].name[:-4] + '.npy')
                sample = {'tgt_thr': imgs_thr[i], 'tgt_thr_depth': d_thr }
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_thr_img  = np.expand_dims(load_as_float(sample['tgt_thr']), axis=2)
        depth_thr    = np.load(sample['tgt_thr_depth']).astype(np.float32)

        imgs_thr, _  = self.tf_thr([tgt_thr_img], None)
        tgt_thr_img  = imgs_thr[0]

        return tgt_thr_img, depth_thr

    def __len__(self):
        return len(self.samples)


class ValidationSetPose(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/Thermal/0000000.jpg
        root/scene_1/Depth/0000000.npy
        root/scene_1/Thermal/0000001.jpg
        root/scene_1/Depth/0000001.npy
        ..
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .

        transform functions must take in a list a images and a numpy array which can be None
    """

    def __init__(self, root, tf_thr=None, scene_type='indoor', sequence_length=3):
        self.root = Path(root)
        if scene_type == 'indoor': # indoor
            scene_list_path = self.root/'val_indoor.txt'
        elif scene_type == 'outdoor': # outdoor
            scene_list_path = self.root/'val_outdoor.txt'

        self.folders = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.tf_thr = tf_thr
        self.crawl_folders(sequence_length, step=1)

    def crawl_folders(self, sequence_length=3, step=1):
        sequence_set = []
        demi_length = (sequence_length - 1) // 2
        shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)
        for folder in self.folders:      
            imgs_thr = sorted((folder/"Thermal").files('*.png')) # "RGB" "Depth" "Thermal" "Thermal_Heat" "Thermal_HeatMM" "Thermal_HeatMMJ"
            poses_T  = np.genfromtxt(folder/'poses_T.txt').astype(np.float64).reshape(-1, 3, 4)

            # construct 5-snippet sequences
            tgt_indices = np.arange(demi_length, len(imgs_thr) - demi_length).reshape(-1, 1)
            snippet_indices = shift_range + tgt_indices
            
            for indices in snippet_indices :
                sample = {'thr_imgs' : [], 'thr_poses' : []}
                for i in indices :
                    sample['thr_imgs'].append(imgs_thr[i])
                    sample['thr_poses'].append(poses_T[i])
                sequence_set.append(sample)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        imgs = [np.expand_dims(load_as_float(img), axis=2) for img in sample['thr_imgs']]

        imgs, _ = self.tf_thr(imgs, None)

        poses = np.stack([pose for pose in sample['thr_poses']])
        first_pose = poses[0]
        poses[:,:,-1] -= first_pose[:,-1]
        compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses

        return imgs, compensated_poses

    def __len__(self):
        return len(self.samples)
