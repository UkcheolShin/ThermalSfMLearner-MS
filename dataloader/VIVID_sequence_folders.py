import torch
import torch.utils.data as data
import numpy as np

import math
import random
from imageio import imread
from path import Path

def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/Thermal/0000000.png
        root/scene_1/Thermal/0000001.png
        ..
        root/scene_1/cam.txt
        root/scene_2/Thermal/0000000.png
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """
    def __init__(self, root, seed=None, train=True, sequence_length=3, tf_share=None,\
                 tf_thr_color=None, tf_thr=None, tf_rgb=None,\
                 scene_type='indoor', interval=1):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        if scene_type == 'indoor': 
            folder_list_path = self.root/'train_indoor.txt' if train else self.root/'val_indoor.txt'
        elif scene_type == 'outdoor': 
            folder_list_path = self.root/'train_outdoor.txt' if train else self.root/'val_outdoor.txt'
        
        self.folders = [self.root/folder[:-1] for folder in open(folder_list_path)]
        self.tf_share = tf_share
        self.tf_thr_color = tf_thr_color
        self.tf_thr = tf_thr        
        self.tf_rgb = tf_rgb        
        self.crawl_folders(sequence_length, interval)

    def crawl_folders(self, sequence_length, interval):
        sequence_set = []
        demi_length = (sequence_length-1)//2 + interval - 1
        shifts = list(range(-demi_length, demi_length + 1))
        for i in range(1, 2*demi_length):
            shifts.pop(1)

        for folder in self.folders:
            imgs_thr           = sorted((folder/"Thermal").files('*.png')) # "RGB" "Depth" "Thermal"
            imgs_rgb           = sorted((folder/"RGB").files('*.png')) 

            intrinsics_thr     = np.genfromtxt(folder/'cam_T.txt').astype(np.float32).reshape((3, 3))
            intrinsics_rgb     = np.genfromtxt(folder/'cam_RGB.txt').astype(np.float32).reshape((3, 3))
            extrinsics_thr2rgb = np.genfromtxt(folder/'Tr_T2RGB.txt').astype(np.float32).reshape((4, 4)) 

            for i in range(demi_length, len(imgs_thr)-demi_length):
                sample = {'intrinsics_thr': intrinsics_thr, 'tgt_thr': imgs_thr[i], 'ref_thr_imgs': [],
                          'intrinsics_rgb': intrinsics_rgb, 'tgt_rgb': imgs_rgb[i], 'ref_rgb_imgs': [],
                          'extrinsics_thr2rgb' : extrinsics_thr2rgb}
                for j in shifts:
                    sample['ref_thr_imgs'].append(imgs_thr[i+j])
                    sample['ref_rgb_imgs'].append(imgs_rgb[i+j])
                sequence_set.append(sample)

        random.shuffle(sequence_set)
        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_thr_img   = np.expand_dims(load_as_float(sample['tgt_thr']), axis=2)
        tgt_rgb_img   = load_as_float(sample['tgt_rgb'])

        tgt_trgb_img  = np.concatenate((tgt_thr_img,tgt_rgb_img), axis=2)
        ref_trgb_imgs = [np.concatenate((np.expand_dims(load_as_float(ref_thr_img), axis=2), load_as_float(ref_rgb_img)), axis=2) \
                        for ref_thr_img, ref_rgb_img in zip(sample['ref_thr_imgs'], sample['ref_rgb_imgs'])]
       
        imgs_trgb, intrinsics_trgb = self.tf_share([tgt_trgb_img] + ref_trgb_imgs, \
                                                    np.stack((np.copy(sample['intrinsics_thr']), np.copy(sample['intrinsics_rgb'])), axis=0) )
        
        imgs_thr_ = [im[:,:,[0]] for im in imgs_trgb]
        imgs_rgb = [im[:,:,1:] for im in imgs_trgb]

        intrinsics_thr = intrinsics_trgb[0,:,:]
        intrinsics_rgb = intrinsics_trgb[1,:,:]
        extrinsics_thr2rgb = sample['extrinsics_thr2rgb']
        
        imgs_rgb, _     = self.tf_rgb(imgs_rgb, None)
        imgs_thr, _     = self.tf_thr(imgs_thr_, None)
        imgs_thr_clr, _ = self.tf_thr_color(imgs_thr_, None)

        tgt_rgb_img     = imgs_rgb[0]
        ref_rgb_imgs    = imgs_rgb[1:]
       
        tgt_thr_img     = imgs_thr[0]
        ref_thr_imgs    = imgs_thr[1:]

        tgt_thr_img_clr = imgs_thr_clr[0]
        ref_thr_img_clr = imgs_thr_clr[1:]

        return tgt_thr_img, ref_thr_imgs, tgt_thr_img_clr, ref_thr_img_clr, tgt_rgb_img, ref_rgb_imgs, \
               intrinsics_thr, intrinsics_rgb, extrinsics_thr2rgb

    def __len__(self):
        return len(self.samples)
