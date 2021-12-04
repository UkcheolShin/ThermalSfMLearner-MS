import numpy as np
from path import Path
from imageio import imread
from tqdm import tqdm


class test_framework_VIVID(object):
    def __init__(self, root, sequence_set, seq_length=3, step=1, input_type='RGB'):
        self.root = root
        self.input_type = input_type
        self.img_files, self.poses, self.sample_indices = read_scene_data(self.root, sequence_set, seq_length, step, input_type)

    def generator(self):
        for img_list, pose_list, sample_list in zip(self.img_files, self.poses, self.sample_indices):
            for snippet_indices in sample_list:
                if self.input_type =='RGB':
                    imgs = [imread(img_list[i]).astype(np.float32) for i in snippet_indices]
                else:
                    imgs = [np.expand_dims(imread(img_list[i]).astype(np.float32), axis=2) for i in snippet_indices]

                poses = np.stack([pose_list[i] for i in snippet_indices])
                first_pose = poses[0]
                poses[:,:,-1] -= first_pose[:,-1]
                compensated_poses = np.linalg.inv(first_pose[:,:3]) @ poses

                yield {'imgs': imgs,
                       'path': img_list[0],
                       'poses': compensated_poses
                       }

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return sum(len(imgs) for imgs in self.img_files)


def read_scene_data(data_root, sequence_set, seq_length=3, step=1, input_type='RGB'):
    data_root = Path(data_root)
    im_sequences = []
    poses_sequences = []
    indices_sequences = []
    demi_length = (seq_length - 1) // 2
    shift_range = np.array([step*i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)

    sequences = []
    for seq in sequence_set:
        sequences.append((data_root/seq))

    print('getting test metadata for theses sequences : {}'.format(sequences))
    for sequence in tqdm(sequences):
        if input_type == 'RGB':
            imgs = sorted((sequence/'RGB').files('*.png'))
            poses = np.genfromtxt(sequence/'poses_RGB.txt').astype(np.float64).reshape(-1, 3, 4)
        else:
            imgs = sorted((sequence/'Thermal').files('*.png'))
            poses = np.genfromtxt(sequence/'poses_T.txt').astype(np.float64).reshape(-1, 3, 4)
   
        # construct 5-snippet sequences
        tgt_indices = np.arange(demi_length, len(imgs) - demi_length).reshape(-1, 1)
        snippet_indices = shift_range + tgt_indices
        im_sequences.append(imgs)
        poses_sequences.append(poses)
        indices_sequences.append(snippet_indices)
    return im_sequences, poses_sequences, indices_sequences
