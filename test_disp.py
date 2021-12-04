import torch
from skimage.transform import resize as imresize
from imageio import imread
import numpy as np
from path import Path
import argparse
from tqdm import tqdm
import time

import sys
sys.path.append('./common/')
import models
from utils.custom_transforms import Celsius2Raw, Raw2Celsius

parser = argparse.ArgumentParser(description='Script for DispNet testing with corresponding groundTruth',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-dispnet", required=True, type=str, help="pretrained DispNet path")
parser.add_argument("--img-height", default=256, type=int, help="Image height") # 256 (kitti)
parser.add_argument("--img-width", default=320, type=int, help="Image width")   # 832 (kitti)
parser.add_argument("--min-depth", default=1e-3)
parser.add_argument("--max-depth", default=80)
parser.add_argument("--dataset-dir", default='.', type=str, help="Dataset directory")
parser.add_argument("--dataset-list", default=None, type=str, help="Dataset list file")
parser.add_argument("--output-dir", default=None, required=True, type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument('--resnet-layers', required=True, type=int, default=18, choices=[18, 50], help='depth network architecture.')
parser.add_argument('--input', type=str, choices=['RGB', 'T'], default='T', help='input data type')
parser.add_argument('--scene_type', type=str, choices=['indoor', 'outdoor'], default='indoor', required=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
def load_tensor_image(filename, args):
    img = imread(filename).astype(np.float32)
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img

def load_tensor_Timage_indoor(filename, args):
    img = np.expand_dims(imread(filename).astype(np.float32), axis=2)
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    Dmin = Celsius2Raw(10)
    Dmax = Celsius2Raw(40)
    img[img<Dmin] = Dmin
    img[img>Dmax] = Dmax
    img = (torch.from_numpy(img).float() - Dmin)/(Dmax - Dmin) # thermal data clip into 30~50 degree clip
    tensor_img = ((img.unsqueeze(0)-0.45)/0.225).to(device)
    return tensor_img

def load_tensor_Timage_outdoor(filename, args):
    img = np.expand_dims(imread(filename).astype(np.float32), axis=2)
    h,w,_ = img.shape
    if (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    Dmin = Celsius2Raw(0)
    Dmax = Celsius2Raw(30)
    img[img<Dmin] = Dmin
    img[img>Dmax] = Dmax
    img = (torch.from_numpy(img).float() - Dmin)/(Dmax - Dmin) # thermal data clip into 30~50 degree clip
    tensor_img = ((img.unsqueeze(0)-0.45)/0.225).to(device)
    return tensor_img

@torch.no_grad()
def main():
    args = parser.parse_args()

    # load models
    if args.input == 'RGB' :
        disp_net = models.DispResNet(args.resnet_layers, False, num_channel=3).to(device)
    else : 
        disp_net = models.DispResNet(args.resnet_layers, False, num_channel=1).to(device)

    weights = torch.load(args.pretrained_dispnet)
    disp_net.load_state_dict(weights['state_dict'])
    disp_net.eval()

    dataset_dir = Path(args.dataset_dir)

    if args.input == 'RGB' :
            load_tensor_img = load_tensor_image
    elif args.input == 'T':
        if args.scene_type == 'indoor' : #indoor
            load_tensor_img = load_tensor_Timage_indoor
        elif args.scene_type == 'outdoor' :
            load_tensor_img = load_tensor_Timage_outdoor

    # read file list
    if args.dataset_list is not None:
        with open(args.dataset_list, 'r') as f:
            test_files = list(f.read().splitlines())
    else:
        if args.input == 'RGB' :
            test_files=sorted((dataset_dir+'RGB').files('*.png'))
        else:
            test_files=sorted((dataset_dir+'Thermal').files('*.png'))

    print('{} files to test'.format(len(test_files)))
  
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_disp_avg = 0
    test_disp_std = 0
    test_depth_avg = 0
    test_depth_std = 0

    avg_time = 0
    for j in tqdm(range(len(test_files))):
        tgt_img = load_tensor_img(test_files[j], args)

        # tgt_img = load_tensor_image( dataset_dir + test_files[j], args)

        # compute speed
        torch.cuda.synchronize()
        t_start = time.time()

        output = disp_net(tgt_img)

        torch.cuda.synchronize()
        elapsed_time = time.time() - t_start
        
        avg_time += elapsed_time

        pred_disp = output.cpu().numpy()[0,0]

        if j == 0:
            predictions = np.zeros((len(test_files), *pred_disp.shape))
        predictions[j] = 1/pred_disp

        test_disp_avg += pred_disp.mean()
        test_disp_std += pred_disp.std()
        test_depth_avg += predictions.mean()
        test_depth_std += predictions.std()
        
    np.save(output_dir/'predictions.npy', predictions)

    avg_time /= len(test_files)
    print('Avg Time: ', avg_time, ' seconds.')
    print('Avg Speed: ', 1.0 / avg_time, ' fps')

    print('Avg disp : {0:0.3f}, std disp : {1:0.5f}'.format(test_disp_avg/len(test_files), test_disp_std/len(test_files)))
    print('Avg depth: {0:0.3f}, std depth: {1:0.5f}'.format(test_depth_avg/len(test_files), test_depth_std/len(test_files)))


if __name__ == '__main__':
    main()