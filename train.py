import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import argparse
import time
import datetime
from path import Path

from dataloader.VIVID_sequence_folders import SequenceFolder

import sys
sys.path.append('./common/')

import models
import utils.custom_transforms as custom_transforms
from utils.utils import tensor2array, save_checkpoint, tensor2array_thermal

from loss.forward_warp import compute_forward_warp, compute_warp_pose
from loss.inverse_warp import pose_vec2mat
from loss.loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors

# logging
import csv
from utils.logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
# optimizer param
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
# logger
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
# data loader
parser.add_argument('--scene_type', type=str, choices=['indoor', 'outdoor'], default='indoor', required=True)
parser.add_argument('--sequence_length', type=int, help='Length of sequence', metavar='N', default=3)
parser.add_argument('--interval', type=int, help='Interval of sequence', metavar='N', default=3)

parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')
# loss param
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('-pr', '--rgb-photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-cr', '--rgb-geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--rgb-ssim', type=float, default=0.85, help='weight for RGB ssim scaling factor')
parser.add_argument('--thr-ssim', type=float, default=0.15, help='weight for T ssim scaling factor')
parser.add_argument('-t', '--thermal-weight', type=float, help='weight for thempearture consistency loss', metavar='W', default=0.25)
parser.add_argument('-r', '--rgb-weight', type=float, help='weight for photometric consistency loss', metavar='W', default=1.0)

parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-thr-mask', type=int, default=1, help='with the thermal image mask for moving objects and occlusions or not')
parser.add_argument('--with-rgb-mask', type=int, default=1, help='with the visible image mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=1, help='with the mask for stationary points')

best_depth_error = -1
best_pose_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def main():
    global best_depth_error, best_pose_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # set dataloader
    if args.scene_type == 'indoor' :  # indoor
        # args.rgb_ssim = 0.85 # 'weight for RGB ssim scaling factor'
        # args.thr_ssim = 0.15 # 'weight for T ssim scaling factor'
        # args.thermal_weight = 0.25
        # args.rgb_weight = 1.0
        args.temp_min = 10
        args.temp_max = 40
        args.vis_depth_max = 4
    elif args.scene_type == 'outdoor' : # outdoor
        # args.rgb_ssim = 0.30 # 'weight for RGB ssim scaling factor'
        # args.thr_ssim = 0.85 # 'weight for T ssim scaling factor'
        # args.thermal_weight = 1.0
        # args.rgb_weight = 0.1
        args.temp_min = 0
        args.temp_max = 30
        args.vis_depth_max = 10

    ArrToTen_rgb = custom_transforms.ArrayToTensor()
    ArrToTen_thr = custom_transforms.ArrayToTensor_Thermal(args.temp_min, args.temp_max)
    TenColorize = custom_transforms.TensorColorize()
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])

    train_transform_share = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
    ])

    transform_thr_clr  = custom_transforms.Compose([ArrToTen_thr, TenColorize, normalize])
    transform_thr      = custom_transforms.Compose([ArrToTen_thr, normalize])
    transform_rgb      = custom_transforms.Compose([ArrToTen_rgb, normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    train_set = SequenceFolder(
        args.data,
        tf_share     = train_transform_share,
        tf_thr_color = transform_thr_clr,
        tf_thr       = transform_thr,
        tf_rgb       = transform_rgb,
        seed         = args.seed,
        scene_type   = args.scene_type,
        sequence_length = args.sequence_length,
        interval     = args.interval,
        train        = True
    )

    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from dataloader.VIVID_validation_folders import ValidationSet, ValidationSetPose
        val_set = ValidationSet(
            args.data,
            tf_thr       = transform_thr,
            sequence_length = args.sequence_length,
            interval     = args.interval,
            scene_type   = args.scene_type,
        )

        val_pose_set = ValidationSetPose(
            args.data,
            tf_thr       = transform_thr,
            scene_type   = args.scene_type,
            sequence_length = args.sequence_length,
        )
    else:
        val_set = SequenceFolder(
            args.data,
            tf_share=custom_transforms.Compose(),
            tf_thr_color = transform_thr_clr,
            tf_thr       = transform_thr,
            tf_rgb       = transform_rgb,
            seed         = args.seed,
            sequence_length = args.sequence_length,
            scene_type   = args.scene_type,
            Interval     = args.interval,
            train        = False
        )

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.folders)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.folders)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_pose_loader = torch.utils.data.DataLoader(
        val_pose_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain, num_channel=1).to(device)
    pose_net = models.PoseResNet(18, args.with_pretrain, num_channel=1).to(device)
    
    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    # set optimizer
    print('=> setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)
    # set logger
    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_depth_loss', 'validation_pose_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_pose_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()

        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            pose_errors, pose_error_names = validate_pose_with_gt(args, val_pose_loader, pose_net, epoch, logger, output_writers)
            depth_errors, detph_error_names = validate_depth_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
            errors = depth_errors + pose_errors
            error_names = detph_error_names + pose_error_names
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_depth_error = errors[1] # ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3', 'ATE', 'RE']
        decisive_pose_error = errors[6] # ATE

        if best_depth_error < 0:
            best_depth_error = decisive_depth_error

        if best_pose_error < 0:
            best_pose_error = decisive_pose_error

        # remember lowest error and save checkpoint
        is_depth_best = decisive_depth_error < best_depth_error
        best_depth_error = min(best_depth_error, decisive_depth_error)

        is_pose_best = decisive_pose_error < best_pose_error
        best_pose_error = min(best_pose_error, decisive_pose_error)

        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_depth_best, is_pose_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_depth_error, decisive_pose_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time   = AverageMeter()
    data_time    = AverageMeter()
    losses       = AverageMeter(precision=4)
    w1, w2, w3   = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    w4, w5       = args.rgb_photo_loss_weight, args.rgb_geometry_consistency_weight
    w_thr, w_rgb = args.thermal_weight, args.rgb_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)


    for i, (tgt_thr_img, ref_thr_imgs, tgt_thr_img_clr, ref_thr_img_clr, tgt_rgb_img, ref_rgb_imgs, \
            intrinsics_thr, intrinsics_rgb, extrinsics_thr2rgb) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)

        tgt_thr_img = tgt_thr_img.to(device)
        ref_thr_imgs = [img.to(device) for img in ref_thr_imgs]

        tgt_thr_img_clr = tgt_thr_img_clr.to(device)
        ref_thr_img_clr = [img.to(device) for img in ref_thr_img_clr]

        tgt_rgb_img = tgt_rgb_img.to(device)
        ref_rgb_imgs = [img.to(device) for img in ref_rgb_imgs]

        intrinsics_thr = intrinsics_thr.to(device)
        intrinsics_rgb = intrinsics_rgb.to(device)
        extrinsics_thr2rgb = extrinsics_thr2rgb.to(device)

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_thr_img, ref_thr_imgs)
        poses, poses_inv      = compute_pose_with_inv(pose_net, tgt_thr_img, ref_thr_imgs)

        # tempearture consistency loss
        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_thr_img_clr, ref_thr_img_clr, intrinsics_thr, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.thr_ssim, args.with_ssim,
                                                         args.with_thr_mask, args.with_auto_mask, args.padding_mode)

        # smoothness loss, not necessary for thermal image
        loss_2 = compute_smooth_loss(tgt_depth, tgt_thr_img, ref_depths, ref_thr_imgs)

        # depth and pose in rgb image plane 
        tgt_depth_rgb = compute_forward_warp(tgt_depth, tgt_depth, extrinsics_thr2rgb[:,0:3,:], intrinsics_rgb, intrinsics_thr, args.num_scales, args.padding_mode)
        ref_depths_rgb = [compute_forward_warp(ref_depth, ref_depth, extrinsics_thr2rgb[:,0:3,:], intrinsics_rgb, intrinsics_thr, args.num_scales, args.padding_mode) for ref_depth in ref_depths]
        poses_rgb, poses_rgb_inv = compute_warp_pose(poses, poses_inv, extrinsics_thr2rgb)

        # photometric consistency loss
        loss_4, loss_5 = compute_photo_and_geometry_loss(tgt_rgb_img, ref_rgb_imgs, intrinsics_rgb, tgt_depth_rgb, ref_depths_rgb,
                                                         poses_rgb, poses_rgb_inv, args.num_scales, args.rgb_ssim, args.with_ssim,
                                                         args.with_rgb_mask, args.with_auto_mask, args.padding_mode)

        loss = w_thr*w1*loss_1 + w2*loss_2 + w_thr*w3*loss_3 + w_rgb*w1*loss_4 + w_rgb*w3*loss_5 

        if log_losses:
            train_writer.add_scalar('image_recon_loss_thermal'         , loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss'        , loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss_thermal', loss_3.item(), n_iter)
            train_writer.add_scalar('image_recon_loss_visible'         , loss_4.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss_rgb'    , loss_5.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_depth_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    for i, (tgt_thr_img, depth_thr) in enumerate(val_loader):

        # original validate_with_gt param
        tgt_thr_img = tgt_thr_img.to(device)
        depth = depth_thr.to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_thr_img)
        output_depth = 1/output_disp

        output_depth = output_depth[:, 0]

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input_T',   tensor2array_thermal(tgt_thr_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image('val target Depth_T', tensor2array(depth_to_show, max_value=args.vis_depth_max), epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                output_writers[i].add_image('val target Disparity Normalized_T', tensor2array(disp_to_show, max_value=None, colormap='magma'), epoch)

            output_writers[i].add_image('val Dispnet Output Normalized_T', tensor2array(output_disp[0], max_value=None, colormap='magma'), epoch)
            output_writers[i].add_image('val Depth Output_T', tensor2array(output_depth[0], max_value=args.vis_depth_max), epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.scene_type))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names


@torch.no_grad()
def validate_pose_with_gt(args, val_pose_loader, pose_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['ATE', 'RE']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    for i, (thr_imgs, poses_gt) in enumerate(val_pose_loader):
        # original validate_with_gt param
        thr_imgs = [img.to(device) for img in thr_imgs]

        # compute output    
        global_pose = np.eye(4)
        poses = []
        poses.append(global_pose[0:3, :])
        for j in range(len(thr_imgs)-1):
            pose = pose_net(thr_imgs[j], thr_imgs[j + 1])
            if pose.shape[0] == 1:
                pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
            else:
                pose_mat = pose_vec2mat(pose[[0]]).squeeze(0).cpu().numpy()

            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
            global_pose = global_pose @  np.linalg.inv(pose_mat)
            poses.append(global_pose[0:3, :])

        final_poses = np.stack(poses, axis=0)

        errors.update(compute_pose_error(poses_gt[0].numpy(), final_poses))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} ATE Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_pose_loader))
    return errors.avg, error_names


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv

@torch.no_grad()
def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return [metric.item() for metric in [ATE/snippet_length, RE/snippet_length]]

if __name__ == '__main__':
    main()
