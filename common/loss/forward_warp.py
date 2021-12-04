from __future__ import division
import torch
import torch.nn.functional as F
from .flow_reversal import FlowReversal
from torch.autograd import Variable

flow_reverse = FlowReversal()
pixel_coords = None

def set_id_grid(depth):
    global pixel_coords
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(
        1, h, w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1, h, w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    if (pixel_coords is None) or pixel_coords.size(2) < h:
        set_id_grid(depth)
    current_pixel_coords = pixel_coords[:, :, :h, :w].expand(
        b, 3, h, w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv @ current_pixel_coords).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)


def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr, padding_mode):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 4, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)

    # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    X_norm = 2*(X / Z)/(w-1) - 1
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b, h, w, 2)


def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat

def pose2flow(depth, pose_mat, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode=None):
    """
    Converts pose parameters to rigid optical flow
    """
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(pose_mat, 'pose_mat', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    check_sizes(intrinsics_inv, 'intrinsics', 'B33')
    assert(intrinsics_inv.size() == intrinsics.size())

    bs, h, w = depth.size()

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(depth).expand_as(depth)  # [bs, H, W]

    cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]
#    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]

    # Get projection matrix for tgt camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]
    src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:], padding_mode)  # [B,H,W,2]

    X = (w-1)*(src_pixel_coords[:,:,:,0]/2.0 + 0.5) - grid_x
    Y = (h-1)*(src_pixel_coords[:,:,:,1]/2.0 + 0.5) - grid_y

    return torch.stack((X,Y), dim=1)

def forward_warp(img, depth, pose_mat, intrinsics, intrinsics_inv, padding_mode='zeros'):
    """
    Forward warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the source image -- [B, H, W]
        pose: 6DoF pose parameters from source to target -- [B, 6]
        intrinsics: camera intrinsic matrix of target camera -- [B, 3, 3]
        intrinsics_inv: intrinsics_inv matrix of source camera -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """

    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(pose_mat, 'pose_mat', 'B34')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()
    rigid_flow_s2t = pose2flow(depth.squeeze(1), pose_mat, intrinsics, intrinsics_inv) # get rigid flow from source to target 

    # Source image warping with reversed source flow map
    rigid_flow_t2s, norm = flow_reverse(rigid_flow_s2t, rigid_flow_s2t)
    rigid_flow_t2s = -rigid_flow_t2s
    rigid_flow_t2s[norm > 0] = rigid_flow_t2s[norm > 0]/norm[norm>0].clone()

    bs, _, h, w = rigid_flow_t2s.size()

    u = rigid_flow_t2s[:,0,:,:]
    v = rigid_flow_t2s[:,1,:,:]

    grid_x = Variable(torch.arange(0, w).view(1, 1, w).expand(1,h,w), requires_grad=False).type_as(u).expand_as(u)  # [bs, H, W]
    grid_y = Variable(torch.arange(0, h).view(1, h, 1).expand(1,h,w), requires_grad=False).type_as(v).expand_as(v)  # [bs, H, W]

    X = grid_x + u
    Y = grid_y + v

    X = 2*(X/(w-1.0) - 0.5)
    Y = 2*(Y/(h-1.0) - 0.5)
    grid_tf = torch.stack((X,Y), dim=3)
    projected_img = F.grid_sample(img, grid_tf, padding_mode=padding_mode)

    valid_points = grid_tf.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points, rigid_flow_s2t, rigid_flow_t2s

def compute_forward_warp(imgs, depths, poses, intrinsics_tgt, intrinsics_src, max_scales, padding_mode='zeros', vis_flag=False):
    """
    Forward warp a source image to the target image plane.
    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the source image -- [B, H, W]
        pose: 6DoF pose parameters from source to target -- [B, 6]
        intrinsics: camera intrinsic matrix of target camera -- [B, 3, 3]
        intrinsics_inv: intrinsics_inv matrix of source camera -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    num_scales = min(len(depths), max_scales)
    b, _, h, w = imgs[0].size()
    warped_imgs = []
    for s in range(num_scales):
        downscale = imgs[s].size(2)/h

        img_scaled = imgs[s]
        depth_scaled = depths[s]
        
        intrinsic_src_scaled = torch.cat((intrinsics_src[:, 0:2]/downscale, intrinsics_src[:, 2:]), dim=1)
        intrinsic_src_scaled = intrinsic_src_scaled.inverse()
        intrinsic_tgt_scaled = torch.cat((intrinsics_tgt[:, 0:2]/downscale, intrinsics_tgt[:, 2:]), dim=1)

        warped_img, valid_mask, rigid_flow_s2t, rigid_flow_t2s  = forward_warp(img_scaled, depth_scaled, poses, intrinsic_tgt_scaled, intrinsic_src_scaled, padding_mode=padding_mode)
        warped_imgs.append(warped_img)

        if vis_flag : 
            valid_masks.append(valid_mask)
            rigid_flow_s2ts.append(rigid_flow_s2t)
            rigid_flow_t2ss.append(rigid_flow_t2s)
            return warped_imgs, valid_masks, rigid_flow_s2ts, rigid_flow_t2ss
        else:
            return warped_imgs

def compute_warp_pose(poses, poses_inv, extrinsics_thr2rgb):
    R_t2r = extrinsics_thr2rgb[:,0:3,0:3]
    t_t2r = extrinsics_thr2rgb[:,0:3,[3]]

    R_r2t = extrinsics_thr2rgb.inverse()[:,0:3,0:3]
    t_r2t = extrinsics_thr2rgb.inverse()[:,0:3,[3]]

    poses_thr = [pose_vec2mat(pose) for pose in poses]
    poses_thr_inv = [pose_vec2mat(pose_inv) for pose_inv in poses_inv]
    
    poses_rgb = []
    poses_rgb_inv = []
    for pose_thr, pose_thr_inv in zip(poses_thr, poses_thr_inv):
        rot  = pose_thr[:,:,0:3]
        tran = pose_thr[:,:,[3]]
        rot_mat = R_t2r@rot@R_r2t
        translation = R_t2r@rot@t_r2t + R_t2r@tran + t_t2r
        poses_rgb.append(torch.cat([rot_mat,translation], dim=2))

        rot  = pose_thr_inv[:,:,0:3]
        tran = pose_thr_inv[:,:,[3]]
        rot_mat = R_t2r@rot@R_r2t
        translation = R_t2r@rot@t_r2t + R_t2r@tran + t_t2r
        poses_rgb_inv.append(torch.cat([rot_mat,translation], dim=2))

    return poses_rgb, poses_rgb_inv