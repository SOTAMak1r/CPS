#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import ssim, l1_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_tensor(voxel_size, world_size):
    N = voxel_size
    interval = N - 1
    grid_unseen_xyz = torch.zeros((N))
    for i in range(N):
        grid_unseen_xyz[i] = i
    grid_unseen_xyz -= (interval / 2.0)
    grid_unseen_xyz /= (interval / 2.0) / world_size
    idx_dict = {}
    for i in range(N):
        idx_dict[grid_unseen_xyz[i].item()] = i
    
    return idx_dict
    

idx_dict = get_tensor(32, 0.5)


def training(
    dataset, opt, pipe, 
    testing_iterations, saving_iterations, checkpoint_iterations, 
    checkpoint, 
    debug_from, 
    prune, prune_points, 
    voxel_size, 
    prepare_data
):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset, prepare_data)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, voxel_size=voxel_size, prepare_data=prepare_data)
    gaussians.training_setup(opt)

    if checkpoint:
        gaussians.load_trained_pth(checkpoint)
        
    bg_color = [0, 0, 0]
    if dataset.white_background == 1:
        bg_color = [1, 1, 1]
    elif dataset.white_background == 2:
        bg_color = [142 / 255.0, 133 / 255.0, 131 / 255.0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    ## modified
    unseen_mask = torch.ones((32**3, 1)).to("cuda")

    for iteration in range(first_iter, opt.iterations + 1): 

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        dist_threshold = 1.5 / (32 - 1)
        loss = loss + F.relu(torch.abs(gaussians._xyz_offsets) - dist_threshold).mean() * 20
        loss.backward()

        iter_end.record()

        # MODIFIED
        if iteration > opt.iterations - 1000:
            THRESHOLD = 1e-3

            feature = gaussians._rotation
            seen_gaussian = (torch.abs(feature.grad) * 1).max(dim=-1)[0]
            unseen_mask[seen_gaussian > THRESHOLD] = 0

            feature = gaussians._features_dc
            seen_gaussian = (torch.abs(feature.grad) * 1).max(dim=-1)[0]
            unseen_mask[seen_gaussian > THRESHOLD] = 0

            feature = gaussians._scaling
            seen_gaussian = (torch.abs(feature.grad) * 1).max(dim=-1)[0]
            unseen_mask[seen_gaussian > THRESHOLD] = 0

            feature = gaussians._opacity
            seen_gaussian = (torch.abs(feature.grad) * 1).max(dim=-1)[0]
            unseen_mask[seen_gaussian > THRESHOLD] = 0

            feature = gaussians._xyz_offsets
            seen_gaussian = (torch.abs(feature.grad) * 1).max(dim=-1)[0]
            unseen_mask[seen_gaussian > THRESHOLD] = 0

            if iteration % 500 == 0:
                print(f'[INFO] unseen_mask = {iteration, unseen_mask.sum().item()}')


        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.prune_and_densify(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, prepare_data)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background > 0 and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            elif iteration == opt.densify_until_iter:
                gaussians.check_full_points()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration >= opt.densify_until_iter and iteration % 100 == 1:
                    gaussians.reg_range(False)
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                if not prepare_data:
                    torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                else:
                    active_sh_degree, xyz, xyz_offsets, features_dc, features_rest, scaling, rotation, opacity, max_radii2D, xyz_gradient_accum, denom, opt_state, spatial_lr_scale = \
                        gaussians.capture()
                    
                    size = 32
                    idx_dict = get_tensor(size, 0.5)
                    xyz         = xyz #.reshape((size, size, size, -1))
                    xyz_offsets = xyz_offsets.reshape((size, size, size, -1))
                    f_dc        = features_dc.reshape((size, size, size, -1))
                    scale       = scaling.reshape((size, size, size, -1))
                    rotation    = rotation.reshape((size, size, size, -1))
                    opacity     = opacity.reshape((size, size, size, -1))
                    
                    features = torch.cat((xyz_offsets, f_dc, scale, rotation, opacity), dim=-1).reshape(size**3, -1).detach().cpu()
                    volume = torch.zeros((size, size, size, features.shape[-1]))
                    mask = torch.zeros((size, size, size, unseen_mask.shape[-1]))

                    idx = [0, 0, 0]
                    for i in range(xyz.shape[0]):
                        for j in range(xyz.shape[1]):
                            idx[j] = idx_dict[xyz[i, j].item()]
                        volume[idx[0], idx[1], idx[2]] = features[i]
                        mask[idx[0], idx[1], idx[2]]   = unseen_mask[i]

                    torch.save(mask, scene.model_path + "/seen_mask.pth")
                    torch.save(volume, scene.model_path + "/train_mask.pth")


def prepare_output_and_logger(args, prepare_data):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    if not prepare_data:
        with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
            cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if not prepare_data:
        if TENSORBOARD_FOUND:
            tb_writer = SummaryWriter(args.model_path)
        else:
            print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer, 
    iteration, 
    Ll1, loss, l1_loss, 
    elapsed, 
    testing_iterations, 
    scene : Scene, 
    renderFunc, renderArgs
):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : None}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--prepare_data', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[ 20_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--prune_points", type=int, default=8192)
    parser.add_argument('--prune', action='store_true', default=False)
    parser.add_argument('--voxel_size', type=int, default=32)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.prune, args.prune_points, args.voxel_size, args.prepare_data)

    # All done
    print("\nTraining complete.")
