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
import sys
import glob
import torch
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, fov2focal_head, focal2fov_head
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    # ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
def load_initial_ply(xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    plydata = PlyData([vertex_element])
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)
def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]
            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = [0, 0, 0]
            if white_background == 1:
                bg = [1, 1, 1]
            elif white_background == 2:
                bg = [142 / 255.0, 133 / 255.0, 131 / 255.0]

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms_head(path, file, white_background):
    cam_infos = []

    fovx = 18.837 * 3.14159 / 180 # contents["camera_angle_x"]

    for idx in range(1, len(file)):
        # if idx > 28 and idx < 35:
        #     continue
        cam_name = str(idx).zfill(3) + '.png'

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(file[idx].cpu().reshape((4,4)))

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = [0, 0, 0]
        if white_background == 1:
            bg = [1, 1, 1]
        elif white_background == 2:
            bg = [142 / 255.0, 133 / 255.0, 131 / 255.0]

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovy = focal2fov_head(fov2focal_head(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readHeadInfo(path, white_background):
    cameras_file = torch.load(os.path.join(path, "cameras.pth"))
    train_cam_infos = readCamerasFromTransforms_head(path, cameras_file, white_background)
    print("Reading Test Transforms")
    test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3 # TODO: CHECK
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info



def readCamerasFromTransforms_objaverse(path, white_background, extension=".png"):
    cam_infos = []
    image_num = len(glob.glob(os.path.join(path, "*.png")))
    fovx = 49.1 * 3.14159 / 180
    for idx in range(image_num):
        # if idx > 28 and idx < 35:
        #     continue
        cam_name = os.path.join(path, str(idx).zfill(3) + '.npy')
        
#         # NeRF 'transform_matrix' is a camera-to-world transform
#         c2w = np.array(frame["transform_matrix"])
#         # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
#         c2w[:3, 1:3] *= -1

#         # get the world-to-camera transform and set R, T
#         w2c = np.linalg.inv(c2w)
#         R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
#         T = w2c[:3, 3]
        w2c = np.load(cam_name)
        R = np.transpose(w2c[:3,:3])
        R[:3, 1:3] *= -1
        T = w2c[:3, 3]
        T[1:3] *= -1
        image_path = os.path.join(path, cam_name.replace('.npy', extension))
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = [0, 0, 0]
        if white_background == 1:
            bg = [1, 1, 1]
        elif white_background == 2:
            bg = [142 / 255.0, 133 / 255.0, 131 / 255.0]

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos
# def get_grid(voxel_size, world_size): # 3, 0.5
#     N = voxel_size
#     grid_unseen_xyz = torch.zeros((N, N, N, 3))
#     for i in range(N): # 0, 1, 2
#         grid_unseen_xyz[i, :, :, 0] = i
#     for j in range(N):
#         grid_unseen_xyz[:, j, :, 1] = j
#     for k in range(N):
#         grid_unseen_xyz[:, :, k, 2] = k
#     grid_unseen_xyz -= (N / 2.0) # -1.5, -0.5, 0.5
#     grid_unseen_xyz /= (N / 2.0) / world_size # -0.5 -1/6 1/6 0.5 
#     # grid_unseen_xyz *= 2.0 * world_size / (N - 1)
#     # grid_unseen_xyz -= world_size
#     grid_unseen_xyz = grid_unseen_xyz.reshape((-1, 3))
#     return grid_unseen_xyz

def get_grid(voxel_size, world_size):
    N = voxel_size
    interval = N - 1
    grid_unseen_xyz = torch.zeros((N, N, N, 3))
    for i in range(N):
        grid_unseen_xyz[i, :, :, 0] = i
    for j in range(N):
        grid_unseen_xyz[:, j, :, 1] = j
    for k in range(N):
        grid_unseen_xyz[:, :, k, 2] = k
    grid_unseen_xyz -= (interval / 2.0)
    grid_unseen_xyz /= (interval / 2.0) / world_size
    grid_unseen_xyz = grid_unseen_xyz.reshape((-1, 3))
    return grid_unseen_xyz


def readCamerasFromTransforms_objaverse_cjyversion(path, white_background, extension='.png'):

    def get_camera_positions_on_sphere(radius, center_point, num_cameras):
        cameras = []
        phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

        for i in range(num_cameras):

            x = 1 - (i / float(num_cameras - 1)) * 2  # y goes from 1 to -1
            radius_at_height = np.sqrt(1 - x * x)  # Radius at height y

            theta = phi * i /  num_cameras * 8 # Golden angle increment

            z = np.cos(theta) * radius_at_height
            y = np.sin(theta) * radius_at_height

            camera_position = np.array([x, y, z]) * radius + np.array(center_point)
            cameras.append(camera_position)

        return cameras

    def get_rotation_matrix_to_look_at(target_point, camera_position):
        forward = target_point - camera_position
        forward = forward / np.linalg.norm(forward)

        right = np.array([0, 1, 0])  # Assume up direction is positive y-axis
        right = right - np.dot(right, forward) * forward
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        rotation_matrix = np.column_stack((right, up, -forward))

        return rotation_matrix

    # 设置参数
    R = 0.02  # 球的半径
    center_point = (0, 0, 2)  # 对准点的坐标
    image_num = 2  # 相机数量 ## modified

    camera_positions = get_camera_positions_on_sphere(R, center_point, image_num)

    target_point = np.array(center_point)

    cam_infos = []
    
    fovx = 49.1 * 3.14159 / 180
    for idx in range(image_num):
        cam_name = os.path.join(path, str(idx).zfill(3) + '.npy')
        

        rotation_matrix = get_rotation_matrix_to_look_at(target_point, camera_positions[idx])
        R = rotation_matrix
        
        T = camera_positions[idx]
        
        image_path = os.path.join(path, str(1).zfill(3) + extension)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = [0, 0, 0]
        if white_background == 1:
            bg = [1, 1, 1]
        elif white_background == 2:
            bg = [142 / 255.0, 133 / 255.0, 131 / 255.0]

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(
            uid=idx, 
            R=R, T=T, 
            FovY=FovY, FovX=FovX, 
            image=image, image_path=image_path, image_name=image_name, 
            width=image.size[0], height=image.size[1]))
    return cam_infos

def get_test_poses(path, white_background, extension='.png'):

    # def get_camera_positions_on_sphere(radius, center_point, num_cameras):
    #     cameras = []
    #     phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

    #     for i in range(num_cameras):

    #         x = 1 - (i / float(num_cameras - 1)) * 2  # y goes from 1 to -1
    #         radius_at_height = np.sqrt(1 - x * x)  # Radius at height y

    #         theta = phi * i /  num_cameras * 8 # Golden angle increment

    #         z = np.cos(theta) * radius_at_height
    #         y = np.sin(theta) * radius_at_height

    #         camera_position = np.array([x, y, z]) * radius + np.array(center_point)
    #         cameras.append(camera_position)

    #     return cameras

    def get_camera_positions_on_sphere(radius, center_point, num_cameras):
        cameras = []
        # phi = np.pi * (3 - np.sqrt(5))  # Golden angle in radians

        for i in range(num_cameras):

            # x = 1 - (i / float(num_cameras - 1)) * 2  # y goes from 1 to -1
            x = -0.5
            radius_at_height = np.sqrt(1 - x * x)  # Radius at height y

            # theta = phi * i /  num_cameras * 8 # Golden angle increment
            theta = np.pi * 2 * i /  num_cameras # Golden angle increment

            z = np.cos(theta) * radius_at_height
            y = np.sin(theta) * radius_at_height

            # camera_position = np.array([x, y, z]) * radius + np.array(center_point)
            # camera_position = np.array([y, x, z]) * radius + np.array(center_point)
            # camera_position = np.array([z, y, x]) * radius + np.array(center_point)
            camera_position = np.array([y, z, x]) * radius + np.array(center_point)
            cameras.append(camera_position)

        return cameras

    def get_rotation_matrix_to_look_at(target_point, camera_position):
        forward = target_point - camera_position
        forward = forward / np.linalg.norm(forward)

        # right = np.array([0, 1, 0])  # Assume up direction is positive y-axis
        right = np.array([0, 0, 1])  # Assume up direction is positive z-axis
        right = right - np.dot(right, forward) * forward
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        rotation_matrix = np.column_stack((right, up, -forward))

        return rotation_matrix

    # 设置参数
    R = 0.02  # 球的半径
    center_point = (0, 0, 2)  # 对准点的坐标
    image_num = 60  # 相机数量

    camera_positions = get_camera_positions_on_sphere(R, center_point, image_num)

    target_point = np.array(center_point)

    cam_infos = []
    
    fovx = 49.1 * 3.14159 / 180
    fovx = 40 * 3.14159 / 180
    for idx in range(image_num):
        cam_name = os.path.join(path, str(idx).zfill(3) + '.npy')
        rotation_matrix = get_rotation_matrix_to_look_at(target_point, camera_positions[idx])
        R = rotation_matrix
        T = camera_positions[idx]
        fovy = focal2fov(fov2focal(fovx, 512), 512)
        FovY = fovy 
        FovX = fovx
        cam = np.zeros((1, 4, 4))
        cam[0, :3, :3] = R
        cam[0, -1, :3] = T
        cam[0, -1, -1] = FovX

        image_path = os.path.join(path, cam_name.replace('.npy', extension))
        # image_path = os.path.join(path, str(1).zfill(3) + '.png')
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = [0, 0, 0]
        if white_background == 1:
            bg = [1, 1, 1]
        elif white_background == 2:
            bg = [142 / 255.0, 133 / 255.0, 131 / 255.0]

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        # T[0] = T[1] = 0
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos

def readObjaverseInfo(path, white_background, voxel_size=32):
    train_cam_infos = get_test_poses(path, white_background)
    test_cam_infos = readCamerasFromTransforms_objaverse_cjyversion(path, white_background)

    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    num_pts = voxel_size ** 3
    print(f"Generating random point cloud ({num_pts})...")
    xyz = get_grid(voxel_size, world_size=0.5)
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    pcd = load_initial_ply(xyz, SH2RGB(shs) * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                        )
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Head": readHeadInfo,
    "Objaverse": readObjaverseInfo
}