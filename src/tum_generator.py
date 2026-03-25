import os
import sys
import cv2 as cv
import numpy as np
import open3d as o3d
from src.helper import Helper

class TUMCamera:
    """TUM camera intrinsics presets."""

    class FREIBURG1:
        @staticmethod
        def intrinsics():
            return {
                "fx": 517.3,
                "fy": 516.5,
                "cx": 318.6,
                "cy": 255.3
            }

class TUMGenerator:
    """Generator for creating 3D meshes from TUM RGBD datasets."""

    @staticmethod
    def optimize_mesh(mesh):
        """Applies optimization to the extracted TSDF mesh."""
        print("Optimizing mesh geometry...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_unreferenced_vertices()

        return mesh

    @staticmethod
    def create_tsdf_volume(voxel_length=0.01, sdf_trunc=0.04):
        """Helper to initialize a blank TSDF Volume."""
        return o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

    @staticmethod
    def integrate_frame(volume, rgb, depth, depth_trunc, intrinsic_o3d, camera_to_world_matrix, depth_scale):
        """Helper to convert arrays and integrate them into the TSDF Volume."""
        color_o3d = o3d.geometry.Image(cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth)

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d,
            depth_o3d,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False
        )

        # Invert Camera-to-World (Pose) to World-to-Camera (Extrinsic)
        extrinsic = np.linalg.inv(camera_to_world_matrix)

        volume.integrate(rgbd, intrinsic_o3d, extrinsic)

    @staticmethod
    def extract_and_save(volume, output_file):
        """Helper to pull the mesh out of the volume, optimize, and save."""
        print("\nExtracting Mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh = TUMGenerator.optimize_mesh(mesh)

        o3d.io.write_triangle_mesh(output_file, mesh)
        print(f"Saved optimized mesh to {output_file}")
        o3d.visualization.draw_geometries([mesh])

    @staticmethod
    def read_tum_file_list(filename):
        """Read TUM dataset file list (rgb.txt, depth.txt, groundtruth.txt)."""
        data = {}

        with open(filename) as f:
            for line in f:
                if line.startswith("#"):
                    continue

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                timestamp = float(parts[0])
                data[timestamp] = parts[1:]

        return data

    @staticmethod
    def associate(first_list, second_list, max_difference=0.02):
        """Associate timestamps from two lists."""
        matches = []
        first_keys = sorted(first_list.keys())
        second_keys = sorted(second_list.keys())

        j = 0
        for ts in first_keys:
            while j + 1 < len(second_keys) and abs(second_keys[j + 1] - ts) <= abs(second_keys[j] - ts):
                j += 1
            if j < len(second_keys) and abs(second_keys[j] - ts) <= max_difference:
                matches.append((ts, second_keys[j]))

        return matches

    @staticmethod
    def load_tum_groundtruth(path):
        """
        Load poses from TUM groundtruth file.

        Format:
        timestamp tx ty tz qx qy qz qw
        """

        poses = []

        if not os.path.isfile(path):
            print("Groundtruth file not found")
            return poses

        with open(path, "r") as f:

            for line in f:

                if line.startswith("#"):
                    continue

                parts = line.strip().split()

                if len(parts) != 8:
                    continue

                _, tx, ty, tz, qx, qy, qz, qw = map(float, parts)

                R = Helper.quaternion_to_rotation_matrix(qx, qy, qz, qw)

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]

                poses.append(T)

        return poses

    @staticmethod
    def generate_from_tum(rgbd_video_dir, output_path=None, output_file="tum_mesh.ply", on_progress=None):
        """Generate fused point cloud using offline TUM dataset."""
        rgb_dir = os.path.join(rgbd_video_dir, "rgb")
        depth_dir = os.path.join(rgbd_video_dir, "depth")
        groundtruth_file = os.path.join(rgbd_video_dir, "groundtruth.txt")

        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
            print("Missing rgb/ or depth/ directories.")
            return

        # Use output_path if provided, otherwise use current directory
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, output_file)

        print("Loading poses...")
        poses = TUMGenerator.load_tum_groundtruth(groundtruth_file)
        if len(poses) == 0:
            print("No poses loaded.")
            return

        intr = TUMCamera.FREIBURG1.intrinsics()
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

        rgb_list = TUMGenerator.read_tum_file_list(os.path.join(rgbd_video_dir, "rgb.txt"))
        depth_list = TUMGenerator.read_tum_file_list(os.path.join(rgbd_video_dir, "depth.txt"))
        pose_list = TUMGenerator.read_tum_file_list(groundtruth_file)

        rgb_depth_matches = TUMGenerator.associate(rgb_list, depth_list)

        frames = []
        for rgb_ts, depth_ts in rgb_depth_matches:
            pose_ts = min(pose_list.keys(), key=lambda x: abs(x - rgb_ts))
            frames.append((rgb_ts, depth_ts, pose_ts))

        first = cv.imread(os.path.join(rgbd_video_dir, rgb_list[frames[0][0]][0]))
        height, width = first.shape[:2]

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        volume = TUMGenerator.create_tsdf_volume(voxel_length=0.01, sdf_trunc=0.04)

        sample_rate = 5
        frame_indices = range(0, len(frames), sample_rate)
        total_samples = len(list(frame_indices))
        processed = 0

        print("Integrating TUM frames...")
        for i in frame_indices:
            processed += 1
            sys.stdout.write(f"\rIntegrating frame {processed}/{total_samples}")
            sys.stdout.flush()
            if on_progress: on_progress(processed, total_samples)

            rgb_ts, depth_ts, pose_ts = frames[i]
            rgb = cv.imread(os.path.join(rgbd_video_dir, rgb_list[rgb_ts][0]))
            depth = cv.imread(os.path.join(rgbd_video_dir, depth_list[depth_ts][0]), cv.IMREAD_UNCHANGED)

            if rgb is None or depth is None: continue

            pose_data = list(map(float, pose_list[pose_ts]))
            tx, ty, tz = pose_data[0:3]
            qx, qy, qz, qw = pose_data[3:7]
            R = Helper.quaternion_to_rotation_matrix(qx, qy, qz, qw)

            camera_to_world = np.eye(4)
            camera_to_world[:3, :3] = R
            camera_to_world[:3, 3] = [tx, ty, tz]

            TUMGenerator.integrate_frame(
                volume, rgb, depth, 8, intrinsic_o3d, camera_to_world, depth_scale=5000.0
            )

        TUMGenerator.extract_and_save(volume, output_file)
