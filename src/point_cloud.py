import os
import sys
import cv2 as cv
import numpy as np
import open3d as o3d


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


class CameraProperties:
    """Utilities for loading camera poses."""

    @staticmethod
    def quaternion_to_rotation_matrix(qx, qy, qz, qw):
        """Convert quaternion to rotation matrix."""

        R = np.array([
            [
                1 - 2*qy*qy - 2*qz*qz,
                2*qx*qy - 2*qz*qw,
                2*qx*qz + 2*qy*qw
            ],
            [
                2*qx*qy + 2*qz*qw,
                1 - 2*qx*qx - 2*qz*qz,
                2*qy*qz - 2*qx*qw
            ],
            [
                2*qx*qz - 2*qy*qw,
                2*qy*qz + 2*qx*qw,
                1 - 2*qx*qx - 2*qy*qy
            ]
        ])

        return R

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

                R = CameraProperties.quaternion_to_rotation_matrix(
                    qx, qy, qz, qw
                )

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]

                poses.append(T)

        return poses


class PointCloudGenerator:

    @staticmethod
    def read_tum_file_list(filename):
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
    def generate_from_video(rgbd_video_dir):
        """Generate fused point cloud using TSDF."""

        rgb_dir = os.path.join(rgbd_video_dir, "rgb")
        depth_dir = os.path.join(rgbd_video_dir, "depth")
        groundtruth_file = os.path.join(rgbd_video_dir, "groundtruth.txt")

        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
            print("Missing rgb/ or depth/ directories.")
            return

        print("Loading poses...")
        poses = CameraProperties.load_tum_groundtruth(groundtruth_file)

        if len(poses) == 0:
            print("No poses loaded.")
            return

        intr = TUMCamera.FREIBURG1.intrinsics()

        fx = intr["fx"]
        fy = intr["fy"]
        cx = intr["cx"]
        cy = intr["cy"]

        rgb_list = PointCloudGenerator.read_tum_file_list(
            os.path.join(rgbd_video_dir, "rgb.txt")
        )
        depth_list = PointCloudGenerator.read_tum_file_list(
            os.path.join(rgbd_video_dir, "depth.txt")
        )
        pose_list = PointCloudGenerator.read_tum_file_list(
            os.path.join(rgbd_video_dir, "groundtruth.txt")
        )

        rgb_depth_matches = PointCloudGenerator.associate(rgb_list, depth_list)

        frames = []
        for rgb_ts, depth_ts in rgb_depth_matches:
            pose_ts = min(
                pose_list.keys(),
                key=lambda x: abs(x - rgb_ts)
            )
            frames.append((rgb_ts, depth_ts, pose_ts))

        total = len(frames)
        print(f"Frames available: {total}")

        first = cv.imread(os.path.join(rgbd_video_dir, rgb_list[frames[0][0]][0]))
        height, width = first.shape[:2]

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        sample_rate = 5

        voxel_length = 0.01
        sdf_trunc = 0.04

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        frame_indices = range(0, len(frames), sample_rate)

        print("Integrating frames...")

        processed = 0
        total_samples = len(list(frame_indices))

        for i in frame_indices:

            processed += 1
            sys.stdout.write(
                f"\rIntegrating frame {processed}/{total_samples}"
            )
            sys.stdout.flush()

            rgb_ts, depth_ts, pose_ts = frames[i]

            rgb_path = os.path.join(rgbd_video_dir, rgb_list[rgb_ts][0])
            depth_path = os.path.join(rgbd_video_dir, depth_list[depth_ts][0])

            rgb = cv.imread(rgb_path)
            depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED)

            if rgb is None or depth is None:
                continue

            color_o3d = o3d.geometry.Image(
                cv.cvtColor(rgb, cv.COLOR_BGR2RGB)
            )

            depth_o3d = o3d.geometry.Image(depth)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=5000.0,
                depth_trunc=8.0,
                convert_rgb_to_intensity=False
            )

            pose_data = list(map(float, pose_list[pose_ts]))

            tx, ty, tz = pose_data[0:3]
            qx, qy, qz, qw = pose_data[3:7]

            R = CameraProperties.quaternion_to_rotation_matrix(qx, qy, qz, qw)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [tx, ty, tz]

            # TUM poses are camera -> world, Open3D expects world -> camera
            extrinsic = np.linalg.inv(T)

            volume.integrate(
                rgbd,
                intrinsic_o3d,
                extrinsic
            )

        print("\nExtracting point cloud...")

        pcd = volume.extract_point_cloud()

        print(f"Points generated: {len(pcd.points)}")

        o3d.visualization.draw_geometries([pcd])