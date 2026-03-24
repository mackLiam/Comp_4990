import os
import sys
import cv2 as cv
import numpy as np
import open3d as o3d
import record3d as r3d
import threading

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
        mesh = PointCloudGenerator.optimize_mesh(mesh)

        o3d.io.write_triangle_mesh(output_file, mesh)
        print(f"Saved optimized mesh to {output_file}")
        o3d.visualization.draw_geometries([mesh])

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
    def generate_from_tum(rgbd_video_dir, output_file="tum_mesh.ply", on_progress=None):
        """Generate fused point cloud using offline TUM dataset."""
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
        fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]

        rgb_list = PointCloudGenerator.read_tum_file_list(os.path.join(rgbd_video_dir, "rgb.txt"))
        depth_list = PointCloudGenerator.read_tum_file_list(os.path.join(rgbd_video_dir, "depth.txt"))
        pose_list = PointCloudGenerator.read_tum_file_list(groundtruth_file)

        rgb_depth_matches = PointCloudGenerator.associate(rgb_list, depth_list)

        frames = []
        for rgb_ts, depth_ts in rgb_depth_matches:
            pose_ts = min(pose_list.keys(), key=lambda x: abs(x - rgb_ts))
            frames.append((rgb_ts, depth_ts, pose_ts))

        first = cv.imread(os.path.join(rgbd_video_dir, rgb_list[frames[0][0]][0]))
        height, width = first.shape[:2]

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
        volume = PointCloudGenerator.create_tsdf_volume(voxel_length=0.01, sdf_trunc=0.04)

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
            R = CameraProperties.quaternion_to_rotation_matrix(qx, qy, qz, qw)

            camera_to_world = np.eye(4)
            camera_to_world[:3, :3] = R
            camera_to_world[:3, 3] = [tx, ty, tz]

            PointCloudGenerator.integrate_frame(
                volume, rgb, depth, 8, intrinsic_o3d, camera_to_world, depth_scale=5000.0
            )

        PointCloudGenerator.extract_and_save(volume, output_file)

class LiveGenerator:
    @staticmethod
    def get_rotation_matrix(pose):
        """Converts quaternion from the Pose object into a 3x3 rotation matrix."""
        qx, qy, qz, qw = pose.qx, pose.qy, pose.qz, pose.qw
        return np.array([
            [1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]
        ])

    @staticmethod
    def is_movement_sufficient(curr_c2w, last_c2w, dist_thresh=0.01, rot_thresh=1.0):
        if last_c2w is None: return True
        dist = np.linalg.norm(curr_c2w[:3, 3] - last_c2w[:3, 3])
        # Check angular change
        R_curr, R_last = curr_c2w[:3, :3], last_c2w[:3, :3]
        rel_R = np.dot(R_curr, R_last.T)
        angle = np.degrees(np.arccos(np.clip((np.trace(rel_R) - 1) / 2, -1.0, 1.0)))
        return dist > dist_thresh or angle > rot_thresh

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
    def extract_and_save(volume, output_file):
        """Helper to pull the mesh out of the volume, optimize, and save."""
        print("\nExtracting Mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh = LiveGenerator.optimize_mesh(mesh)

        o3d.io.write_triangle_mesh(output_file, mesh)
        print(f"Saved optimized mesh to {output_file}")
        o3d.visualization.draw_geometries([mesh])

    @staticmethod
    def generate_from_live_video(output_file="live_mesh.ply"):
        devs = r3d.Record3DStream.get_connected_devices()
        if not devs:
            print("No iPhone detected.")
            return

        session = r3d.Record3DStream()
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.002, sdf_trunc=0.01,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        ctx = {"last_c2w": None, "intrinsic": None, "count": 0, "stop": threading.Event()}

        def on_new_frame():
            try:
                rgb = session.get_rgb_frame()
                depth = session.get_depth_frame()
                pose_obj = session.get_camera_pose()
                intr = session.get_intrinsic_mat()

                if rgb is None or depth is None or rgb.size == 0: return

                # Build the Pose Matrix
                c2w = np.eye(4)
                c2w[:3, :3] = LiveGenerator.get_rotation_matrix(pose_obj)
                c2w[:3, 3] = [pose_obj.tx, pose_obj.ty, pose_obj.tz]

                # ARKit to Open3D space
                flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
                c2w_final = c2w @ flip

                if LiveGenerator.is_movement_sufficient(c2w_final, ctx["last_c2w"]):
                    d_h, d_w = depth.shape
                    r_h, r_w = rgb.shape[:2]

                    # Sync resolutions
                    rgb_fixed = cv.resize(rgb, (d_w, d_h), interpolation=cv.INTER_AREA)
                    depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

                    # 3. Scale Intrinsics to match resized RGB
                    if ctx["intrinsic"] is None:
                        sx, sy = d_w / r_w, d_h / r_h
                        ctx["intrinsic"] = o3d.camera.PinholeCameraIntrinsic(
                            d_w, d_h, intr.fx * sx, intr.fy * sy, intr.tx * sx, intr.ty * sy
                        )

                    # Integrate
                    rgb_o3d = o3d.geometry.Image(rgb_fixed)
                    depth_o3d = o3d.geometry.Image(depth_clean)
                    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                        rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1.0, convert_rgb_to_intensity=False
                    )

                    volume.integrate(rgbd, ctx["intrinsic"], np.linalg.inv(c2w_final))

                    ctx["last_c2w"] = c2w_final
                    ctx["count"] += 1
                    print(f"Frames: {ctx['count']}", end="\r")

            except Exception as e:
                print(f"\nError: {e}")

        session.on_new_frame = on_new_frame
        session.on_stream_stopped = lambda: ctx["stop"].set()
        session.connect(devs[0])

        try:
            print("Streaming... Stop recording on the Record3D app when finished.")
            while not ctx["stop"].is_set():
                ctx["stop"].wait(0.1)
        finally:
            session.disconnect()

        if ctx["count"] > 0:
            LiveGenerator.extract_and_save(volume, output_file)