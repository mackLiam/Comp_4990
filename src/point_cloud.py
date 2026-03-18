import os
import sys
import json
import socket
import cv2 as cv
import numpy as np
import open3d as o3d

try:
    import NDIlib as ndi
except ImportError:
    ndi = None
    print("Warning: ndi-python is not installed. Live streaming disabled.")

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


class MeshGenerator:

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
    def read_tum_file_list(filename):
        data = {}
        with open(filename) as f:
            for line in f:
                if line.startswith("#"): continue
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

    # --- TUM DATASET GENERATOR ---
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
    def generate_from_video(rgbd_video_dir, output_file="tum_mesh.ply" , on_progress=None):
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

        rgb_list = MeshGenerator.read_tum_file_list(
            os.path.join(rgbd_video_dir, "rgb.txt")
        )
        depth_list = MeshGenerator.read_tum_file_list(
            os.path.join(rgbd_video_dir, "depth.txt")
        )
        pose_list = MeshGenerator.read_tum_file_list(
            os.path.join(rgbd_video_dir, "groundtruth.txt")
        )

        rgb_depth_matches = MeshGenerator.associate(rgb_list, depth_list)

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

            if on_progress:
                on_progress(processed, total_samples)
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

        print("\nExtracting Mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh = MeshGenerator.optimize_mesh(mesh)
        o3d.io.write_triangle_mesh(output_file, mesh)

        print(f"Saved optimized mesh to {output_file}")
        o3d.visualization.draw_geometries([mesh])

        @staticmethod
        def generate_from_live_stream(udp_ip="0.0.0.0", udp_port=50000, output_file="live_mesh.ply"):
            if ndi is None:
                return print("Missing ndi-python.")

            # --- 1. SAFE UDP SETUP ---
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((udp_ip, udp_port))
            sock.setblocking(False)
            print(f"Listening for UDP Poses on port {udp_port}...")

            # --- 2. SAFE NDI SETUP ---
            ndi.initialize()
            ndi_recv_color = ndi.recv_create_v3()
            ndi_recv_depth = ndi.recv_create_v3()
            ndi_find = ndi.find_create_v2()

            print("Looking for Zig Sim Pro NDI streams (Make sure the app is running)...")
            sources = []
            while not sources:
                ndi.find_wait_for_sources(ndi_find, 1000)
                sources = ndi.find_get_current_sources(ndi_find)

            color_source = None
            depth_source = None

            for s in sources:
                print(f"Found NDI Source: {s.ndi_name}")
                if "Depth" in s.ndi_name or "DEPTH" in s.ndi_name:
                    depth_source = s
                elif "ZIG SIM" in s.ndi_name or "IMAGE" in s.ndi_name or "iPhone" in s.ndi_name:
                    color_source = s

            # Connect to whatever sources we found
            if color_source:
                ndi.recv_connect(ndi_recv_color, color_source)
                print(f"Connected Main Video to: {color_source.ndi_name}")
            else:
                print("Could not find main video stream. Exiting.")
                return

            if depth_source:
                ndi.recv_connect(ndi_recv_depth, depth_source)
                print(f"Connected Depth Video to: {depth_source.ndi_name}")
            else:
                print("Warning: No explicit Depth stream found. Attempting to extract depth from Alpha channel.")

            # --- 3. OPEN3D SETUP (Face ID Parameters) ---
            vis = o3d.visualization.Visualizer()
            vis.create_window("Live Scan Preview", width=800, height=600)
            pcd_vis = o3d.geometry.PointCloud()
            vis.add_geometry(pcd_vis)

            # High detail (2mm voxels), short range (1cm truncation) for Face ID
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=0.002, sdf_trunc=0.01,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
            )
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            )
            latest_transform = np.eye(4)

            print("\n--- STREAM ACTIVE: Close the 3D window to stop and save ---")
            try:
                while vis.poll_events():
                    vis.update_renderer()

                    # --- A. CATCH POSE (UDP) ---
                    try:
                        data, _ = sock.recvfrom(4096)
                        msg = json.loads(data.decode('utf-8'))

                        # Handle ARKit Face Pose Data
                        tracker = msg.get('arkit', msg.get('facedata', {}))
                        if 'position' in tracker and 'rotation' in tracker:
                            pos, rot = tracker['position'], tracker['rotation']
                            T = np.eye(4)
                            T[:3, :3] = CameraProperties.quaternion_to_rotation_matrix(
                                rot['x'], rot['y'], rot['z'], rot['w']
                            )
                            T[:3, 3] = [pos['x'], pos['y'], pos['z']]
                            latest_transform = np.linalg.inv(T)
                    except (BlockingIOError, json.JSONDecodeError):
                        pass  # Ignore empty buffers or squashed JSON packets

                    # --- B. CATCH NDI FRAMES (Non-Blocking) ---
                    t_color, v_color, _, _ = ndi.recv_capture_v3(ndi_recv_color, 0)

                    rgb_img = None
                    depth_img = None

                    if t_color == ndi.FRAME_TYPE_VIDEO:
                        frame = np.copy(v_color.data)
                        rgb_img = frame[:, :, :3][:, :, ::-1]  # BGR to RGB

                        # Fallback: If no dedicated depth stream, assume depth is encoded in the Alpha channel
                        if depth_source is None and frame.shape[2] == 4:
                            depth_img = frame[:, :, 3].astype(np.uint16) * 10

                        ndi.recv_free_video_v3(ndi_recv_color, v_color)

                    # Check explicit depth stream if it exists
                    if depth_source:
                        t_depth, v_depth, _, _ = ndi.recv_capture_v3(ndi_recv_depth, 0)
                        if t_depth == ndi.FRAME_TYPE_VIDEO:
                            depth_img = np.copy(v_depth.data)[:, :, 0].astype(np.uint16) * 10
                            ndi.recv_free_video_v3(ndi_recv_depth, v_depth)

                    # --- C. INTEGRATE INTO MESH ---
                    if rgb_img is not None and depth_img is not None:
                        # Filter out purely black depth maps to prevent Open3D crashes
                        if np.max(depth_img) > 0:
                            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                o3d.geometry.Image(rgb_img),
                                o3d.geometry.Image(depth_img),
                                depth_scale=1000.0, depth_trunc=1.0, convert_rgb_to_intensity=False
                            )
                            volume.integrate(rgbd, intrinsic_o3d, latest_transform)

                            extracted_pcd = volume.extract_point_cloud()
                            if not extracted_pcd.is_empty():
                                pcd_vis.points = extracted_pcd.points
                                pcd_vis.colors = extracted_pcd.colors
                                vis.update_geometry(pcd_vis)

            finally:
                print("\nClosing stream and generating final mesh...")
                vis.destroy_window()
                ndi.recv_destroy(ndi_recv_color)
                if depth_source:
                    ndi.recv_destroy(ndi_recv_depth)
                ndi.find_destroy(ndi_find)
                ndi.destroy()

                mesh = volume.extract_triangle_mesh()
                if len(mesh.vertices) > 0:
                    mesh = MeshGenerator.optimize_mesh(mesh)
                    o3d.io.write_triangle_mesh(output_file, mesh)
                    print(f"Saved optimized mesh to {output_file}")
                    o3d.visualization.draw_geometries([mesh], window_name="Final Optimized Mesh")
                else:
                    print("No data collected. The NDI stream was empty or the face tracking was lost.")