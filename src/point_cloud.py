import os
import cv2 as cv
import numpy as np
import open3d as o3d
import sys

class CameraProperties:
    """Handles camera intrinsic/extrinsic parameters loading."""

    @staticmethod
    def load_intrinsics(intrinsics_path):
        """Load camera intrinsic parameters from a text file.

        Args:
            intrinsics_path: Path to intrinsics file

        Returns:
            dict: Dictionary containing the intrinsic parameters (fx, fy, cx, cy)
        """
        try:
            with open(intrinsics_path, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            if len(lines) < 3:
                print(f"Warning: Invalid intrinsics format, using defaults")
                return {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}

            # Parse 3x3 matrix: [fx, 0, cx], [0, fy, cy], [0, 0, 1]
            fx, _, cx = map(float, lines[0].split())
            _, fy, cy = map(float, lines[1].split())

            return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}

        except Exception as e:
            print(f"Error loading intrinsics from {intrinsics_path}: {e}")
            return {'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5}

    @staticmethod
    def load_extrinsics(extrinsics_file):
        """Load and parse extrinsics from a single file.

        Args:
            extrinsics_file: Path to extrinsics file

        Returns:
            list: List of 4x4 camera poses
        """
        if not os.path.exists(extrinsics_file):
            print(f"Error: Extrinsics file not found at {extrinsics_file}")
            return []

        try:
            with open(extrinsics_file, 'r') as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            rows = [list(map(float, ln.split())) for ln in lines]
            poses = []

            for i in range(0, len(rows), 3):
                block = np.array(rows[i:i + 3])  # 3x4
                T = np.eye(4)
                T[:3, :4] = block
                poses.append(T)

            print(f"Loaded {len(poses)} poses from {extrinsics_file}")
            return poses

        except Exception as e:
            print(f"Error parsing extrinsics file {extrinsics_file}: {e}")
            return []

class PointCloudGenerator:
    """Core point cloud generation functionality."""

    @staticmethod
    def create_frame_cloud(rgb, depth, fx, fy, cx, cy, depth_scale=1000.0, flip_orientation=True):
        """Create point cloud from one RGB-D frame.

        Args:
            rgb: RGB image array
            depth: Depth image array
            fx, fy, cx, cy: Camera intrinsics
            depth_scale: Scale factor for depth values (default: 1000.0)
            flip_orientation: Whether to flip point cloud orientation (default: True)

        Returns:
            Open3D PointCloud object
        """
        depth_m = depth.astype('float32') / depth_scale

        height, width = depth_m.shape[:2]

        u_coords, v_coords = np.meshgrid(
            np.arange(width),
            np.arange(height)
        )

        u = u_coords.flatten()
        v = v_coords.flatten()
        z = depth_m.flatten()

        # Filter invalid depth values
        valid = z > 0
        u = u[valid]
        v = v[valid]
        z = z[valid]

        # Project pixels to 3D points
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        # Create point cloud
        points = np.vstack((x, y, z)).T
        rgb_flat = rgb.reshape(-1, 3)
        colors = rgb_flat[valid] / 255.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Flip orientation if requested
        if flip_orientation:
            pcd.transform([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])

        return pcd

    @staticmethod
    def generate_from_video(rgbd_video_dir):
        """Generate point cloud from video sequence using TSDF fusion.

        Args:
            rgbd_video_dir: Path to the RGBD video directory containing image, depth, extrinsics, and intrinsics
        """
        rgb_dir = os.path.join(rgbd_video_dir, "image")
        depth_dir = os.path.join(rgbd_video_dir, "depth")
        intrinsics_path = os.path.join(rgbd_video_dir, "intrinsics.txt")
        extrinsics_file = os.path.join(rgbd_video_dir, "extrinsics", "extrinsics.txt")

        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
            print(f"Error: Missing 'image' or 'depth' directory under {rgbd_video_dir}")
            return

        # Load poses
        print("Loading extrinsics...")
        poses = CameraProperties.load_extrinsics(extrinsics_file)

        if not poses:
            print("Error: No camera poses loaded; cannot fuse frames")
            return

        # Load intrinsics
        intr = CameraProperties.load_intrinsics(intrinsics_path)
        fx = intr.get('fx')
        fy = intr.get('fy')
        cx = intr.get('cx')
        cy = intr.get('cy')

        # Frame lists
        rgb_files = sorted(os.listdir(rgb_dir))
        depth_files = sorted(os.listdir(depth_dir))

        if not rgb_files or not depth_files:
            print("Error: No RGB or depth frames found")
            return

        total = min(len(rgb_files), len(depth_files), len(poses))
        print(f"\nFrames available: {total}")

        # Determine resolution dynamically from the first readable RGB frame
        width  = None
        height = None
        img = cv.imread(os.path.join(rgb_dir, rgb_files[0]))
        if img is not None:
            height, width = img.shape[:2]

        if width is None or height is None:
            print("Error: Could not read any RGB frames to determine resolution")
            return

        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

        # Sampling
        sample_rate = 25
        print(f"Sampling every {sample_rate} frames")

        # TSDF Volume settings
        voxel_length = 0.02
        sdf_trunc = 0.10

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_length,
            sdf_trunc=sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # Integration loop
        frame_indices = range(0, total, sample_rate)
        total_samples = len(frame_indices)
        processed = 0
        for i in frame_indices:
            processed += 1
            sys.stdout.write(f"\rIntegrating frame {processed}/{total_samples}")
            sys.stdout.flush()

            rgb_path = os.path.join(rgb_dir, rgb_files[i])
            depth_path = os.path.join(depth_dir, depth_files[i])
            rgb = cv.imread(rgb_path)
            depth = cv.imread(depth_path, cv.IMREAD_UNCHANGED)

            if rgb is None or depth is None:
                continue

            # Convert to Open3D images
            color_o3d = o3d.geometry.Image(cv.cvtColor(rgb, cv.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d,
                depth_o3d,
                depth_scale=10000.0,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False
            )

            volume.integrate(
                rgbd,
                intrinsic_o3d,
                np.linalg.inv(poses[i])
            )

        print("\n\nFusion complete.")

        # Extract fused point cloud
        print("Extracting fused point cloud...")
        fused_cloud = volume.extract_point_cloud()

        print(f"Fused points: {len(fused_cloud.points)}")

        # Extract mesh
        print("Extracting mesh...")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()

        # Visualization
        print("Visualizing results...")

        o3d.visualization.draw_geometries(
            [fused_cloud],
            window_name="Fused Point Cloud"
        )

        o3d.visualization.draw_geometries(
            [mesh],
            window_name="TSDF Mesh"
        )
