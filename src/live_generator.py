import numpy as np
import open3d as o3d
import cv2 as cv
import record3d as r3d
from src.helper import Helper

class LiveGenerator:
    """Generator for creating 3D meshes from live Record3D iPhone streams."""

    def __init__(self):
        self.session = None
        self.volume = None
        self.intrinsic = None
        self.last_c2w = None
        self.count = 0
        self.is_scanning = False
        self.current_pos = [0.0, 0.0, 0.0]
        self.on_frame_callback = None

    def start_scan(self):
        """Initialize and start the live scan session."""
        devs = r3d.Record3DStream.get_connected_devices()
        if not devs:
            return False, "No iPhone detected! Check USB connection."

        # Reset State
        self.count = 0
        self.last_c2w = None
        self.intrinsic = None
        self.volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.004, sdf_trunc=0.04,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )

        # Connect to iPhone
        self.session = r3d.Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(devs[0])

        self.is_scanning = True
        return True, "Connected successfully"

    def stop_scan(self):
        """Stop the scanning session."""
        if self.session:
            self.session.disconnect()

    def on_stream_stopped(self):
        """Called when the stream stops."""
        self.is_scanning = False

    def on_new_frame(self):
        """Process each new frame from the Record3D stream."""
        try:
            rgb = self.session.get_rgb_frame()
            depth = self.session.get_depth_frame()
            pose_obj = self.session.get_camera_pose()
            intr = self.session.get_intrinsic_mat()

            # Skip if no valid data yet (waiting for recording to start)
            if rgb is None or depth is None or pose_obj is None:
                return

            if rgb.size == 0 or depth.size == 0:
                return

            self.current_pos = [pose_obj.tx, pose_obj.ty, pose_obj.tz]

            # Build camera-to-world matrix using helper
            c2w = np.eye(4)
            qx, qy, qz, qw = pose_obj.qx, pose_obj.qy, pose_obj.qz, pose_obj.qw
            c2w[:3, :3] = Helper.quaternion_to_rotation_matrix(qx, qy, qz, qw)
            c2w[:3, 3] = self.current_pos

            # ARKit to Open3D coordinate system
            flip = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            c2w_final = c2w @ flip

            # Check if movement is sufficient
            if self.last_c2w is not None:
                dist = np.linalg.norm(c2w_final[:3, 3] - self.last_c2w[:3, 3])
                rel_R = np.dot(c2w_final[:3, :3], self.last_c2w[:3, :3].T)
                angle = np.degrees(np.arccos(np.clip((np.trace(rel_R) - 1) / 2, -1.0, 1.0)))
                if dist <= 0.01 and angle <= 1.0:
                    return

            d_h, d_w = depth.shape
            r_h, r_w = rgb.shape[:2]
            rgb_fixed = cv.resize(rgb, (d_w, d_h), interpolation=cv.INTER_AREA)
            depth_clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            # Initialize intrinsics on first valid frame
            if self.intrinsic is None:
                sx, sy = d_w / r_w, d_h / r_h
                self.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    d_w, d_h, intr.fx * sx, intr.fy * sy, intr.tx * sx, intr.ty * sy
                )

            # Create RGBD image and integrate
            rgb_o3d = o3d.geometry.Image(rgb_fixed)
            depth_o3d = o3d.geometry.Image(depth_clean)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=1.5, convert_rgb_to_intensity=False
            )

            self.volume.integrate(rgbd, self.intrinsic, np.linalg.inv(c2w_final))
            self.last_c2w = c2w_final
            self.count += 1

            # Notify callback if set
            if self.on_frame_callback:
                self.on_frame_callback()

        except Exception as e:
            print(f"Error in on_new_frame: {e}")

    def extract_mesh(self, output_file="live_mesh.ply", visualize=False):
        """Extract and save the mesh from the TSDF volume."""
        if self.count == 0:
            return False, "No frames captured"

        try:
            print("\nExtracting Mesh...")
            mesh = self.volume.extract_triangle_mesh()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_unreferenced_vertices()

            o3d.io.write_triangle_mesh(output_file, mesh)
            print(f"Saved optimized mesh to {output_file}")

            if visualize:
                o3d.visualization.draw_geometries([mesh])

            return True, output_file
        except Exception as e:
            return False, f"Error extracting mesh: {e}"
