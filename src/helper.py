import numpy as np

class Helper:
    """Helper utilities for common operations."""

    @staticmethod
    def quaternion_to_rotation_matrix(qx, qy, qz, qw):
        """
        Convert quaternion to rotation matrix.

        Args:
            qx, qy, qz, qw: Quaternion components

        Returns:
            3x3 rotation matrix as numpy array
        """
        R = np.array([
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy
            ]
        ])

        return R
