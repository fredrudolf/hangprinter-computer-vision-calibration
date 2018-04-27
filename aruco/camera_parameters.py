import cv2
import numpy as np


class CameraParameters:
    def __init__(self, cam_mat, dist_coeffs, width, height, use_fisheye):
        assert isinstance(cam_mat, np.ndarray) and cam_mat.shape == (3, 3)
        assert isinstance(cam_mat, np.ndarray) and dist_coeffs.shape == (4, 1)
        assert isinstance(width, int) and isinstance(height, int)
        assert isinstance(use_fisheye, bool)

        self.cam_mat = cam_mat.copy()
        self.dist_coeffs = dist_coeffs.copy()
        self.size = height, width
        self.use_fisheye = use_fisheye

        if use_fisheye:
            new_cam_mat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                cam_mat, dist_coeffs, self.size, np.eye(3), balance=1.0
            )
            self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
                cam_mat, dist_coeffs, np.eye(3),
                new_cam_mat, self.size, cv2.CV_16SC2
            )
            self.new_cam_mat = new_cam_mat
            self.new_dist_coeffs = np.zeros((4, 1))


def read_camera_parameters(filename):
    # Open file
    cam_file = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
    # Read parameters
    cam_mat = cam_file.getNode('camera_matrix').mat()
    dist_coeffs = cam_file.getNode('distortion_coefficients').mat()
    height = int(cam_file.getNode('image_height').real())
    width = int(cam_file.getNode('image_width').real())
    use_fisheye = bool(cam_file.getNode('fisheye_model'))
    # Release file
    cam_file.release()
    # Create and return CameraParameters object
    return CameraParameters(cam_mat, dist_coeffs, width, height, use_fisheye)

