
import cv2
import pandas as pd
from .camera_parameters import CameraParameters


class Detector:
    # TBD Add calibration parameters and other stuff
    def __init__(self, cam_params, detector_params, default_dictionary, default_marker_length):
        assert isinstance(cam_params, CameraParameters)
        assert isinstance(detector_params, cv2.aruco_DetectorParameters)
        assert isinstance(default_dictionary, cv2.aruco_Dictionary)
        assert isinstance(default_marker_length, float)

        self.cam_params = cam_params
        self.detector_params = detector_params
        self.dictionary = default_dictionary
        self.marker_length = default_marker_length

    # TBD Find corners and return  dataframe
    def undistort_and_estimate(self, img):
        assert False

    def estimate_markers(self, img, undistorted=False):
        if not undistorted:
            img = self.undistort_image(img)
        # Find aruco markers
        corners, ids, _ = cv2.aruco.detectMarkers(
            img, self.dictionary,
            parameters=self.detector_params,
            cameraMatrix=self.cam_params.cam_mat,
            distCoeff=self.cam_params.dist_coeffs
        )
        if ids.size:
            # Draw markers
            cv2.aruco.drawDetectedMarkers(img, corners, ids)
            # Estimate poses
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.cam_params.cam_mat, self.cam_params.dist_coeffs
            )

            columns = ['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
            data = dict(zip(
                columns,
                (ids, tvecs[0::, 3], tvecs[1::, 3], tvecs[2::, 3],
                 rvecs[0::, 3], rvecs[1::, 3], rvecs[1::, 3])
            ))
            dataframe = pd.DataFrame(data, columns=columns)

            for i, _ in enumerate(ids):
                cv2.aruco.drawAxis(
                    img, self.cam_params.cam_mat, self.cam_params.dist_coeffs,
                    rvecs[i], tvecs[i], self.marker_length*0.5
                )

    def undistort_image(self, img):
        assert img.shape == (*self.cam_params.size, 3)
        if self.cam_params.use_fisheye:
            return cv2.remap(
                img, self.cam_params.map1, self.cam_params.map2,
                interpolation=cv2.INTER_CUBIC,
                bordermode=cv2.BORDER_CONSTANT
            )
        else:
            assert False
