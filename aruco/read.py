import cv2
import yaml

def read_detector_params(file_name):
    detector_params = cv2.aruco.DetectorParameters_create()
    with open(file_name, 'r') as stream:
        try:
            params = yaml.load(stream)
            for key, value in params.items():
                setattr(detector_params, key, value)
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)
    return detector_params

def read_camera_file(file_name):
    cam_file = cv2.FileStorage(file_name, cv2.FILE_STORAGE_READ)
    cam_mat = cam_file.getNode('camera_matrix').mat()
    dist_coeffs = cam_file.getNode('distortion_coefficients').mat()
    img_size = (
        int(cam_file.getNode('image_height').real()),
        int(cam_file.getNode('image_width').real())
    )
    use_fisheye = cam_file.getNode('fisheye_model')
    print(use_fisheye)
    return cam_mat, dist_coeffs, img_size