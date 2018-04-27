import argparse
import yaml
import cv2
import numpy as np
import pandas as pd
from aruco.read import read_detector_params
from aruco.read import read_camera_file


# Camera matrix, distcoeffs, dictionary, detectorparams marker length isFisheye imagelist
def estimate_markers(
        images,  img_size, cam_mat, dist_coeffs, dictionary,
        detect_params, marker_length=1.0, use_fisheye=False
):
    new_images = []
    marker_data = pd.DataFrame(
        columns=['id', 'tx', 'ty', 'tz', 'rx', 'ry', 'rz']
    )

    # Compute matrix after undistorting fisheye
    if use_fisheye:
        new_cam_mat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            cam_mat, dist_coeffs, img_size, np.eye(3), balance=1.0
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            cam_mat, dist_coeffs, np.eye(3),
            new_cam_mat, img_size, cv2.CV_16SC2
        )
        cam_mat = new_cam_mat
        dist_coeffs = np.zeros((4, 1))

    for img in images:
        assert img.shape[0:2] == img_size
        # Undistort img
        if use_fisheye:
            img = cv2.remap(
                img, map1, map2, cv2.INTER_CUBIC,
                cv2.BORDER_CONSTANT
            )
        else:
            pass

        corners, ids, _ = cv2.aruco.detectMarkers(
            img, dictionary,
            parameters=detect_params,
            cameraMatrix=cam_mat,
            distCoeff=dist_coeffs
        )

        if ids.size:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, marker_length, cam_mat, dist_coeffs
            )

            cv2.aruco.drawDetectedMarkers(img, corners, ids)

            for i, _ in enumerate(ids):
                cv2.aruco.drawAxis(
                    img, cam_mat, dist_coeffs,
                    rvecs[i], tvecs[i], marker_length*0.5
                )
                row = marker_data.shape[0]
                marker_data.loc[row] = {
                    'id': ids[i, 0],
                    'tx': tvecs[i, 0, 0],
                    'ty': tvecs[i, 0, 1],
                    'tz': tvecs[i, 0, 2],
                    'rx': rvecs[i, 0, 0],
                    'ry': rvecs[i, 0, 1],
                    'rz': rvecs[i, 0, 2],
                }

        new_images.append(img)

    return marker_data, new_images


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='TBD Description.')
    parser.add_argument('-c', '--camera-file', required=True,
                        dest='camera_file', type=str, nargs=1,
                        help='camera file')
    parser.add_argument('-d', '--dictionary', required=True,
                        dest='dictionary', type=int, nargs=1,
                        help='dictionary')
    parser.add_argument('-dp', '--detector-params', required=True,
                        dest='detect_params', type=str, nargs=1,
                        help='detector parameters')
    parser.add_argument('-l', '--length', required=False,
                        type=float, nargs=1, default=[1.0],
                        help='marker length')
    parser.add_argument('-fe', '--fisheye', required=False,
                        dest='use_fisheye',
                        action='store_true', default=False,
                        help='use fisheye')
    parser.add_argument('-o', '--output', required=False,
                        dest='output', type=str, nargs=1,
                        help='output file')
    parser.add_argument('images', metavar='image', type=str, nargs='+',
                        help='image files')
    args = parser.parse_args()

    # Read dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(args.dictionary[0])
    # Get fisheye mode
    use_fisheye = args.use_fisheye
    # Get marker length
    marker_length = args.length[0]
    # Read detector params file
    detect_params = read_detector_params(args.detect_params[0])

    # Read camera file
    cam_mat, dist_coeffs, img_size = read_camera_file(args.camera_file[0])

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame', 1024, 768)
    images = [cv2.imread(img) for img in args.images]
    marker_data, new_images = estimate_markers(
        images, img_size, cam_mat, dist_coeffs, dictionary, detect_params, marker_length=marker_length, use_fisheye=True
    )

    if args.output:
        marker_data.to_csv(args.output[0])
    else:
        print(marker_data)

    for img in new_images:
        cv2.imshow('frame', img)
        key = cv2.waitKey()
        if key == ord('q'):
            break


if __name__ == "__main__":
    main()
