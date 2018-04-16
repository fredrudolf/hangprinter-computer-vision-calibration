
import numpy as np
import pandas as pd
import argparse
import cv2


def normal_from_rotationvector(rot_vec):
    """ Given a rotation vector, return its normal"""
    # Get rotation matrix
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    # Define normal before rotation
    p = np.array([0, 0, 1])
    # Return normal after rotation
    return np.matmul(p, rot_mat)


def normals_from_rotationvectors(df):
    """ Given a dataframe of rotation vectors, return its normals"""
    normals = df.apply(normal_from_rotationvector, raw=True, axis=1)
    normals.columns = ['nx', 'ny', 'nz']
    return normals


def avg_normal_from_rotationvectors(df):
    """ Given a dataframe of rotation vectors, return its average normal"""
    # Compute normals for each rotation vector
    normals = normals_from_rotationvectors(df)
    # Return average normal
    avg_normal = np.sum(normals.values, axis=0)
    avg_normal = avg_normal/np.linalg.norm(avg_normal)
    # Variance on x, y and z
    var_xyz = np.var(normals, ddof=1).values

    return avg_normal, var_xyz


def main():
    parser = argparse.ArgumentParser(
        description='Find the average normal for pose data.')
    parser.add_argument('file', type=str, nargs=1,
                        help='a csv-file with pose data')
    parser.add_argument('--cols', metavar='col', type=str, nargs=3, default=['rx', 'ry', 'rz'],
                        help='column names for rotation vector (default: rx ry rz)')
    args = parser.parse_args()

    data = pd.read_csv(args.file[0])
    cols = args.cols
    rtn = avg_normal_from_rotationvectors(data[cols])
    print('Vector (x,y,z): ({:.4} {:.4} {:.4})'.format(*rtn[0]))
    print('Variance (x,y,z): ({:.4} {:.4} {:.4})'.format(*rtn[1]))


if __name__ == "__main__":
    main()
