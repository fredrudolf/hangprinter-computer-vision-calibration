
import argparse
import cv2
import numpy as np
import pandas as pd


def normal_from_rotationvector(rot_vec):
    """ Given a rotation vector, return its normal"""
    # Get rotation matrix
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    # Return normal after rotation
    return np.matmul(rot_mat, [0, 0, 1])


def normals_from_rotationvectors(df):
    """ Given a dataframe of rotation vectors, return its normals"""
    normals = df.apply(normal_from_rotationvector, raw=True, axis=1)
    normals.columns = ['nx', 'ny', 'nz']
    return normals


def avg_vector(vectors):
    """ Given a dataframe of vectors, return its average"""
    # Return average normal
    avg = np.sum(vectors, axis=0)
    avg = avg/np.linalg.norm(vectors)
    # Variance on x, y and z
    var = np.var(vectors, ddof=1).values

    return avg, var


def avg_normal_from_rotationvectors(df):
    """ Given a dataframe of rotation vectors, return its average normal"""
    # Compute normals for each rotation vector
    normals = normals_from_rotationvectors(df)
    return avg_vector(normals)


def rotationvector_to_normal(rotation, new_normal):
    # print('')
    #print('rotation', rotation)
    normal = normal_from_rotationvector(rotation)
    #print('normal', normal)
    #print('new_normal', new_normal)
    axis = np.cross(normal, new_normal)
    k = np.dot(normal, new_normal)
    axis = axis * np.arccos(k) / np.linalg.norm(axis)
    #print('angle', np.arccos(k) * 180 / np.pi)
    fst, _ = cv2.Rodrigues(axis)
    snd, _ = cv2.Rodrigues(rotation)
    rtn = np.matmul(fst, snd)
    rtn, _ = cv2.Rodrigues(rtn)
    p = np.matmul(cv2.Rodrigues(rtn)[0], [1, 0, 0])
    print(np.arccos(np.dot(p, [1, 0, 0])) * 180 / np.pi)
    #print(np.matmul(fst, normal))
    """
    print('axis', axis)

    print(np.matmul([1,0,0], rtn))
    print(np.matmul([0,1,0], rtn))
    print(np.matmul([0,0,1], rtn))
    rtn, _ = cv2.Rodrigues(rtn)
    print('new', normal_from_rotationvector(rtn))
    print('rtn', np.linalg.norm(rtn) * 180 / np.pi)
    """
    return rotation


def rotate_rotation_to_new_normal(df, normal):
    """ Given a dataframe of rotation vectors, rotate their normals to a given directions"""
    #df_norms = normals_from_rotationvectors(df)
    rot_vecs = df.apply(rotation_around_z, axis=1, raw=True)
    # print(df)
    # print(rot_vecs)
    #print(rot_vecs.apply(np.linalg.norm, axis=1, raw=True))
    #print(np.arcsin(np.linalg.norm(rot_vec), ))
    #print((df.apply(lambda x: np.dot(x, normal), axis=1, raw=True)))
    #rot_vec = df.apply(lambda x, y=x: x, axis=1, raw=True)
    #print((rot_vec.apply(np.linalg.norm, axis=1, raw=True)))
    # print(np.linalg.norm(df.iloc[[0]]))
    # print(rods.values)


def rotation_around_z(rotation):
    """
    Rotate rotation vector in closest direction to normal, then compute rotation around it.
    Return a tuple of radians. First is rotation around new normal. Second is rotation to get
    to given normal.
    """
    normal = (0, 0, 1)
    new_normal = (0, 0, -1)
    fst, _ = cv2.Rodrigues(rotation)
    current_normal = np.matmul(fst, normal)
    angle_to_normal = np.arccos(np.dot(current_normal, new_normal))
    rot_axis = np.cross(current_normal, new_normal)
    rot_axis = rot_axis * angle_to_normal / np.linalg.norm(rot_axis)
    snd, _ = cv2.Rodrigues(rot_axis)

    old_x = (1, 0, 0)
    new_x = np.matmul(snd, np.matmul(fst, old_x))
    angle_around_z = np.arccos(np.dot(old_x, new_x))

    return angle_around_z, angle_to_normal


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
    #rtn = avg_normal_from_rotationvectors(data[cols])
    #print('Vector (x,y,z): ({:.4} {:.4} {:.4})'.format(*rtn[0]))
    #print('Variance (x,y,z): ({:.4} {:.4} {:.4})'.format(*rtn[1]))
    data[['rotation_z', 'rotation_to_normal']] = data[cols].apply(rotation_around_z, raw=True, axis=1).apply(pd.Series)
    print(data)
    data.to_csv('lol.csv')

if __name__ == "__main__":
    main()
