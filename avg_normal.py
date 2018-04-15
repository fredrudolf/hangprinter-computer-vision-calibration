import numpy as np
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
    return np.mean(normals.values, axis=0)
