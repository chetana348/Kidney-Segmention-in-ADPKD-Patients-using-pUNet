#PCA for dimension reduction (np arrays)
import numpy as np
from numpy import linalg as LA


def compute_Z(X, centering=True, scaling=True):
    # copying X because need a float type
    Z = X.astype(float)

    # shapes
    rows = Z.shape[0]
    cols = Z.shape[1]

    if centering:
        # means calculation for centering
        means = X.mean(0)
        for i in range(rows):
            for j in range(cols):
                Z[i, j] = Z[i, j] - means[j]

    # scaling with std dev
    if scaling:
        std = np.std(Z, axis=0)
        for i in range(cols):
            for j in range(rows):
                Z[j][i] = Z[j][i] / std[i]  # standardize every feature

    return Z


def compute_covariance_matrix(Z):
    return Z.T.dot(Z)


def find_pcs(COV):
    eigenValues, eigenVectors = LA.eig(COV)
    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    return eigenValues, eigenVectors


def project_data(Z, PCS, L, k, var):
    eigen_pairs = [(np.abs(L[i]), PCS[:, i]) for i in range(len(L))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    # copy of components could be useful
    component_matrix = np.copy(PCS)
    if k != 0:
        # delete unnecessary components
        component_matrix = np.delete(component_matrix, range(k, component_matrix.shape[1]), axis=1)
        Z_star = Z.dot(component_matrix)

    else:
        print()
        tot_var = 0
        eigen_val_index = 0
        while tot_var < var:
            tot_var = tot_var + L[eigen_val_index] / np.sum(L)
            eigen_val_index = eigen_val_index + 1
        component_count = eigen_val_index
        component_matrix = np.delete(component_matrix, range(component_count, component_matrix.shape[1]), axis=1)
        Z_star = Z.dot(component_matrix)
    return Z_star

#Test PCA
import numpy as np
import pca as p
import compress as c

TRAINING_DATA = "Data/Train/"
TEST_DATA = "Data/Test/"

X = c.load_data(TRAINING_DATA)
# c.compress_images(X, 10)
c.compress_images(X, 100)
# c.compress_images(X, 500)
# c.compress_images(X, 1000)
# c.compress_images(X, 2000)
# c.compress_images(X, 50)

# X = c.load_data(TEST_DATA)
# c.compress_images(X, 10)
# c.compress_images(X, 100)
# c.compress_images(X, 500)
# c.compress_images(X, 1000)
# c.compress_images(X, 2000)

# X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
# X = np.array([[1, 1], [2, 7], [3, 3], [4, 4], [5, 5]])
# Z = p.compute_Z(X)
# COV = p.compute_covariance_matrix(Z)
# L, PCS = p.find_pcs(COV)
# Z_star = p.project_data(Z, PCS, L, 2, 0)
# print(Z_star)

exit()