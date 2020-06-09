import os
import sys
import glob
import argparse

import cv2
import numpy as np
import scipy

from scipy import io, misc, sparse
from scipy.linalg import solve_banded
from scipy.ndimage.filters import uniform_filter
from scipy.sparse.linalg import spsolve
from sklearn.preprocessing import normalize


# code was borrowed from https://github.com/asafdav2/colorization_using_optimization/blob/master/colorization.py
def cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def neighborhood(X, r):
    d = 1
    l1, h1 = max(r[0]-d, 0), min(r[0]+d+1, X.shape[0])
    l2, h2 = max(r[1]-d, 0), min(r[1]+d+1, X.shape[1])
    return X[l1:h1, l2:h2]


def weight(r, V, Y, S):
    return [np.exp(-1 * np.square(Y[r] - Y[v]) / S[r]) if S[r] > 0.0 else 0.0 for v in V]


def find_marked_locations(mat):

    return list(zip(*np.nonzero(mat)))


def std_matrix(A):
    S = np.empty_like(A)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            S[i, j] = np.square(np.std(neighborhood(A, [i, j])))
    return S


def build_weights_matrix(Y):
    (n, m) = [Y.shape[0], Y.shape[1]]
    S = std_matrix(Y)
    size = n * m
    cart = cartesian([list(range(n)), list(range(m))])
    cart_r = cart.reshape(n, m, 2)
    xy2idx = np.arange(size).reshape(n, m)  # [x,y] -> index
    W = sparse.lil_matrix((size, size))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            idx = xy2idx[i, j]
            N = neighborhood(cart_r, [i, j]).reshape(-1, 2)
            N = [tuple(neighbor) for neighbor in N]
            N.remove((i, j))
            p_idx = [xy2idx[xy] for xy in N]
            weights = weight((i, j), N, Y, S)
            W[idx, p_idx] = -1 * np.asmatrix(weights)

    Wn = normalize(W, norm='l1', axis=1)
    Wn[np.arange(size), np.arange(size)] = 1

    return Wn


def compute_envelope(args, matMin, matMax, img):

    Y = np.array(img, dtype='float64')

    n, m = img.shape  # extract image dimensions
    size = n * m
    Wn = 0

    save_list = ['min', 'max']
    sol_list = []
    for i, mat_extrema in enumerate([matMin, matMax]):

        Wn = 0
        pic = './img/{}_{}.mtx'.format(args.filename, args.k)
        if (os.path.isfile(pic)):
            Wn = scipy.io.mmread(pic).tocsr()
        else:
            Wn = build_weights_matrix(Y)
            scipy.io.mmwrite(pic, Wn)

        ## once markes are found
        id_extrema = find_marked_locations(mat_extrema)

        ## set rows in colored indices
        Wn = Wn.tolil()
        xy2idx = np.arange(size).reshape(n, m)  # [x,y] -> index
        for idx in [xy2idx[i, j] for [i, j] in id_extrema]:
            Wn[idx] = sparse.csr_matrix(([1.0], ([0], [idx])), shape=(1, size))

        LU = scipy.sparse.linalg.splu(Wn.tocsc())


        b = np.zeros(size)
        idx_selected = np.nonzero(mat_extrema.flatten())
        img_flat = img.flatten()
        b[idx_selected] = img_flat[idx_selected]

        x = LU.solve(b)

        sol = x.reshape((n, m))
        sol_list.append(sol)

        if args.save:
            misc.imsave(os.path.join(args.output_dir, "envelope_{}.bmp".format(save_list[i])), sol)

    return sol_list[0], sol_list[1]
