import os
import sys
import glob
import argparse

import cv2
import numpy as np
import scipy
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg


wd_width = 1

# code was borrowed from http://blog.sws9f.org/computer-vision/2017/09/07/colorization-using-optimization-python.html
def compute_envelope(args, matMin, matMax, img):

    save_list = ['min', 'max']
    for i, mat_extrema in enumerate([matMin, matMax]):
        # Prepare the Matrix: A
        pic_rows, pic_cols = img.shape
        pic_size = pic_rows * pic_cols

        weightData = []
        num_pixel_bw = 0

        # build the weight matrix for each window.
        for c in range(pic_cols):
            for r in range(pic_rows):
                res = []
                w = WindowNeighbor(wd_width, (r,c), img)
                if not mat_extrema[r,c]:
                    weights = affinity_a(w)
                    for e in weights:
                        weightData.append([w.center,(e[0],e[1]), e[2]])
                weightData.append([w.center, (w.center[0],w.center[1]), 1.])

        sp_idx_rc_data = [[to_seq(e[0][0], e[0][1], pic_rows), to_seq(e[1][0], e[1][1], pic_rows), e[2]] for e in weightData]
        sp_idx_rc = np.array(sp_idx_rc_data, dtype=np.integer)[:,0:2]
        sp_data = np.array(sp_idx_rc_data, dtype=np.float64)[:,2]

        matA = sparse.csr_matrix((sp_data, (sp_idx_rc[:,0], sp_idx_rc[:,1])), shape=(pic_size, pic_size))


        # Prepare the Vector: B
        b = np.zeros(pic_size)
        idx_selected = np.nonzero(mat_extrema.flatten())
        pic_flat = img.flatten()
        b[idx_selected] = pic_flat[idx_selected]

        # Solve the optimazation problem
        ans = sparse.linalg.spsolve(matA, b)
        cv2.imwrite(os.path.join(args.output_dir, "envelope_{}.jpg".format(save_list[i])), ans.reshape((pic_rows, pic_cols)))


# the window class, find the neighbor pixels around the center.
class WindowNeighbor:
    def __init__(self, offset, center, pic):
        # center is a list of [row, col, intensity]
        self.center = [center[0], center[1], pic[center]]
        self.offset = offset
        self.neighbors = None
        self.find_neighbors(pic)
        self.mean = None
        self.var = None

    def find_neighbors(self, pic):
        self.neighbors = []
        ix_r_min = max(0, self.center[0] - self.offset)
        ix_r_max = min(pic.shape[0], self.center[0] + self.offset + 1)
        ix_c_min = max(0, self.center[1] - self.offset)
        ix_c_max = min(pic.shape[1], self.center[1] + self.offset + 1)
        for r in range(ix_r_min, ix_r_max):
            for c in range(ix_c_min, ix_c_max):
                if r == self.center[0] and c == self.center[1]:
                    continue
                self.neighbors.append([r, c, pic[r,c]])

    def __str__(self):
        return 'windows c=(%d, %d, %f) size: %d' % (self.center[0], self.center[1], self.center[2], len(self.neighbors))

# affinity functions, calculate weights of pixels in a window by their intensity.
def affinity_a(w):
    nbs = np.array(w.neighbors)
    sY = nbs[:,2]
    cY = w.center[2]
    diff = sY - cY
    sig = np.var(np.append(sY, cY))
    if sig < 1e-6:
        sig = 1e-6
    wrs = np.exp(- np.power(diff,2) / (sig * 2.0))
    wrs = - wrs / np.sum(wrs)
    nbs[:,2] = wrs
    return nbs

# translate (row,col) to/from sequential number
def to_seq(r, c, rows):
    return c * rows + r

def fr_seq(seq, rows):
    r = seq % rows
    c = int((seq - r) / rows)
    return (r, c)
