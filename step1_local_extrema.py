import os
import sys
import glob
import argparse

import cv2
import numpy as np
import scipy.ndimage.filters as nd_filters


def cal_max_min(p, neighbors, k):
    M_cnt, m_cnt = 0, 0
    for n in neighbors:
        if n > p:   M_cnt += 1
        if n < p:   m_cnt += 1

    if M_cnt <= k - 1:
        isMax = True
    else:
        isMax = False

    if m_cnt <= k - 1:
        isMin = True
    else:
        isMin = False

    return isMax, isMin


def _find_local_extrema(args, img, k=3):
    height, width = img.shape
    offset = int((k - 1) / 2)

    # Pixel p is reported as a maxima (resp. minima)
    # if at most k − 1 elements in in the k × k neighborhood around p
    # are greater (resp. smaller) than the value at pixel p.
    maxima_list, minima_list = [], []
    matMax, matMin = np.zeros((height, width)), np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            p = img[h][w]
            neighbors = img[h-offset:h+1+offset, w-offset:w+1+offset].flatten()
            #print (h, w, p, neighbors)
            isMax, isMin = cal_max_min(p, neighbors, k)

            if isMax:
                maxima_list.append((h, w))
                matMax[h][w] = 1

            if isMin:
                minima_list.append((h, w))
                matMin[h][w] = 1

    if args.save:
        imgMin = np.array([ [255 if p else 0 for p in m] for m in matMin])
        imgMax = np.array([ [255 if p else 0 for p in m] for m in matMax])

        cv2.imwrite(os.path.join(args.output_dir, "gray.jpg"), img)
        cv2.imwrite(os.path.join(args.output_dir, "max.jpg"), imgMax)
        cv2.imwrite(os.path.join(args.output_dir, "min.jpg"), imgMin)

    return matMax, matMin


# code was borrowed from https://www.reddit.com/r/learnpython/comments/3nluao/python_implementation_of_ordfilt2_matlab_api_any/
def local_filter(x, order):
    x.sort()
    return x[order]


def ordfilt2(A, order, mask_size):
    return nd_filters.generic_filter(A, lambda x, ord=order: local_filter(x, ord), size=(mask_size, mask_size))


def find_local_extrema(args, img, k=3):
    height, width = img.shape

    matMin = np.array(ordfilt2(img, k, k) >= img, dtype=np.int8)
    matMax = np.array(ordfilt2(img, k*k-k+1, k) <= img, dtype=np.int8)

    if args.save:
        imgMin = np.array([ [255 if p else 0 for p in m] for m in matMin])
        imgMax = np.array([ [255 if p else 0 for p in m] for m in matMax])

        cv2.imwrite(os.path.join(args.output_dir, "gray.jpg"), img)
        cv2.imwrite(os.path.join(args.output_dir, "b_max.jpg"), imgMax)
        cv2.imwrite(os.path.join(args.output_dir, "b_min.jpg"), imgMin)

        def mat2img(mat_extrema):
            b = np.zeros(height * width)
            idx_selected = np.nonzero(mat_extrema.flatten())
            img_flat = img.flatten()
            b[idx_selected] = img_flat[idx_selected]

            return b.reshape(height, width)

        imgMin = mat2img(matMin)
        imgMax = mat2img(matMax)

        cv2.imwrite(os.path.join(args.output_dir, "gray.jpg"), img)
        cv2.imwrite(os.path.join(args.output_dir, "o_max.jpg"), imgMax)
        cv2.imwrite(os.path.join(args.output_dir, "o_min.jpg"), imgMin)

    return matMin, matMax
