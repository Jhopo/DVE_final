import os
import sys
import glob
import argparse

import cv2
import numpy as np
from scipy import misc

from step1_local_extrema import find_local_extrema
from step2_envelope import compute_envelope
from step3_smoothed_mean import compute_smoothed_mean


if __name__ == '__main__':
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default = 'mountain', type = str)
    parser.add_argument("--ext", default = 'jpg', type = str)
    parser.add_argument("--output_dir", default = 'output', type = str)
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--rgb", action='store_true')

    parser.add_argument("--channel", default = 'gray', type = str)

    parser.add_argument("--k", default = 3, type = int)
    args = parser.parse_args()


    args.output_dir = args.output_dir + '/{}_{}_{}'.format(args.filename, 'color' if args.rgb else 'gray', args.k)
    os.makedirs(args.output_dir, exist_ok = True)


    # load image
    filename = './img/{}.{}'.format(args.filename, args.ext)
    img = cv2.imread(filename)

    # resize iamge if necessary
    #img = cv2.resize(img, (int(img.shape[1]/5), int(img.shape[0]/5)), interpolation=cv2.INTER_CUBIC)


    # decomposition in grayscale
    if not args.rgb:
        args.channel = channel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # step 1
        matMin, matMax = find_local_extrema(args, gray, k=args.k)

        # step 2
        envMin, envMax = compute_envelope(args, matMin, matMax, gray)

        # step 2
        mean, diff = compute_smoothed_mean(args, envMin, envMax)


    # decomposition in RGB
    if args.rgb:
        mean_rgb, diff_rgb = np.zeros(img.shape), np.zeros(img.shape)
        for c_i, channel in enumerate(['B', 'G', 'R']):
            args.channel = channel
            gray = img[:, :, c_i]

            # step 1
            matMin, matMax = find_local_extrema(args, gray, k=args.k)

            # step 2
            envMin, envMax = compute_envelope(args, matMin, matMax, gray)

            # step 2
            mean, diff = compute_smoothed_mean(args, envMin, envMax)

            mean_rgb[:, :, 2 - c_i] = mean
            diff_rgb[:, :, 2 - c_i] = diff

        if args.save:
            misc.imsave(os.path.join(args.output_dir, "mean_rgb.bmp"), mean_rgb)
            misc.imsave(os.path.join(args.output_dir, "diff_rgb.bmp"), diff_rgb)
