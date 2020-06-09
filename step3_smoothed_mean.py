import os
import sys
import glob
import argparse

import cv2
import numpy as np

from scipy import misc


def compute_smoothed_mean(args, envMin, envMax):

    mean = np.clip((envMin + envMax) / 2., 0, 255)
    diff = np.clip((envMax - envMin), 0, 255)

    if args.save:
        misc.imsave(os.path.join(args.output_dir, "mean.bmp"), mean)
        misc.imsave(os.path.join(args.output_dir, "diff.bmp"), diff)

    return mean, diff
