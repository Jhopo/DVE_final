import os
import sys
import glob
import argparse

import cv2
import numpy as np


img = cv2.imread('./img/example.png')
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)


print (yuv[:,:,0])
print (yuv[:,:,1])
print (yuv[:,:,2])
