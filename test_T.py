import numpy as np
import cv2
from matplotlib import pyplot as plt

def ResizeWithAspectRatio(image, width=None, inter=cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]

    if width is None:
        return image
    else:
        r = width / float(w)
        dim = (int(w * r), int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

im1 = cv2.imread('S2A_MSIL1C_20230312T042701_N0509_R133_T45QZF_20230312T062152.SAFE\GRANULE\L1C_T45QZF_A040313_20230312T043803\IMG_DATA\T45QZF_20230312T042701_B04.jp2')
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

im2 = cv2.imread('S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2') 
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

im1 = cv2.equalizeHist(im1)
im2 = cv2.equalizeHist(im2)

