import numpy as np
import cv2
from matplotlib import pyplot as plt

def resizeWithAspectRatio(image, width=None):
    dim = None
    h, w = image.shape[:2]

    if width is None:
        return image
    else:
        r = width / float(w)
        dim = (int(w * r), int(h * r))

    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def preprocess(im,equalize=False,normalize=False,width=-1):
    """
        return an image that is preprocessed (converted to 8bit / equalized / normalized / re-sized)

        parameters
        ----------
            im: matrix
                Intensity matrix of a given image
            equalize: bool, (optional)
                Parameter which allows image equalization before displaying
            normal: bool, (optional)
                Parameter which allows image normalization before displaying
            width: int, (optional)
                Parameter which resize the size of the image with a given width (pixels). height depend on width to keep image proportion
        return
        ------
            im: matrix
                Intensity matrix of a given image after preprocessing
    """
    
    if im.dtype != 'uint8':

        ratio = 1/16
        im = (im*ratio).astype('uint8')

    if normalize:
        im = cv2.normalize(im, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    if equalize:
        im = cv2.equalizeHist(im)
    
    if width != -1:
        im = resizeWithAspectRatio(im,width)

    return im

if __name__ == '__main__':
    ref_path = 'S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    reg_path = 'S2A_MSIL1C_20230312T042701_N0509_R133_T45QZF_20230312T062152.SAFE\GRANULE\L1C_T45QZF_A040313_20230312T043803\IMG_DATA\T45QZF_20230312T042701_B04.jp2'

    preprocess(ref_path,'B04_im1_T46.jp2')
    preprocess(reg_path,'B04_im2_T45.jp2')
    


