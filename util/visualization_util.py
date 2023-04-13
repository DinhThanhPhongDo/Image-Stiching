import numpy as np
import cv2
from matplotlib import pyplot as plt
import os


def show(im,title=None,width=580,figsize=(5,5),equalize=False):
    """
        Display an image with a specified title / width / figsize.
        
        parameters
        ----------
            im: matrix
                Intensity matrix of a given image
            title: string, optional
                title of the displayed image
            width: int, optional
                width of the  displayed image (number of pixels). The height is deduced by keeping the proportion
            equalize: bool, (optional)
                Parameter which allows image equalization before displaying
    """

    if equalize:
        im = cv2.equalizeHist(im)

    h, w = im.shape[:2]
    r = width / float(w)
    dim = (int(w * r), int(h * r))

    tmp = cv2.resize(im, dim, interpolation=cv2.INTER_AREA)
    plt.figure(figsize = figsize,dpi=500)
    plt.imshow(cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB),aspect='equal')
    if title != None:
        plt.title(title,fontsize=5)
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.show()
    return


def drawMatches(im1, kp1, im2, kp2, matches, equalize=False):
    """
        Display 2 images with their associated keypoints and the matching between them.\n
        Inspired by:
        https://stackoverflow.com/questions/11114349/how-to-visualize-descriptor-matching-using-opencv-module-in-python

        parameters
        ----------
            im1, im2: Matrices
                Intensity matrix of a given image

            kp1, kp2: List of keypoints (opencv)

            matches: List of matches (opencv)

            equalize: bool
                Parameter which allows image equalization before displaying

    """

    # Create a new output image that concatenates the two images together
    rows1 = im1.shape[0]
    cols1 = im1.shape[1]
    rows2 = im2.shape[0]
    cols2 = im2.shape[1]
    if equalize:
        im1 = cv2.equalizeHist(im1)
        im2 = cv2.equalizeHist(im2)

    out = np.zeros((max(rows1,rows2),cols1+cols2,3), dtype='uint8')

    # Place the first image to the left
    out[:rows1,:cols1] = np.dstack([im1, im1, im1])

    # Place the next image to the right of it
    out[:rows2,cols1:cols1+cols2,:] = np.dstack([im2, im2, im2])

    # Draw keypoints
    for kp in kp1:
        (x,y) = kp.pt
        cv2.circle(out, (int(x),int(y)), int(kp.size), (0, 255, 0), 16)
    for kp in kp2:
        (x,y) = kp.pt
        cv2.circle(out, (int(x)+cols1,int(y)), int(kp.size), (0, 255, 0), 16)

    # Draw matches
    for mat in matches:

        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 180, 0), 16)

    return out

