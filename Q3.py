import numpy as np
import cv2
import time
from util.preprocessing_util import preprocess
from util.matching_util     import orb_matching

def generate_images(ref_img,width,height):
    """
    From the initial reference image, generate 2 sub images and their relative translation.

    parameters:
    ----------
        ref_img: array_like (m,n)
            Matrix representing the reference image
    
    returns:
    --------
        im1: array_like (w,h)
            Matrix representing a sub image of the reference image chosen randomly
        im2: array_like (w,h)
            Matrix representing a sub image of the reference image where a translation of max 20px is applied compared to im1
        dx : int
            Number of pixels translated in im2 in the im1's pixel coordinate along x-axis
        dy : int 
            Number of pixels translated in im2 in the im1's pixel coordinate along y-axis

    """

    m,n = ref_img.shape

    # upper left pixel of im1 in ref_img pixel coordinate
    x1 = np.random.choice(np.arange(20,m-20-width+1,1))
    y1 = np.random.choice(np.arange(20,n-20-height+1,1))

    # translation applied to im2
    tx = np.random.choice(np.arange(-20,21,1))
    tx = np.random.choice(np.arange(-20,21,1))

    # upper left pixel of im2 in ref_img pixel coordinate
    x2 = x1 + tx
    y2 = y1 + tx

    im1 = ref_img[x1:x1+width, y1:y1+height]
    im2 = ref_img[x2:x2+width, y2:y2+height]


    return im1,im2,tx,tx

def intersect(x1,x2,shift):
    """
    Compute intersection of x1 and x2 when a shift is applied to x2.\n
    Ie., Remove region of x1 and x2 that do not overlap when a shift is applied to x2.e

    parameters
    ----------
        x1: array_like (n,)
        First signal
        x2: array_like (n,)
        Second signal
    
    return
    ------
        xx1: array_like (m,)
        The signal x1 but non-overlapping region with x2 are removed
        xx2: array_like (m,)
        The signal x2 but non-overlapping region with x1 are removed
    """

    if shift>0:
        xx1 = x1[shift:]
        xx2 = x2[:-shift]
    elif shift<0:
        xx1 = x1[:shift]
        xx2 = x2[-shift:]
    else:
        xx1 = x1
        xx2 = x2
    return xx1,xx2

def compute_transformation(im1,im2):
    """
    From 2 images, compute the transformation matrix T such that im1(x,y) = T @ im2(x,y).\n This function can only deal with translation.

    parameters
    ----------
        img_1: array_like (m,n)
            Matrix representing the reference image
        img_2: array_like (m,n)
            Matrix representing the reference image
    return
    ------
        T : array_like (3,3)
            Transformation matrix such that im1(x,y) = T @ im2(x,y)

    """
    x1 = np.sum(im1,axis=1,dtype=np.double)
    y1 = np.sum(im1,axis=0,dtype=np.double)
    x2 = np.sum(im2,axis=1,dtype=np.double)
    y2 = np.sum(im2,axis=0,dtype=np.double)

    best_corr_x = 0
    best_corr_y = 0

    for t in np.arange(-20,21,1):
        # remove non-overlapping region between x1-x2 or y1-y2 in order to compare them
        xx1,xx2 = intersect(x1,x2,t)
        yy1,yy2 = intersect(y1,y2,t)

        # compute normed correlation in both x-axis and y-axis
        corr_x = np.corrcoef(xx1,xx2)[0,1]#np.dot(xx1,xx2)/ (np.linalg.norm(xx1)*np.linalg.norm(xx2))
        corr_y = np.corrcoef(yy1,yy2)[0,1]#np.dot(yy1,yy2)/ (np.linalg.norm(yy1)*np.linalg.norm(yy2))

        # store the best translation for both axis
        if corr_x > best_corr_x:
            best_corr_x = corr_x
            tx = t
        if corr_y > best_corr_y:
            best_corr_y = corr_y
            ty = t

    return np.array([[1,0,tx],[0,1,ty],[0,0,1]])

def evaluate_performance(im_ref,width,height,n_samples=1000,method='dummy'):
    """
    Analyze my method
    
    parameters
    ----------
        im_ref: matrix
            Name/ Path of the reference image
        width: int
            width of the subimages (number of pixels)
        height: int
            height of the subimages (number of pixels)
        n_sample: int
            Number of samble on which we analyse
        method: proposed-method/ ORB:
            parameter to chose which method to compute translation
    
    returns
    -------
        tp: int
            Number of well found transform
        dt: float
            Execution time
    """

    tp = 0 # True positive
    dt = 0 # Execution time

    for _ in range(n_samples):
        im1,im2,true_tx,true_ty = generate_images(im_ref,width=width,height=height)
        
        t0      = time.time()
        if   method=='proposed-method':
            T       = compute_transformation(im1,im2)
        elif method=='ORB':
            # The input arrays should have at least 4 corresponding point sets 
            # to calculate Homography in function 'cv::findHomography'
            kp1, des1, kp2, des2, matches, T = orb_matching(im1,im2)


        t1      = time.time()
        tx      = int(np.round(T[0,2],0))
        ty      = int(np.round(T[1,2],0))
        

        if true_tx==tx and true_ty==ty:
            tp += 1

        dt += t1-t0
    
    return tp,dt


    

if __name__ == '__main__':

    ref_filename = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    im_ref = cv2.imread(ref_filename,-1)
    im_ref = preprocess(im_ref)
    sizes = [400,800,1200,1600,2000,2400,2600,3000]
    n_samples = 100

    for size in sizes:
        tp,dt = evaluate_performance(im_ref,width=size,height=size,n_samples=n_samples,method='proposed-method')
        print('(size= %d) mAcc = %f     mean time = %f  tp= %d'%(size,tp/n_samples,dt/n_samples,tp))