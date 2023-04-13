import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from util.preprocessing_util import preprocess
from util.visualization_util import show,drawMatches
from util.matching_util     import orb_matching
from util.metrics_util       import rmsd,correlation



def find_registration(ref_filename,reg_filename,out_filename=None,keypoints_method='ORB',vizualisation=False):
    im1 = cv2.imread(ref_filename,-1)
    im2 = cv2.imread(reg_filename,-1)
    

    if im1.dtype == 'uint16':
        im1 = preprocess(im1)
    if im2.dtype =='uint16':
        im2 = preprocess(im2)

    
    # keypoints method
    if keypoints_method   =='ORB':
        kp1, des1, kp2, des2, matches, T = orb_matching(im1,im2,transform='homography')
    elif keypoints_method =='SIFT':
        return
    
    if vizualisation == True:
        print('Transformation matrix:')
        print(T)

        # Draw keypoints & matches
        show("Keypoints & Matches", drawMatches(im1,kp1,im2,kp2,matches[:20],equalize=True),width=2700)
        
        im2 = cv2.warpPerspective(im2,T,im1.shape,borderValue=0)
        add = cv2.addWeighted(im2,0.5,im1,0.6,0)

        show('Weighted sum of both images',np.hstack((im2,add)),width=1160)
        cv2.waitKey()

    # Saving the transformation matrix for later
    if out_filename != None:
        np.save(out_filename,T)

    return T

def evaluate_registration(ref_filename,reg_filename,mat_filename,metric='mi',vizualisation=False):

    im1 = cv2.imread(ref_filename,-1)
    im2 = cv2.imread(reg_filename,-1)

    # Convert to dtype = 'uint8' (compatibility with opencv)
    if im1.dtype == 'uint16':
        im1 = preprocess(im1)
    if im2.dtype =='uint16':
        im2 = preprocess(im2)

    # Load previously stored transformation matrix and apply a perspective transform 
    # on the image to be referenced
    T   = np.load(mat_filename)
    im2 = cv2.warpPerspective(im2,T,im1.shape,borderValue=0,flags=cv2.INTER_LINEAR)#INTER_CUBIC

    # drop zero intensities pixels (due to padding)
    is_inlier    = (im2!=0)

    # Visualize the intensity histogram of both images 
    if vizualisation:
        plt.grid()
        plt.hist(im1[is_inlier == True].ravel(),256,[0,255],color='red'  , alpha = 0.3, density=True,label='reference image')
        plt.hist(im2[is_inlier == True].ravel(),256,[0,255],color='green', alpha = 0.3, density=True,label='registered image')
        plt.title('Histogram of registered and reference images')
        plt.ylabel('Density')
        plt.xlabel('Intensity')
        plt.legend(loc="upper left")
        plt.legend()
        plt.show()

    # Evaluate the performance of the registration with 2 metrics:
    #   1. Root Mean Square Difference (RMSD)
    #   2. Correlation
    
    root_msd,exp,var = rmsd(im1,im2,is_inlier,vizualisation=vizualisation)
    cc               = correlation(im1,im2,is_inlier,vizualisation=vizualisation)

    return root_msd,exp,var,cc
    


if __name__ == '__main__':

    # # registration and evaluation of non-padded images
    print('----------------------------not pad case---------------------------')
    ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    reg = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T45QZF_20230312T062152.SAFE\GRANULE\L1C_T45QZF_A040313_20230312T043803\IMG_DATA\T45QZF_20230312T042701_B04.jp2'
    tr  = 'mat\T_ORB.npy'
    find_registration(ref,reg,tr,vizualisation=True)
    evaluate_registration(ref,reg,tr,metric='mse',vizualisation=True)
    evaluate_registration(ref,reg,tr,metric='mi',vizualisation=False)

    # # comparison with the result obtained with the transform provided by sentinel2
    # print('----------- sentinel result-----------')
    # ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    # reg = 'img\warp_T46.jp2'
    # tr  = 'mat\identity.npy'
    # find_registration(ref,reg,tr,vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mse',vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mi',vizualisation=False)


    # registration and evaluation of padded images
    print('------------------------------pad case---------------------------')
    ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    reg = 'img\im2_T45_pad_False.jp2'
    # reg = 'C:\Users\dinht\Documents\GitHub\Aerospacelab\img\im2_T45_pad_True.jp2'
    tr  = 'mat\T_ORB1_pad.npy'
    find_registration(ref,reg,tr,vizualisation=False)
    evaluate_registration(ref,reg,tr,metric='mse',vizualisation=False)
    evaluate_registration(ref,reg,tr,metric='cc',vizualisation=False)

    print('------------------------------pad case (with good reprojection)---------------------------')
    ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    reg = 'img\warp_T46_pad_False.jp2'
    # reg = 'C:\Users\dinht\Documents\GitHub\Aerospacelab\img\im2_T45_pad_True.jp2'
    tr  = 'mat\T_ORB2_pad.npy'
    find_registration(ref,reg,tr,vizualisation=False)
    evaluate_registration(ref,reg,tr,metric='mse',vizualisation=False)
    evaluate_registration(ref,reg,tr,metric='cc',vizualisation=False)

    # # comparison with the result obtained with the transform provided by sentinel2
    print('----------- sentinel result-----------')
    ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    reg = 'img\warp_T46_pad_True.jp2'
    tr  = 'mat\identity_pad.npy'
    find_registration(ref,reg,tr,vizualisation=True)
    evaluate_registration(ref,reg,tr,metric='mse',vizualisation=True)
    evaluate_registration(ref,reg,tr,metric='cc',vizualisation=False)






    


    


