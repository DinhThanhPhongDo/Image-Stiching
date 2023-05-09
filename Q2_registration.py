import os
import numpy as np
import cv2
from util.preprocessing_util import preprocess
from util.visualization_util import show,drawMatches
from util.matching_util     import orb_matching
from util.metrics_util       import rmsd,mutual_information,correlation


def find_registration(ref_filename,reg_filename,out_filename=None,keypoints_method='ORB',vizualisation=False):
    im1 = cv2.imread(ref_filename,-1)
    im2 = cv2.imread(reg_filename,-1)


    if im1.dtype == 'uint16':
        im1 = preprocess(im1)
    if im2.dtype =='uint16':
        im2 = preprocess(im2)
    
    # keypoints method
    if keypoints_method   =='ORB':
        kp1, des1, kp2, des2, matches, T = orb_matching(im1,im2,transform='homography',matcher='flann')
    elif keypoints_method =='SIFT':
        return
    
    if vizualisation == True:
        
        print('Transformation matrix:')
        print(T)

        # Draw keypoints & matches
        show(drawMatches(im1,kp1,im2,kp2,matches[:20],equalize=True),"Keypoints \& Matches",width=1160)
        
        im2 = cv2.warpPerspective(im2,T,im1.shape,borderValue=0)
        add = cv2.addWeighted(im2,0.5,im1,0.6,0)

        show(np.hstack((im2,add)),'both image added',width=1160)
        cv2.waitKey()

    if out_filename != None:
        np.save(out_filename,T)
    return T

def evaluate_registration(ref_filename,reg_filename,mat_filename,metric='mi',vizualisation=False):

    im1 = cv2.imread(ref_filename,-1)
    im2 = cv2.imread(reg_filename,-1)

    if im1.dtype == 'uint16':
        im1 = preprocess(im1)
    if im2.dtype =='uint16':
        im2 = preprocess(im2)

    T   = np.load(mat_filename)

    im2 = cv2.warpPerspective(im2,T,im1.shape,borderValue=0,flags=cv2.INTER_LINEAR)#INTER_CUBIC

    mask    = (im2!=0)

    # Evaluate the performance of the registration with 2 metrics:
    #   1. Root Mean Square Difference (RMSD)
    #   2. Cross-correlation

    if metric == 'mi' :
        mi,nmi = mutual_information(im1,im2,mask,vizualisation=vizualisation)
        # print('mi= {:< 3.5f} nmi= {:< 3.5f}'.format(mi,nmi))
        print('mi=', mi,'nmi=', nmi)
    if metric == 'mse':
        root_mse,exp,var = rmsd(im1,im2,mask,vizualisation=vizualisation)
        print('rmse=',root_mse,'exp=',exp,'var=',var)
    if metric == 'cc':
        cc = correlation(im1,im2,mask,vizualisation=vizualisation)
        print('cross correlation=',cc)
    return
    


if __name__ == '__main__':
    print(cv2.__version__)

    # # registration and evaluation of non-padded images
    # print('----------------------------not pad case---------------------------')
    # ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    # reg = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T45QZF_20230312T062152.SAFE\GRANULE\L1C_T45QZF_A040313_20230312T043803\IMG_DATA\T45QZF_20230312T042701_B04.jp2'
    # tr  = 'mat\T_ORB.npy'
    # find_registration(ref,reg,tr,vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mse',vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mi',vizualisation=False)

    # # comparison with the result obtained with the transform provided by sentinel2
    # print('----------- sentinel result-----------')
    # ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    # reg = 'img\warp_T46.jp2'
    # tr  = 'mat\identity.npy'
    # find_registration(ref,reg,tr,vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mse',vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mi',vizualisation=False)


    # registration and evaluation of padded images
    print('test small case')
    print('------------------------------pad case---------------------------')
    ref = 'B04_im1_T46.jp2'
    reg = 'B04_im2_T45.jp2'
    tr  = 'mat\optest.npy'
    find_registration(ref,reg,tr,vizualisation=True)
    evaluate_registration(ref,reg,tr,metric='mse',vizualisation=True)
    evaluate_registration(ref,reg,tr,metric='cc',vizualisation=True)

    # ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    # reg = 'img\im2_45_pad30_False.jp2'
    # # reg = 'C:\Users\dinht\Documents\GitHub\Aerospacelab\img\im2_T45_pad_True.jp2'
    # tr  = 'mat\T_ORB1_pad.npy'
    # find_registration(ref,reg,tr,vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mse',vizualisation=False)
    # evaluate_registration(ref,reg,tr,metric='cc',vizualisation=False)

    # print('------------------------------pad case (with good reprojection)---------------------------')
    # ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    # reg = 'img\warp_T46_pad_False.jp2'
    # # reg = 'C:\Users\dinht\Documents\GitHub\Aerospacelab\img\im2_T45_pad_True.jp2'
    # tr  = 'mat\T_ORB2_pad.npy'
    # find_registration(ref,reg,tr,vizualisation=False)
    # evaluate_registration(ref,reg,tr,metric='mse',vizualisation=False)
    # evaluate_registration(ref,reg,tr,metric='cc',vizualisation=False)

    # # # comparison with the result obtained with the transform provided by sentinel2
    # print('----------- sentinel result-----------')
    # ref = 'src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2'
    # reg = 'img\warp_T46_pad_True.jp2'
    # tr  = 'mat\identity_pad.npy'
    # find_registration(ref,reg,tr,vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='mse',vizualisation=True)
    # evaluate_registration(ref,reg,tr,metric='cc',vizualisation=False)






    


    


