import cv2
import numpy as np

def orb_matching(im1,im2,matcher='bruteforce',transform='homography'):
    """
        Perform ORB keypoints detector and descriptor. Use a feature matching method, and then find the transformation matrix

        parameters
        ----------
            im1, im2: matrix 
                Intensity matrices of reference image (im1) and image to be registered (im2)
            matcher: 'bruteforce' / 'flann
                Specify the feature matching method used
            transform: 'affine' / 'homography'
                Specify the type of transformation matrix needed to be found
        
        return
        ------
            kp1 , kp2  : List of keypoints (opencv)
            des1, des2 : List of descriptors (opencv)
            matches    : List of matches (opencv)
            T_orb      : 3x3 transformation matrice (np array)
    """
    # Keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(im1, None)  #500 features
    kp2, des2 = orb.detectAndCompute(im2, None)  #500 features

    # Feature matching method
    if   matcher == 'bruteforce':
        points1, points2, matches = bf_matcher(kp1,kp2,des1,des2, normType=cv2.NORM_HAMMING, crossCheck=True)

    elif matcher == 'flann':
        FLANN_INDEX_LSH = 6

        index_params  = dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6, # 12
                            key_size = 12,     # 20
                            multi_probe_level = 1) #2
        
        search_params = dict(checks=50)   # or pass empty dictionary

        points1, points2, matches = flann_matcher(kp1,kp2,des1,des2,index_params,search_params)

    # Find transform
    if   transform=='homography':
        T_orb,mask_ORB = cv2.findHomography(points2,points1,method=cv2.RANSAC,confidence=0.97)

    elif transform=='affine':
        T_orb,mask_ORB = cv2.estimateAffine2D(points2,points1,method=cv2.RANSAC,confidence=0.97)
        T_orb = np.concatenate((T_orb,np.array([[0,0,1]])),axis=0)

    return kp1, des1, kp2, des2, matches, T_orb

# def sift_matching(im1,im2,transform='homography',matcher='bruteforce'):
#     """
#         Perform SIFT keypoints detector and descriptor. Use a feature matching method, and then find the transformation matrix

#         parameters
#         ----------
#             im1, im2: matrix 
#                 Intensity matrices of reference image (im1) and image to be registered (im2)
#             matcher: 'bruteforce' / 'flann
#                 Specify the feature matching method used
#             transform: 'affine' / 'homography'
#                 Specify the type of transformation matrix needed to be found
        
#         return
#         ------
#             kp1 , kp2  : List of keypoints (opencv)
#             des1, des2 : List of descriptors (opencv)
#             matches    : List of matches (opencv)
#             T_orb      : 3x3 transformation matrice (np array)
#     """
#     # Keypoints and descriptors
#     orb = cv2.SIFT_create()
#     kp1, des1 = orb.detectAndCompute(im1, None)  
#     kp2, des2 = orb.detectAndCompute(im2, None)  

#     # Feature matching method
#     if   matcher == 'bruteforce':
#         points1, points2, matches = bf_matcher(kp1,kp2,des1,des2,normType=cv2.NORM_L2, crossCheck= False)

#     elif matcher == 'flann':

#         FLANN_INDEX_KDTREE = 1
#         index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
#         search_params = dict(checks=50)   # or pass empty dictionary
#         points1, points2, matches = flann_matcher(kp1,kp2,des1,des2,index_params,search_params)
    

#     # Find transform
#     if   transform=='homography':
#         T_SIFT,mask_SIFT = cv2.findHomography(points2,points1,method=cv2.RANSAC,confidence=0.97)

#     elif transform=='affine':
#         T_SIFT,mask_SIFT = cv2.estimateAffine2D(points2,points1,method=cv2.RANSAC,confidence=0.97)
#         T_SIFT = np.concatenate((T_SIFT,np.array([[0,0,1]])),axis=0)

#     return kp1, des1, kp2, des2, matches, T_SIFT

def flann_matcher(kp1,kp2,des1,des2, index_params, search_params):

    flann         = cv2.FlannBasedMatcher(index_params,search_params)
    matches       = flann.knnMatch(des1,des2,k=2)

    # return only good matches (Lowe's paper)
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append([m])

    points1 = np.zeros((len(good), 2), dtype=np.float32)  
    points2 = np.zeros((len(good), 2), dtype=np.float32)

    for i, match in enumerate(good):
        points1[i, :] = kp1[match[0].queryIdx].pt   #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match[0].trainIdx].pt   #gives index of the descriptor in the list of train descriptors

    
    good = sorted(matches, key = lambda x:x[0].distance)
    
    return points1, points2, good

def bf_matcher(kp1,kp2,des1,des2,normType,crossCheck):

    matcher = cv2.BFMatcher(normType,crossCheck)
    matches = matcher.match(des1,des2)

    points1 = np.zeros((len(matches), 2), dtype=np.float32)  
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    matches = sorted(matches, key = lambda x:x.distance)

    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt   #gives index of the descriptor in the list of query descriptors
        points2[i, :] = kp2[match.trainIdx].pt   #gives index of the descriptor in the list of train descriptors
    
   

    return points1, points2, matches