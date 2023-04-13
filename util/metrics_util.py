import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def rmsd(im1,im2,is_inliers,vizualisation = False):

    # Difference between registered and reference image
    diff    = im2.astype(np.int16)-im1.astype(np.int16)
    # Root mean square difference
    rmsd    = np.sqrt(np.sum(is_inliers*np.power(diff,2),dtype=np.double)/(np.sum(is_inliers)))

    # Expectation and Variance of diff
    exp     = np.mean(diff[is_inliers == True].ravel())
    var     = np.var(diff[is_inliers == True].ravel())
    if vizualisation:

        # Distribution of diff 
        plt.figure(figsize=(5,5))
        plt.grid()
        plt.hist(diff[is_inliers == True].ravel(),25*2+1,[-25,25],density=True)
        plt.title('Intensity Differences Distribution \nE = %f      Var= %f      RMSD=%f'%(exp,var, rmsd))
        plt.ylabel('Density')
        plt.xlabel('Intensity Differences')
        plt.show()

        # Matrix to show where errors appear
        fig = plt.figure(figsize=(5,5))
        ax  = plt.axes()
        im  = ax.matshow(is_inliers*np.abs(diff),norm=LogNorm())
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title('Spatial Distribution of Intensity \n Differences')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    
    return rmsd,exp,var

def correlation(im1,im2,is_inlier,vizualisation=False):
    N = np.sum(is_inlier)
    mean_im1 = np.mean(im1[is_inlier==True])
    mean_im2 = np.mean(im2[is_inlier==True])

    # Standard deviation of intensity values
    s_im1 = np.sqrt((1/(N-1))*np.sum(np.power(im1[is_inlier==True]-mean_im1,2,dtype=np.double)))
    s_im2 = np.sqrt((1/(N-1))*np.sum(np.power(im2[is_inlier==True]-mean_im2,2,dtype=np.double)))
    # Covariance
    s_12  = (1/(N-1))*np.sum((im1[is_inlier==True]-mean_im1)*(im2[is_inlier==True]-mean_im2),dtype=np.double)
    # Correlation
    corr = s_12/(s_im1*s_im2)

    if vizualisation:
        fig = plt.figure(figsize=(5,5))
        ax  = plt.axes()
        h   = ax.hist2d(im1[is_inlier == True].ravel(),im2[is_inlier!=0].ravel(),(256,256),[[0,255],[0,255]],density=False,norm=LogNorm())
        ax.set_ylabel('Intensity of Registered Image')
        ax.set_xlabel('Intensity of Reference Image')
        ax.set_title('2D histogram \n'+r' Correlation $\rho_{1,2}$='+' %.3f'%corr)
        ax.grid()
        fig.colorbar(h[3], ax=ax)
        plt.show()

    return s_12/(s_im1*s_im2)

def mutual_information(im1,im2,is_inliers,vizualisation=False):
    # compute the marginal histograms
    pdf_im1, _ = np.histogram(im1[is_inliers == True].ravel(), bins=256,density=True)
    pdf_im2, _ = np.histogram(im2[is_inliers == True].ravel(), bins=256,density=True)

    # compute the joint histogram
    pdf_joint,_,_ = np.histogram2d(im1[is_inliers == True].ravel(), im2[is_inliers == True].ravel(), bins=256, density=True)

    eps = np.finfo(float).eps

    EI1 =  - np.sum(pdf_im1 * np.log(pdf_im1 + eps),dtype=np.double)
    EI2 =  - np.sum(pdf_im2 * np.log(pdf_im2 + eps),dtype=np.double)
    EI12=  - np.sum(pdf_joint*np.log(pdf_joint + eps),dtype=np.double)
    mi = EI1 + EI2 - EI12
    nmi = mi / EI12
    print('EI12',EI12)

    if vizualisation:

        plt.hist(im1[is_inliers == True].ravel(),255,[1,255],color='red', alpha = 0.3,density=True)
        plt.hist(im2[is_inliers == True].ravel(),255,[1,255],color='green', alpha = 0.3,density=True)
        plt.title('Histogram')
        plt.show()

        fig, ax = plt.subplots()
        h       = ax.hist2d(im1[is_inliers == True].ravel(),im2[is_inliers!=0].ravel(),(256,256),[[0,255],[0,255]],density=False,norm=LogNorm())
        fig.colorbar(h[3], ax=ax)
        plt.title('hist2d')
        plt.show()
    return mi,nmi


if __name__ == '__main__':

    x = np.arange(0,21,1)
    y = np.arange(0,42,2)
    corr=1.000
    plt.plot(x,y)
    plt.title('2D histogram \n'+r' Correlation $\rho_{1,2}$='+'%f'%corr)
    plt.show()