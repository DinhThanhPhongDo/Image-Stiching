import rasterio as rio
import rasterio.merge 
import rasterio.features
import rasterio.warp
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.plot import show_hist
from matplotlib import pyplot
import numpy as np


def reproject_raster(src,dst,out_filename):

    """
    src-->dst
    https://stackoverflow.com/questions/60288953/how-to-change-the-crs-of-a-raster-with-rasterio
    """

    #crs = rasterio.CRS({'init': 'EPSG:32646'})
    dst_crs = dst.crs
    # reproject raster to project crs
    src_crs = src.crs
    transform, width, height = rasterio.warp.calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()

    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height})
    print('height hihi =',height)
    
    with rio.open(out_filename, 'w', **kwargs) as dst2:
        for i in range(1, src.count + 1):
            rasterio.warp.reproject(
                source=rio.band(src, i),
                destination= rio.band(dst2,i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst.transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
        
    return out_filename

def merge(out_filename,datasets_lst,bounds=None, res=None, nodata=None, precision=None):
    #The first dataset should be the destination dataset.
    mosaic, out_transf = rasterio.merge.merge(datasets_lst,bounds,res,nodata,precision)

    
    #Update metadata
    dst = datasets_lst[0]
    out_meta = dst.meta.copy()
    out_meta.update({
        "height"   : mosaic.shape[1],
        "width"    : mosaic.shape[2],
        "transform": out_transf,
        "crs"      : dst.crs
        })

    with rasterio.open(out_filename, "w", **out_meta) as dst2:
        dst2.write(mosaic)
    
    return

    


    

if __name__ == '__main__':
    #T46
    reference = 'S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE'
    #T45
    registered= 'S2A_MSIL1C_20230312T042701_N0509_R133_T45QZF_20230312T062152.SAFE'

    # with rasterio.open(reference) as s2a:
    #     subdatasets = s2a.subdatasets


    # with rasterio.open(subdatasets[0]) as b10m:
    #     print(b10m.profile)
    dataset_ref = rio.open("src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2")
    dataset_reg = rio.open("src\S2A_MSIL1C_20230312T042701_N0509_R133_T45QZF_20230312T062152.SAFE\GRANULE\L1C_T45QZF_A040313_20230312T043803\IMG_DATA\T45QZF_20230312T042701_B04.jp2")

    print(dataset_ref.count,dataset_reg.count)
    # #georeferences : ok!
    # show_georeference(dataset_ref) #T46
    # show_georeference(dataset_reg) #T45

    # print('type=',type(dataset_reg))
    # print(dataset_reg.name)
    # print(dataset_reg.mode)
    # print(dataset_reg.closed)
    # print(dataset_reg.width,dataset_reg.height)

    #what is dataset.bounds? why do they not merge toghether well?
    print('--\n reg=',dataset_reg.bounds,'\n',dataset_reg.transform,'\n--')
    print('\n \n')
    print('------Registered Image------')
    print('Coordinate Reference System:', dataset_reg.crs)
    print('Coordinate Transformation:')
    print(dataset_reg.transform)
    print('\n \n')

    print(dataset_reg.profile)

    print('\n \n')
    print('------Reference Image------')
    print('Coordinate Reference System:', dataset_ref.crs)
    print('Coordinate Transformation:')
    print(dataset_ref.transform)
    print('\n \n')


    print('--\n ref=',dataset_ref.bounds,'\n',dataset_ref.transform,'\n--')
    print(dataset_ref.crs)
    print(dataset_ref.profile)

    reproject_raster(dataset_reg,dataset_ref,out_filename='im2.jp2')
    
    im2 = rio.open('im2.jp2')

    print('--\n ref=',im2.bounds,'\n',im2.transform,'\n--')
    print(im2.crs)
    print(im2.profile)

    show(dataset_ref,cmap='gray')
    # show_hist(dataset_ref,bins=10, lw=0.0, stacked=False, alpha=0.3,
    # histtype='stepfilled', title="Histogram")
    show(dataset_reg,cmap='gray')
    # show_hist(dataset_reg,bins=10, lw=0.0, stacked=False, alpha=0.3,
    # histtype='stepfilled', title="Histogram")
    # show(im2)
    
    # merging 2 dataset
    # raster,transf = rasterio.merge.merge([im2,dataset_ref])
    # show(raster)

    dataset_lst  =[dataset_ref,im2]
    merge("mosaic_true.jp2",dataset_lst)

    mosaic = rio.open('mosaic_true.jp2')
    print(mosaic.count)
    print(type(mosaic.read(1)))
    show(mosaic,cmap='gray')
    show_hist(mosaic,bins=256, lw=0.0, stacked=False, alpha=0.3,
    histtype='stepfilled', title="Histogram")
    print(np.max(mosaic.read(1)))


    # print(np.array(dataset_ref.transform).reshape((3,3)))

    T1 = np.array(dataset_ref.transform).reshape((3,3))
    T2 = np.array(im2.transform).reshape((3,3))
    T3 = np.load('T3.npy')
    T3_gt = np.dot(T1,np.linalg.inv(T2))
    T3_gt_inv = np.linalg.inv(T3_gt)

    print('Ground truth')
    print(T3_gt_inv)
    print('ground truth')
    print(T3_gt)
    print('my ORB')
    print(np.load('T3_FULL_ORB.npy'))

    np.save('T1.npy',T1)
    np.save('T2.npy',T2)
    np.save('T3_FULL_gt.npy',T3_gt)
    np.save('T3_FULL_inv.npy',T3_gt_inv)
    # pyplot.imshow(mosaic.read(1))
    # pyplot.show()


    # pyplot.imshow(dataset_reg.read(1), cmap='Reds')
    # pyplot.imshow(dataset_ref.read(1),cmap='Greens')
    # pyplot.show()