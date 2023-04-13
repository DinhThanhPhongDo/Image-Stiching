from __future__ import division

import rasterio
import rasterio.warp
import rasterio.merge
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil

from rasterio.vrt import WarpedVRT

import numpy as np


def project_boundingbox(reg_filename,ref_filename,out_filename):
    """
        Project the raster to be registered (stored in reg_filename), to the bounding box of the reference raster
        (stored in ref_filename) and save this new raster in out_filename. The method use a cubic ressampling.

        parameters
        ----------
        ref_filename: string
            Name / path of the reference raster
        reg_filename: string
            Name / path of the raster to be registered
        out_filename: string
            Name / path of the raster to be stored after the projection
    """

    dst_dataset = rasterio.open(ref_filename)
    vrt_options = {
        'resampling': Resampling.cubic,
        'crs': dst_dataset.crs,
        'transform': dst_dataset.transform,
        'height': dst_dataset.height,
        'width': dst_dataset.width,
    }

    with rasterio.open(reg_filename) as src:
        with WarpedVRT(src, **vrt_options) as vrt:
            data = vrt.read()
            for _, window in vrt.block_windows():
                data = vrt.read(window=window)
            
            rio_shutil.copy(vrt, out_filename, driver='JP2OpenJPEG')

    return

def add_padding(in_filename,out_filename,padx,pady,update_transform=False):
    """
        add a padding (padx, pady) to a raster (stored in in_filename), and save the result in a new raster out_filename

        parameters
        ----------
        in_filename: string
            Name / path of the initial raster
        out_filename: string
            Name / path of the raster to be stored after the projection
        padx/pady: int (positive)
            Number of pixels that will be applied on the x-axis and y-axis.
        update_transform: bool (optional)
            Parameter which allows to update the transform (if True) or not (if False) in order to create 
            a wrongly georeferenced raster
        
    """
    data = rasterio.open(in_filename)
    height, width = data.shape
    
    #update height and width parameter to take the padding into account
    height += padx
    width  += pady

    # create a padded matrix 
    pad_data = np.zeros((1,height, width ), dtype=data.read(1).dtype)
    pad_data[:,padx:height , pady :width ] = data.read(1)

    # update metadata of the raster
    out_meta = data.meta.copy()
    out_meta.update({
        "height"   : height,
        "width"    : width,
        })
    if update_transform == True:
        t1 = np.array(data.transform).reshape((3,3))
        t2 = np.array([[1,0,-padx],[0,1,-pady],[0,0,1]])
        tr = t1@t2
        transform = rasterio.Affine(tr[0,0],tr[0,1],tr[0,2],tr[1,0],tr[1,1],tr[1,2])
        out_meta.update({"transform"   : transform,})

    # save the result in a new raster
    with rasterio.open(out_filename, 'w', **out_meta) as dst:
        dst.write(pad_data)
    return

def project_and_merge(reg_filename,ref_filename,out_filename):
    """
        Project the raster to be registered (stored in reg_filename), to the reference raster
        (stored in ref_filename) by changing its CRS and save this new raster in out_filename. The method use a cubic ressampling.

        parameters
        ----------
        ref_filename: string
            Name / path of the reference raster
        reg_filename: string
            Name / path of the raster to be registered
        out_filename: string
            Name / path of the raster to be stored after the projection
    """
    with rasterio.open(ref_filename) as dst:
        # reproject to the reference CRS
        with rasterio.open(reg_filename) as src:
        
            # compute the transform in the reference CRS
            transform, width, height = rasterio.warp.calculate_default_transform(src.crs, dst.crs, src.width, src.height, *src.bounds)
            out_meta = src.meta.copy()

            out_meta.update({
                'crs': dst.crs,
                'transform': transform,
                'width': width,
                'height': height})
            
            with rasterio.open(out_filename, 'w', **out_meta) as dst2:
                rasterio.warp.reproject(
                    source        = rasterio.band(src, 1),
                    destination   =  rasterio.band(dst2,1),
                    src_transform = src.transform,
                    src_crs       = src.crs,
                    dst_transform = dst.transform,
                    dst_crs       = dst.crs,
                    resampling    = Resampling.cubic)

        # Once in the same CRS, merge both rasters together
        with rasterio.open(out_filename) as src:       
            mosaic, out_transf = rasterio.merge.merge([src,dst],resampling=Resampling.cubic)
            out_meta = dst.meta.copy()
            out_meta.update({
                "height"   : mosaic.shape[1],
                "width"    : mosaic.shape[2],
                "transform": out_transf,
                "crs"      : dst.crs
                })
        with rasterio.open(out_filename, "w", **out_meta) as dst:
            dst.write(mosaic)


    return out_filename


if __name__ == '__main__':
    ref_path = "src\S2A_MSIL1C_20230312T042701_N0509_R133_T46QBL_20230312T062152.SAFE\GRANULE\L1C_T46QBL_A040313_20230312T043803\IMG_DATA\T46QBL_20230312T042701_B04.jp2"
    reg_path = "src\S2A_MSIL1C_20230312T042701_N0509_R133_T45QZF_20230312T062152.SAFE\GRANULE\L1C_T45QZF_A040313_20230312T043803\IMG_DATA\T45QZF_20230312T042701_B04.jp2"
    pad = 30
    # # Normalize reg_path (dimension, crs,...) within a pre-defined bounding box given by ref_path 
    # warped_vrt(reg_path,ref_path,'img\warp_T46.jp2')

    # Save modified raster with a padding without changing the transform
    add_padding(reg_path,'img\im2_T45_pad_False.jp2',pad,pad,update_transform=False)
    # Project the wrongly registered raster to the reference raster
    project_and_merge('img\im2_T45_pad_False.jp2',ref_path,'img\merge_False.jp2')
    project_boundingbox('img\im2_T45_pad_False.jp2',ref_path,'img\warp_T46_pad_False.jp2')

    # # Save modified raster with a padding and changing the transform
    add_padding(reg_path,'img\im2_T45_pad_True.jp2',pad,pad,update_transform=True)
    # # Project the well registered raster to the reference raster
    project_and_merge('img\im2_T45_pad_True.jp2',ref_path,'img\merge_True.jp2')
    project_boundingbox('img\im2_T45_pad_True.jp2',ref_path,'img\warp_T46_pad_True.jp2')


    




