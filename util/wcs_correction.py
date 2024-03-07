import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord,skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder,IRAFStarFinder
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndimage_shift

from util.sphot_workflow import plot_image_logscale

def apply_sigma_clip(data,sigma):
    from astropy.stats import sigma_clip
    clip = sigma_clip(data,sigma=sigma)
    return data[~clip.mask], ~clip.mask

def identify_matching_sources(image_ref,image_tgt,filter_ref,
                              filter_tgt,fwhms,cutout_wcs,
                              obj_threshold_sigma = 2.5,
                              source_offset_sigma_clip = 3,
                              max_offset_arcsec = 0.06,
                              plot=True,
                              markersize=80):
    '''
    Inputs:
        image_ref (2D array): reference image
        image_tgt (2D array): target image
        filter_ref (str): filter of the reference image
        filter_tgt (str): filter of the target image
        fwhms (list): FWHM of the images (e.g., [fwhm_ref,hwhm_tgt]) in pixels
    '''
    # step 1: initial shift correction
    # first perform cross-correlation to roughly align the images
    # select reference and target images
    # BE CAREFUL: the "shift" indices are [y,x] (i.e. [dec,ra])

    shift, error, diffphase = phase_cross_correlation(image_ref,image_tgt,
                                                    upsample_factor=10,
                                                    reference_mask=~np.isnan(image_ref), 
                                                    moving_mask=~np.isnan(image_tgt))
    shifted_tgt = ndimage_shift(image_tgt, shift)
    shifted_tgt[shifted_tgt==0.] = np.nan

    # step 2: catalog preparation
    # Perform source detection on both images.
    coords = []
    cats = []
    # shifts = [[0,0],shift[::-1]]
    for img,fwhm in zip([image_ref,image_tgt],fwhms):
        # background subtraction
        mean, median, std = sigma_clipped_stats(img, sigma=3.0, maxiters=15)
        img_bgsub = img - median

        # source detection
        star_finder = IRAFStarFinder(
            threshold=obj_threshold_sigma*std, 
            fwhm=fwhm.to(u.pixel).value)
        sources = star_finder.find_stars(img_bgsub)
        xs,ys = sources['xcentroid'],sources['ycentroid']
        cat = pixel_to_skycoord(xs,ys,cutout_wcs)
        coords.append([xs,ys])
        cats.append(cat)
        
    # account for the shift
    pixel_scale = cutout_wcs.proj_plane_pixel_scales()[0].to(u.arcsec) / u.pixel
    delta_ra  = shift[1] * u.pixel * pixel_scale
    delta_dec = shift[0] * u.pixel * pixel_scale
    cats_tgt_shifted = SkyCoord([coord.spherical_offsets_by(-delta_ra,delta_dec) for coord in cats[1]])
        
    # step 3: catalog matching
    # identify matching sources
    idx,d2d,d3d = cats_tgt_shifted.match_to_catalog_sky(cats[0])
    offset_r = cats_tgt_shifted.separation(cats[0][idx]).to(u.arcsec).value
    offset_theta = cats_tgt_shifted.position_angle(cats[0][idx]).to(u.deg).value
    _,mask = apply_sigma_clip(offset_r,source_offset_sigma_clip)
    mask = mask & (offset_r < max_offset_arcsec)
    masked_idx = idx[mask]

    # coordinates of identified sources 
    ref_x = coords[0][0][masked_idx] # pixel position in image_ref    
    ref_y = coords[0][1][masked_idx] # pixel position in image_ref    
    tgt_x = coords[1][0][mask] + shift[1] # pixel position in shifted_tgt
    tgt_y = coords[1][1][mask] + shift[0] # pixel position in shifted_tgt
    tgt_x_beforeshift = coords[1][0][mask] # pixel position in image_tgt
    tgt_y_beforeshift = coords[1][1][mask] # pixel position in image_tgt
       

    sky_ref = cats[0][masked_idx] #pixel_to_skycoord(ref_x,ref_y,cutout_wcs)
    sky_tgt = cats[1][mask] #pixel_to_skycoord(tgt_x_beforeshift,tgt_y_beforeshift,cutout_wcs)

    if plot:
        # plot results
        fig,axes = plt.subplots(3,3,figsize=(13,13))
        
        # raw images
        norm1,offset1 = plot_image_logscale(image_tgt,ax=axes[0,0])
        norm2,offset2 = plot_image_logscale(image_ref,ax=axes[0,1])
        axes[0,2].imshow(norm1(image_tgt+offset1)-norm2(image_ref+offset2),origin='lower',cmap='gray')#,ax=axes[2])
        axes[0,2].set_xticks([])
        axes[0,2].set_yticks([])
        axes[0,0].set_title('raw target image ('+filter_tgt+')')
        axes[0,1].set_title('reference image ('+filter_ref+')')
        axes[0,2].set_title('residual (target - reference) image')

        # cross-correlation shift results
        norm1,offset1 = plot_image_logscale(shifted_tgt,ax=axes[1,0])
        norm2,offset2 = plot_image_logscale(image_ref,ax=axes[1,1])
        axes[1,2].imshow(norm1(shifted_tgt+offset1)-norm2(image_ref+offset2),origin='lower',cmap='gray')#,ax=axes[2])
        axes[1,2].set_xticks([])
        axes[1,2].set_yticks([])
        axes[1,0].set_title(f'shifted target image:\nΔRA={shift[1]}, ΔDEC={shift[0]} (pix)')
        axes[1,1].set_title('reference image')
        axes[1,2].set_title('residual (shifted - reference) image')

        # detected sources -- tgt
        plot_image_logscale(image_tgt,ax=axes[2,0],percentiles=[0.1,99.99],cmap='gray_r')
        axes[2,0].scatter(coords[1][0],coords[1][1],
                        ec='b',fc='none',s=markersize,marker='o',label='tgt')
        axes[2,0].set_title('Bright & isolated stars detected\nin raw tgt image')
        
        # detected sources -- ref
        plot_image_logscale(image_ref,ax=axes[2,1],percentiles=[0.1,99.99],cmap='gray_r')
        axes[2,1].scatter(coords[0][0],coords[0][1],
                        ec='r',fc='none',s=markersize,label='ref',marker='s')

        axes[2,1].set_title('Bright & isolated stars detected\nin ref image')



        # detected sources
        plot_image_logscale(image_ref,ax=axes[2,2],percentiles=[0.1,99.99],cmap='gray_r')
        axes[2,2].scatter(ref_x,ref_y,
                        ec='r',fc='none',s=markersize,label='ref',marker='s')
        axes[2,2].scatter(tgt_x,tgt_y,
                        ec='b',fc='none',s=markersize,marker='o',label='tgt')
        axes[2,2].set_title('cross-matched stars\n(w/ shifted tgt coordinates)')
        axes[2,2].legend(frameon=True)

    return sky_ref,sky_tgt

def shift_and_rotate(xy,shift_x,shift_y,angle,center):
    ''' shift and rotate xy by shift and angle, respectively, around center. All operations are performed in pixel coordinates, and coordinates are assumed to be equally spaced in both xy directions.
    Inputs:
        xy (2xN array): x,y coordinates
        shift_x (float): shift in x
        shift_y (float): shift in y
        angle (float): angle in degrees
        center (2-tuple): center of rotation
    '''
    shift = np.array([shift_x,shift_y])
    xy_shifted = xy - center[:,np.newaxis]
    xy_shifted_rotated = np.array([
        xy_shifted[0]*np.cos(np.deg2rad(angle)) - xy_shifted[1]*np.sin(np.deg2rad(angle)),
        xy_shifted[0]*np.sin(np.deg2rad(angle)) + xy_shifted[1]*np.cos(np.deg2rad(angle))
    ])
    xy_shifted_rotated_shifted = xy_shifted_rotated + center[:,np.newaxis] + shift[:,np.newaxis]
    return xy_shifted_rotated_shifted