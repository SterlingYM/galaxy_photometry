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
from scipy.optimize import leastsq

from util.sphot_workflow import plot_image_logscale
from util.sphot2 import astroplot


def read_fits(filename):
    class Object(object):
        pass
    hdul = fits.open(filename)
    keys = [hdu.name for hdu in hdul]

    # select HDU
    hdu_name = 'SCI' if 'SCI' in keys else 'PRIMARY'
    
    # read in data
    data = hdul[hdu_name].data
    header = hdul[hdu_name].header
    wcs = WCS(header)
    hdul.close()
    
    # save to object
    return_obj = Object()
    return_obj.data = data
    return_obj.header = header
    return_obj.wcs = wcs
    return return_obj

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

def generate_cutout(images,filternames,wcs,center_coord,cutout_size,
                    offset_min,offset_max,maxiter=15,plot=True):
    ''' keep randomly generating cutouts until a valid one is found
    Inputs:
        images (list): list of 2D arrays (images)
        filternames (list): list of strings representing filter names
        wcs (WCS): astropy WCS of images (assumed to be the same for all images)
        center_coord (SkyCoord): astropy SkyCoord
        cutout_size (tuple): (Nrows,Ncols)
        offset_min (float): minimum offset in arcmin
        offset_max (float): maximum offset in arcmin
        maxiter (int): maximum number of iterations to try before giving up
    '''
    success = False
    for Niter in range(maxiter):
        try:
            offset_angle = np.random.uniform(0, 360) * u.deg
            offset_separation = np.random.uniform(offset_min, offset_max) * u.arcmin
            cutout_loc = center_coord.directional_offset_by(offset_angle, offset_separation)
            cutouts = []
            for img,filtername in zip(images,filternames):
                cutout = Cutout2D(img, cutout_loc, cutout_size, wcs=wcs, mode='strict')
                assert (~np.isfinite(cutout.data)).sum() == 0 #<= 0.05 * cutout_size[0] * cutout_size[1]
                cutouts.append(cutout)
            success = True
            break
        except Exception:
            if Niter+1 == maxiter:
                print(f'{maxiter} iterations failed to produce a valid cutout. Exiting.')
                success = False
                cutouts = None
            continue

    if success and plot:
        fig,axes = plt.subplots(2,3,figsize=(15,10))
        for cutout,ax,filtername in zip(cutouts,axes.ravel(),filternames):
            plot_image_logscale(cutout.data,ax=ax)
            ax.set_title(filtername) 

    return cutouts

def plot_offset(image_ref,ref_x,ref_y,tgt_x,tgt_y,filter_ref,filter_tgt):
    fig,(ax1,ax3) = plt.subplots(1,2,figsize=(14,7))

    astroplot(image_ref,ax=ax1,cmap='gray_r',set_bad='w')
    ax1.scatter(ref_x,ref_y,s=20,facecolor='none',edgecolor='r')
    # ax2.scatter(tgt_x,tgt_y,s=30,facecolor='none',edgecolor='b')
    ax3.scatter(tgt_x-ref_x,tgt_y-ref_y,s=5,c='k')

    if abs(np.mean(tgt_x-ref_x)) < 1.5:
        ax3.set_xlim(-1.5,1.5)
    if abs(np.mean(tgt_y-ref_y)) < 1.5:
        ax3.set_ylim(-1.5,1.5)
    for val in [-1,0,1]:
        ax3.axhline(val,c='yellowgreen',ls='--')
        ax3.axvline(val,c='yellowgreen',ls='--')
    ax3.set_xlabel('x offset (pixels)',fontsize=13)
    ax3.set_ylabel('y offset (pixels)',fontsize=13)
    ax3.tick_params(direction='in')
    ax3.text(x=0.05,y=0.95,
            s=f'$\Delta x$ = {np.mean(tgt_x-ref_x):.2f} $\pm$ {np.std(tgt_x-ref_x):.2f} pixels\n$'+\
            f'\Delta y$ = {np.mean(tgt_y-ref_y):.2f} $\pm$ {np.std(tgt_y-ref_y):.2f} pixels',
            ha='left',va='top',transform=ax3.transAxes,fontsize=13)
    fig.suptitle(f'Offset of {filter_tgt} sources w.r.t. matched sources in {filter_ref}',fontsize=15)
    
def apply_sigma_clip(data,sigma):
    from astropy.stats import sigma_clip
    clip = sigma_clip(data,sigma=sigma)
    return data[~clip.mask], ~clip.mask

def update_wcs(original_file,header_to_use,ref_x,ref_y,tgt_x,tgt_y,wcs_reprojected,pix_scale):
    ''' take in the wcs to be corrected (original file) and update the wcs based on the matched sources'''
    # set rotation center to the CRVAL locations of the *ORIGINAL* images (before reprojected)
    hdul = fits.open(original_file)
    header_original = hdul[header_to_use].header
    wcs_original = WCS(header_original)
    wcs_original = WCS(wcs_original.to_header()) # auto-run the default correction (e.g., use PC instead of CD)
    center_ra = hdul[header_to_use].header['CRVAL1']
    center_dec = hdul[header_to_use].header['CRVAL2']
    center_coord = SkyCoord(ra=center_ra*u.deg,dec=center_dec*u.deg)
    center_x,center_y = skycoord_to_pixel(center_coord,wcs_reprojected)

    # fit shift and rotation
    def residual(params): 
        shift_x,shift_y,angle = params
        new_xy = shift_and_rotate(np.array([tgt_x,tgt_y]),shift_x,shift_y,angle,np.array([center_x,center_y]))
        # chi2 = np.sum(**2)
        return np.ravel(new_xy - np.array([ref_x,ref_y]))

    # print results
    paramnames = ['shift_x','shift_y','rotation']
    units = ['pix','pix','deg']
    initial_guess = [0,0,0]
    bounds = [[-50,50],[-50,50],[-180,180]]
    pfit, pcov, _, _, _ = leastsq(residual,initial_guess,full_output=True)
    for paramname,param,err,unit in zip(paramnames,pfit,np.sqrt(np.diag(pcov)),units):
        print(f'{paramname:10}: {param:8.4f} +/- {err:.4f} {unit}')
        
    # update wcs
    wcs_updated = wcs_original.deepcopy()
    delta_ra = (pfit[0] * u.pixel * pix_scale).to(u.deg)
    delta_dec = (pfit[1] * u.pixel * pix_scale).to(u.deg)
    th = np.deg2rad(-1*pfit[2])
    rotation = np.array([[np.cos(th), -np.sin(th)],
                        [np.sin(th), np.cos(th)]]) 
    wcs_updated.wcs.crval = [((center_ra * u.deg) - delta_ra).value,
                            ((center_dec * u.deg) + delta_dec).value]
    wcs_updated.wcs.pc = rotation.dot(wcs_original.wcs.pc)
    return wcs_updated