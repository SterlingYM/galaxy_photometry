# helpers to perform interpolated aperture photometry.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,Normalize
from scipy.interpolate import RegularGridInterpolator

from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.aperture import SkyCircularAperture, CircularAperture
from photutils.utils._wcs_helpers import _pixel_scale_angle_at_skycoord


def interp_aperture(image,wcs,pos,theta,
                    cutout_radius_factor = 10,
                    upscale_factor = 50,
                    ):
    """
    Interpolates input image and performs aperture photometry.
    inputs:
        image: 2D numpy array of [x,y]. Prepare with image = hdu.data.
        wcs: WCS object. Prepare with wcs = WCS(hdu.header).
        pos: SkyCoord object
        theta: aperture radius (in u.arcsec)
        cutout_radius_factor: the relative size of initial aperture to the target aperture
        upscale_factor: the factor of interpolated (upscaled) pixel resolution
    """
    # pixel-world coordinate transformations
    centers,angular_resolution,_ = _pixel_scale_angle_at_skycoord(pos, wcs)

    # raw aperture photometry (before interpolation)
    aperture_sky = SkyCircularAperture(pos, (theta).to(u.arcsec))
    aperture = aperture_sky.to_pixel(wcs)
    local_flux = aperture.to_mask().multiply(image)
    
    # dummy aperture to get the cutout coordinates
    aperture_cutout_sky = SkyCircularAperture(pos, (theta*cutout_radius_factor).to(u.arcsec))
    aperture_cutout = aperture_cutout_sky.to_pixel(wcs)
    
    # cutout the area near aperture
    extent = aperture_cutout._bbox[0].extent
    ixmin = aperture_cutout._bbox[0].ixmin
    ixmax = aperture_cutout._bbox[0].ixmax
    iymin = aperture_cutout._bbox[0].iymin
    iymax = aperture_cutout._bbox[0].iymax
    image_cutout = image[iymin:iymax,ixmin:ixmax]

    # old pixel coordinates
    x_range = np.arange(ixmin,ixmax)
    y_range = np.arange(iymin,iymax)

    # new pixel coordinates
    x_new = np.linspace(ixmin,ixmax-1,len(x_range)*upscale_factor)
    y_new = np.linspace(iymin,iymax-1,len(y_range)*upscale_factor)
    xx_new,yy_new = np.meshgrid(x_new,y_new,indexing='xy')

    # perform interpolation
    interp = RegularGridInterpolator((y_range,x_range),
                                     image_cutout,
                                     method='linear')
    image_cutout_interp = interp(np.array([yy_new,xx_new]).T).T
    
    # convert aperture pixel coordinates to new pixel coordinates
    center_x_new = (centers[0]-ixmin)*upscale_factor
    center_y_new = (centers[1]-iymin)*upscale_factor
    centers_new = [center_x_new,center_y_new]
    radius_new = (theta/angular_resolution*upscale_factor).value

    # apply aperture
    aperture_new = CircularAperture(centers_new,radius_new)
    local_flux_new = aperture_new.to_mask().multiply(image_cutout_interp)
    
    # conserve flux
    pixcounts = aperture.to_mask().data.sum() 
    pixcounts_new = aperture_new.to_mask().data.sum()
    flux_conservation_factor = pixcounts / pixcounts_new
    local_flux_new = local_flux_new * flux_conservation_factor
    
    images = {
        'original': image,
        'cutout':   image_cutout,
        'cutout_interp': image_cutout_interp
    }
    apertures = {
        'original': aperture,
        'cutout': aperture_cutout,
        'cutout_interp': aperture_new
    }
    flux = {
        'original': dict(data = local_flux,
                         total = local_flux.sum()),
        'interp': dict(data = local_flux_new,
                       total = local_flux_new.sum())
    }
    
    results = dict(wcs = wcs,
                   pos = pos,
                   theta = theta,
                   images = images,
                   apertures = apertures,
                   flux = flux,
                   flux_conservation_factor = flux_conservation_factor)
    
    
    return results

def plot_image(img,ax=None,vmin=1e-1,vmax=4,scale='log',**kwargs):
    img[img<=0] = vmin
    if scale=='log':
        norm = LogNorm(vmin=vmin, vmax=vmax,clip=True)
    if scale=='linear':
        norm = Normalize(vmin=vmin, vmax=vmax)
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(8,6))
    ax.imshow(img,origin='lower',norm=norm,**kwargs)
    
def plot_results(results):
    wcs = results['wcs']
    pos = results['pos']
    theta = results['theta']
    # cutout original image
    _aperture_cutout_sky = SkyCircularAperture(pos, (theta*100).to(u.arcsec))
    _aperture_cutout = _aperture_cutout_sky.to_pixel(wcs)
    ixmin = _aperture_cutout._bbox[0].ixmin
    ixmax = _aperture_cutout._bbox[0].ixmax
    iymin = _aperture_cutout._bbox[0].iymin
    iymax = _aperture_cutout._bbox[0].iymax
    _image_cutout = results['images']['original'][iymin:iymax,ixmin:ixmax]



    fig = plt.figure(figsize=(15,7))
    ax_orig = fig.add_axes([0.08,0.05,0.4,0.9])
    ax_cutout = fig.add_axes([0.5,0.53,0.15,0.4])
    ax_cutout_interp = fig.add_axes([0.68,0.53,0.15,0.4])
    ax_flux = fig.add_axes([0.5,0.1,0.15,0.4])
    ax_flux_interp = fig.add_axes([0.68,0.1,0.15,0.4])

    plot_image(_image_cutout,vmin=5e-2,ax=ax_orig,
               extent = _aperture_cutout._bbox[0].extent)

    results['apertures']['cutout']._bbox[0].extent
    plot_image(results['images']['cutout'],ax=ax_cutout,scale='linear',
              extent = results['apertures']['cutout']._bbox[0].extent)
    plot_image(results['images']['cutout_interp'],ax=ax_cutout_interp,scale='linear')
    plot_image(results['flux']['original']['data'],ax=ax_flux,scale='linear',
              extent = results['apertures']['original']._bbox[0].extent)
    plot_image(results['flux']['interp']['data']/results['flux_conservation_factor'],
               ax=ax_flux_interp,scale='linear',
              extent = results['apertures']['cutout_interp']._bbox[0].extent)

    # plot apertures
    results['apertures']['original'].plot(ax=ax_orig,lw=2,color='red')
    results['apertures']['original'].plot(ax=ax_cutout,lw=2,color='red')
    results['apertures']['original'].plot(ax=ax_flux,lw=2,color='red')
    results['apertures']['cutout_interp'].plot(ax=ax_flux_interp,lw=2,color='red')
    results['apertures']['cutout'].plot(ax=ax_flux,lw=2,color='red')

    ax_orig.indicate_inset_zoom(ax_cutout, edgecolor="k")
    ax_cutout.indicate_inset_zoom(ax_flux, edgecolor="orange")
    ax_cutout_interp.indicate_inset_zoom(ax_flux_interp, edgecolor="orange")

    ax_cutout.set_title('Raw data',fontsize=14)
    ax_cutout_interp.set_title('Interpolated data',fontsize=14)
    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.set_facecolor('w')
    results['apertures']['cutout_interp'].plot(ax=ax_cutout_interp,lw=2,color='red')
    
    return fig