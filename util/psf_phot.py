import numpy as np
import matplotlib.pyplot as plt
from util.sphot2 import astroplot

from scipy.ndimage import gaussian_filter
from astropy.nddata import overlap_slices
from astropy.table import QTable
from astropy.stats import sigma_clip

from photutils.aperture import CircularAperture
from photutils.psf import (SourceGrouper, IterativePSFPhotometry, 
                           PSFPhotometry, FittableImageModel)
from photutils.detection import DAOStarFinder
from photutils.background import LocalBackground, MMMBackground, MADStdBackgroundRMS

def make_modelimg(fit_models,shape,psf_shape):
    ''' modified version of photutil's function.
    No background is added.
    Args:
        fit_models: list of PSF models
    Returns:
        model_img: rendered model image    
    '''
    model_img = np.zeros(shape)
    for fit_model in fit_models:
        x0 = getattr(fit_model, 'x_0').value
        y0 = getattr(fit_model, 'y_0').value
        try:
            slc_lg, _ = overlap_slices(shape, psf_shape, (y0, x0),
                                        mode='trim')
        except Exception:
            continue
        yy, xx = np.mgrid[slc_lg]
        model_img[slc_lg] += fit_model(xx, yy)
    return model_img

def do_psf_photometry(data,psfimg,sigma_psf,psf_oversample,psf_blurring,
                      th=2,Niter=3,fit_shape=(3,3),render_shape=(25,25),plot=True):
    """ Performs PSF photometry.
    Args:
        data (2d array): the data to perform PSF photometry.
        psfimg (2d array): the PSF image.
        sigma_psf (float): the HWHM of the PSF. Use FWHM/2
        psf_oversample (int): the oversampling factor of the PSF.
        psf_blurring (float): the amount of Gaussian blurring to apply to the PSF.
        th (float): the detection threshold in background STD.
        Niter (int): the number of iterations to repeat the photometry (after cleaning up the data).
        fit_shape (2-tuple): the shape of the fit.
        render_shape (2-tuple): the shape of each PSF to be rendered.

    Returns:
        phot_result (QTable): the photometry result.
        model_img (2d array): the model image.
        resid (2d array): the residual image.
    """
    # tools
    bkgrms = MADStdBackgroundRMS()
    mmm_bkg = MMMBackground()

    # PSF
    blurred_psf = gaussian_filter(psfimg, psf_blurring)
    psf_model = FittableImageModel(blurred_psf, flux=1.0, x_0=0, y_0=0, 
                                   oversampling=psf_oversample, fill_value=0.0)

    # take data stats & prepare background-subtracted data
    bkg_level = mmm_bkg(data)
    data_bksub = data - bkg_level
    bkg_std = bkgrms(data_bksub)
    error = np.ones_like(data_bksub) * bkg_std

    # more tools
    daofinder = DAOStarFinder(threshold=th*bkg_std, fwhm=sigma_psf*2, 
                            roundhi=1.0, roundlo=-1.0,
                            sharplo=0.30, sharphi=1.40)
    localbkg_estimator = LocalBackground(2*sigma_psf, 5*sigma_psf, mmm_bkg)
    grouper = SourceGrouper(min_separation=3.0 * sigma_psf) # nearby sources to be fit simultaneously

    # run phootmetry
    psf_iter = IterativePSFPhotometry(psf_model, fit_shape, finder=daofinder,
                                    mode='new',grouper=grouper,
                                    localbkg_estimator=localbkg_estimator,
                                    aperture_radius=3)
    phot_result = psf_iter(data_bksub, error=error)

    for _ in range(Niter):
        s = phot_result['flux_fit'] > 0
        s = s & (phot_result['flags'] <= 1)
        init_params = QTable()
        init_params['x'] = phot_result['x_fit'].value[s]
        init_params['y'] = phot_result['y_fit'].value[s]
        phot_result = psf_iter(data_bksub, error=error, init_params=init_params)

    # final run
    s = phot_result['flux_fit'] > 0
    s = s & (phot_result['flags'] <= 1)
    init_params = QTable()
    init_params['x'] = phot_result['x_fit'].value[s]
    init_params['y'] = phot_result['y_fit'].value[s]
    psfphot = PSFPhotometry(psf_model, fit_shape, finder=daofinder, 
                            localbkg_estimator=localbkg_estimator,
                            grouper=grouper,
                            aperture_radius=3)
    phot_result = psfphot(data_bksub, error=error, init_params=init_params)

    # results
    # Remove flagged PSFs
    fit_models = np.asarray(psfphot._fit_models)
    s = phot_result['flags'] <= 1
    model_img = make_modelimg(fit_models[s],shape=data.shape,
                            psf_shape=(25,25))
    resid = data_bksub - model_img

    if plot:
        fig,axes = plt.subplots(1,3,figsize=(15,5))
        norm,offset = astroplot(data_bksub,ax=axes[0],percentiles=[0.1,99.9])
        astroplot(model_img,ax=axes[1],norm=norm,offset=offset)
        astroplot(resid,ax=axes[2],norm=norm,offset=offset)
        
    return phot_result[s], model_img, resid

# sigma-clip pixels outside r_eff
def sigma_clip_outside_aperture(data,r_eff,clip_sigma=4,
                                aper_size_in_r_eff=1,plot=True):
    mask = sigma_clip(data,sigma=clip_sigma).mask
    aperture = CircularAperture((data.shape[0]/2,data.shape[1]/2),
                                r_eff*aper_size_in_r_eff)
    aperture_mask = aperture.to_mask(method='center')
    aperture_mask_img = aperture_mask.to_image(data.shape).astype(bool)
    mask[aperture_mask_img] = False
    return mask # bad pixels are True