import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import glob
import os

import astropy.units as u
from astropy.io import fits

from util.sphot2 import CutoutData, MultiBandCutout, astroplot, calc_mag
import logging
import multiprocessing as mp

#### TODO: code below is just a copy from test_SersicFit as a skeleton

# --------------------------------------------
# setups
# --------------------------------------------
folder_PSF = 'PSF/'
folder_galaxy = 'cutouts_ceers/'
folder_foreground = 'cutouts_raw/'
filters = ['F555W','F814W','F090W','F150W','F160W','F277W']
galaxy_catalog = pd.read_csv('cutouts_ceers/catalog.csv')

# logger
logger = logging.getLogger('sphot2')
logger.setLevel(logging.INFO)


def run_test(galaxy_catalog,
             folder_PSF,folder_galaxy,folder_foreground,
             filters,
             galaxy_idx=None,foreground_ID=None,
             logger=None,plot=False,test_ID_prefix='',
             random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
        
    if logger is None:
        logger = logging.getLogger('sphot2')
        logger.setLevel(logging.INFO)
    
    # pick a galaxy
    if galaxy_idx is None:
        logger.info('galaxy_idx is not specied: picking a random galaxy')
        catalog_idx = int(np.random.uniform(0,len(galaxy_catalog)))
    elif galaxy_idx == 'randomize':
        logger.info('Picking a random galaxy')
        catalog_idx = int(np.random.uniform(0,len(galaxy_catalog)))
    else:
        catalog_idx = galaxy_idx
        logger.info(f'using galaxy_idx {catalog_idx}')
    galaxy_ID = galaxy_catalog.loc[catalog_idx, 'ID']
    CEERS_field = galaxy_catalog.loc[catalog_idx, 'FIELD']
    
    # pick foreground
    galaxy_only = False
    if foreground_ID is None:
        logger.info('foreground_ID is not specied: no foreground will be used. Use \'randomize\' to randomly select a foreground.')
        galaxy_only = True
        foreground_ID = 'GalaxyOnly'
    elif foreground_ID == 'randomize':
        logger.info('Picking a random foreground')
        foreground_ID = str(int(np.random.uniform(0,500))).zfill(3)
    else:
        logger.info('Using a specified foreground ID')
        pass
    # foreground_ID = 'GalaxyOnly' if galaxy_only else int(np.random.uniform(0,500))

    test_ID = test_ID_prefix + f'{galaxy_ID}_{foreground_ID}'
    logger.info(f'test_ID: {test_ID}')
    
    # --------------------------------------------
    # load data
    # --------------------------------------------

    # load PSFs
    logger.info('loading PSFs')
    psfs_data = []
    for filtername in filters:
        path = glob.glob(folder_PSF + f'*{filtername}_PSF*.npy')[0]
        psfs_data.append(np.load(path))#
    PSFs_dict = dict(zip(filters, psfs_data))

    # load galaxy data
    logger.info('loading galaxy data')
    with h5py.File(folder_galaxy+f'Field{CEERS_field}_bksub.h5', 'r') as fl:
        galaxy = fl[str(galaxy_ID)]
        galaxy_imgs = dict(zip(filters, 
                            [galaxy[f][:] for f in filters]))
        galaxy_data = dict(galaxy.attrs)

    # load foreground data
    if galaxy_only:
        logger.info('fitting galaxy only -- skipping foreground data prep')
        foreground_imgs = dict(zip(filters, 
                            [np.ones_like(galaxy_imgs[f])*0 for f in filters]))
    else:
        logger.info('loading foreground data')
        with h5py.File(folder_foreground + f'{foreground_ID}.h5', 'r') as fl:
            foreground_imgs = dict(zip(filters, 
                                [fl[f][:] for f in filters]))

    # --------------------------------------------
    # generate MultiBandCutout object
    # --------------------------------------------
    logger.info('generating MultiBandCutout object')
    galaxy = MultiBandCutout(name = galaxy_ID,
                            catalog_data = galaxy_data)
    for filtername in filters:
        simimage = galaxy_imgs[filtername] + foreground_imgs[filtername]
        psf = PSFs_dict[filtername]
        cutoutdata = CutoutData(data = simimage, 
                                psf = psf,
                                psf_oversample = 4,
                                filtername = filtername)
        galaxy.add_image(filtername, cutoutdata)

    # plot
    if plot:
        fig,axes = plt.subplots(1,3,figsize=(15,5))
        astroplot(galaxy_imgs['F150W'],ax=axes[0])
        astroplot(foreground_imgs['F150W'],ax=axes[1])
        astroplot(galaxy.F150W.data,ax=axes[2])
        plt.show()

    # --------------------------------------------
    # do photometry
    # --------------------------------------------
    logger.info('starting photometry')
    # pick fitting precision
    fitting_kwargs = dict(mag_tol = 0.001, 
                          rtol_init = 1e-3, 
                          rtol_iter = 1e-3)

    # fit reference image
    galaxy.F150W.do_photometry(plot=plot,**fitting_kwargs)

    # fix r_eff, n, x_0, y_0, ellip, theta
    fixed_params_names = ['r_eff', 'n', 'x_0', 'y_0', 'ellip', 'theta']
    fixed_params_values = [galaxy.F150W.fitter.bestfit[p] for p in fixed_params_names]
    fixed_params_dict = dict(zip(fixed_params_names,fixed_params_values))

    # fit all images
    galaxy.F555W.do_photometry(plot=plot,fixed_params = fixed_params_dict, **fitting_kwargs)
    galaxy.F814W.do_photometry(plot=plot,fixed_params = fixed_params_dict, **fitting_kwargs)
    galaxy.F090W.do_photometry(plot=plot,fixed_params = fixed_params_dict, **fitting_kwargs)
    galaxy.F160W.do_photometry(plot=plot,fixed_params = fixed_params_dict, **fitting_kwargs)
    galaxy.F277W.do_photometry(plot=plot,fixed_params = fixed_params_dict, **fitting_kwargs)

    # --------------------------------------------
    dAB_F555W = galaxy.F555W.mag - galaxy.catalog_data['AB606']
    dAB_F814W = galaxy.F814W.mag - galaxy.catalog_data['AB814']
    dAB_F090W = galaxy.F090W.mag - galaxy.catalog_data['AB115']
    dAB_F150W = galaxy.F150W.mag - galaxy.catalog_data['AB150']
    dAB_F160W = galaxy.F160W.mag - galaxy.catalog_data['AB160']
    dAB_F277W = galaxy.F277W.mag - galaxy.catalog_data['AB277']
    logger.info(f'F555W AB={galaxy.F555W.mag}, ΔAB={dAB_F555W}')
    logger.info(f'F814W AB={galaxy.F814W.mag}, ΔAB={dAB_F814W}')
    logger.info(f'F090W AB={galaxy.F090W.mag}, ΔAB={dAB_F090W}')
    logger.info(f'F150W AB={galaxy.F150W.mag}, ΔAB={dAB_F150W}')
    logger.info(f'F160W AB={galaxy.F160W.mag}, ΔAB={dAB_F160W}')
    logger.info(f'F277W AB={galaxy.F277W.mag}, ΔAB={dAB_F277W}')
    
    cols = ['test_ID',
            *[f'sphot_{filt}' for filt in filters],
            *[f'dAB_{filt}' for filt in filters],
            *[f'bkg_mean_{filt}' for filt in filters],
            *[f'bkg_std_{filt}' for filt in filters],
            *fixed_params_names]
    
    vals = [test_ID, 
            *[galaxy.images[filt].mag for filt in filters], 
            dAB_F555W, dAB_F814W, dAB_F090W, dAB_F150W, dAB_F160W, dAB_F277W, 
            *[galaxy.images[filt].bkg_mean for filt in filters],
            *[galaxy.images[filt].bkg_std for filt in filters],
            *fixed_params_values]
    result = pd.Series(dict(zip(cols,vals)))
    return result