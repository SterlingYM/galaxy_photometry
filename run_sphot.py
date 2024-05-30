from sphot.utils import load_and_crop
from sphot.core import run_basefit, run_scalefit
from sphot.data import read_sphot_h5

import sys
import os 

if __name__ == '__main__':
    scalefit_only = False
    
    datafile = sys.argv[1]
    if len(sys.argv) > 2:
        out_folder = sys.argv[2]
    else:
        out_folder = './'
        
    if len(sys.argv) > 3:
        for arg in sys.argv[3:]:
            if arg.startswith('--scalefit_only'):
                scalefit_only = True

    filters =  ['F555W','F814W','F090W','F150W','F160W','F277W']
    folder_PSF = 'PSF/'   # a folder that contains filtername.npy (which stores PSF 2D array)
    base_filter = 'F150W' # the name of filter to which the Sersic model is fitted
    blur_psf = dict(zip(filters,[5,5,3.8,3.8,9,9])) # the sigma of PSF blurring in pixels
    iter_basefit = 7  # Number of iterative Sersic-PSF fitting for the base_filter fit
    iter_scalefit = 5 # Number of iterative Sersic-PSF fitting for scale-fit
    fit_complex_model = True # two-Sersic if True, single-Sersic if False
    allow_refit = False # Sersic profile is re-fitted for each filter if True
    custom_initial_crop = 1 # a float between 0 and 1: make this number smaller to manually crop data before analysis
    sigma_guess = 10 # initial guess of galaxy size in pixels

    # 1. load data
    if not scalefit_only:
        galaxy = load_and_crop(datafile,filters,folder_PSF,
                            base_filter = base_filter,
                            plot = False,
                            custom_initial_crop = custom_initial_crop,
                            sigma_guess = sigma_guess)
        out_path = os.path.join(out_folder,f'{galaxy.name}_sphot.h5')
        print('* Galaxy data loaded: sphot file will be saved as',out_path)

        # 2. fit Sersic model using the base filter
        run_basefit(galaxy,
                    base_filter = base_filter,
                    fit_complex_model = fit_complex_model,
                    blur_psf = blur_psf[base_filter],
                    N_mainloop_iter = iter_scalefit)
        galaxy.save(out_path)
    else:
        print('* Loading an existing sphot filt:',datafile)
        galaxy = read_sphot_h5(datafile)
        out_path=datafile
    
    # 3. Scale Sersic model
    print('* Starting Scale fit')
    base_params = galaxy.images[base_filter].sersic_params
    for filt in filters:
        try:
            run_scalefit(galaxy,filt,base_params,
                        allow_refit=allow_refit,
                        fit_complex_model=fit_complex_model,
                        N_mainloop_iter=7,
                        blur_psf=blur_psf[filt])
            galaxy.save(out_path)
        except Exception:
            pass

    print('Completed Sphot')