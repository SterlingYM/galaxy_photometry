from sphot.utils import load_and_crop
from sphot.core import run_basefit, run_scalefit
import sys
import os 

if __name__ == '__main__':
    datafile = sys.argv[1]
    if len(sys.argv) > 2:
        out_folder = sys.argv[2]
    else:
        out_folder = './'
    filters =  ['F555W','F814W','F090W','F150W','F160W','F277W']
    folder_PSF = 'PSF/'
    base_filter = 'F150W'
    blur_psf = dict(zip(filters,[5,5,3.8,3.8,9,9]))
    iter_basefit = 7
    iter_scalefit = 5
    fit_complex_model = True
    allow_refit = False
    custom_initial_crop = 1 # a float between 0 and 1: make this number smaller to manually crop data before analysis
    sigma_guess = 10 # initial guess of galaxy size in pixels

    # 1. load data
    galaxy = load_and_crop(datafile,filters,folder_PSF,
                           base_filter = base_filter,
                           plot = False,
                           custom_initial_crop = custom_initial_crop,
                           sigma_guess = sigma_guess)
    out_path = os.path.join(out_folder,f'{galaxy.name}_sphot.h5')
    print('Galaxy data loaded: sphot file will be saved as',out_path)

    # 2. fit Sersic model using the base filter
    run_basefit(galaxy,
                base_filter = base_filter,
                fit_complex_model = fit_complex_model,
                blur_psf = blur_psf,
                N_mainloop_iter = iter_scalefit)
    galaxy.save(out_path)
    
    # 3. Scale Sersic model
    base_params = galaxy.images[base_filter].sersic_params
    for filt in filters:
        try:
            run_scalefit(galaxy,filt,base_params,
                        allow_refit=allow_refit,
                        fit_complex_model=fit_complex_model,
                        N_mainloop_iter=7,
                        blur_psf=blur_psf)
            galaxy.save(out_path)
        except Exception:
            pass

    print('Completed Sphot')