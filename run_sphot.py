from sphot.utils import load_and_crop, prep_model
from sphot.psf import PSFFitter
from sphot.fitting import ModelFitter, ModelScaleFitter
from sphot.plotting import plot_sphot_results

import multiprocessing as mp
import glob
import sys
from warnings import filterwarnings

def run_sphot(datafile,filters,folder_PSF,out_folder,
              base_filter,blur_psf,
              N_mainloop_iter=7,
              fit_complex_model=False,allow_refit=False,):
    ######## base fit ########
    # 1. load data & perform initial analysis
    print('Preparing data...', datafile)
    galaxy = load_and_crop(datafile,filters,folder_PSF,
                           base_filter,plot=False,
                           custom_initial_crop=1,
                           sigma_guess=10)

    # 2. select base filter to fit
    cutoutdata = galaxy.images[base_filter]
    cutoutdata.perform_bkg_stats()
    cutoutdata.blur_psf(blur_psf[base_filter])

    # 3. make models & prepare fitters
    model_1 = prep_model(cutoutdata,simple=True)
    fitter_1 = ModelFitter(model_1,cutoutdata)
    if fit_complex_model:
        model_2 = prep_model(cutoutdata,simple=False)
        fitter_2 = ModelFitter(model_2,cutoutdata)
    else:
        model_2 = model_1
        fitter_2 = fitter_1
    fitter_psf = PSFFitter(cutoutdata)

    # 4. fit the profile
    print(f'Fitting the base filter {base_filter}...')
    fitter_1.fit(fit_to='data',max_iter=20)
    fitter_psf.fit(fit_to='sersic_residual',plot=False)
    fitter_2.fit(fit_to='psf_sub_data',
                method='iterative_NM',max_iter=30)
    for _ in range(N_mainloop_iter):
        fitter_2.fit(fit_to='psf_sub_data',method='iterative_NM',max_iter=15)
        fitter_psf.fit(fit_to='sersic_residual',plot=False)
    galaxy.save(out_folder+f'{galaxy.name}_sphot.h5')

    ######## fit each filter ########
    base_params = cutoutdata.sersic_params
    for filtername in galaxy.filters:
        print(f'\n*** working on {filtername} ***')
        _cutoutdata = galaxy.images[filtername]
        
        # basic statistics
        _cutoutdata.perform_bkg_stats()
        _cutoutdata.blur_psf(blur_psf[filtername])

        # initialize model and fitters
        if fit_complex_model:
            _model = prep_model(_cutoutdata,simple=False)
        else:
            _model = prep_model(_cutoutdata,simple=True)
        _fitter_scale = ModelScaleFitter(_model,_cutoutdata,base_params)
        if allow_refit:
            _fitter_2 = ModelFitter(_model,_cutoutdata)
        _fitter_psf = PSFFitter(_cutoutdata)
            
        # run fitting
        _fitter_scale.fit(fit_to='data')
        _fitter_psf.fit(fit_to='sersic_residual',plot=False)
        if allow_refit:
            _fitter_2.model.x0 = _cutoutdata.sersic_params
            _fitter_2.fit(fit_to='psf_sub_data',max_iter=20)
        for _ in range(N_mainloop_iter):
            if allow_refit:
                _fitter_2.fit(fit_to='psf_sub_data',max_iter=10)
            else:
                _fitter_scale.fit(fit_to='data')
            _fitter_psf.fit(fit_to='sersic_residual',plot=False)
            
        # update saved data
        galaxy.save(out_folder+f'{galaxy.name}_sphot.h5')
    
def run_helper(**kwargs):
    try:
        run_sphot(**kwargs)
    except Exception as e:
        datafile = kwargs['datafile']
        print(f'Error in {datafile}: {e}')
        
if __name__ == '__main__':
    filters = ['F555W','F814W','F090W','F150W','F160W','F277W']
    run_helper(
        datafile = sys.argv[1],
        filters = filters,
        folder_PSF = 'PSF/',
        out_folder = 'sphot_out_May23/',
        base_filter = 'F150W',
        blur_psf = dict(zip(filters,[5,5,3.8,3.8,9,9])),
        N_mainloop_iter = 7,
        fit_complex_model = False,
        allow_refit = False,
    )