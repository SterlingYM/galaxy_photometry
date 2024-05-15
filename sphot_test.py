folder_PSF = 'PSF/'
data_folder = 'cutouts_DDU/'
filters = ['F555W','F814W','F090W','F150W','F160W','F277W']
base_filter = 'F150W'
N_mainloop_iter = 7
fit_complex_model = True
blur_psf = 4


from sphot.utils import load_and_crop, prep_model
from sphot.psf import PSFFitter
from sphot.fitting import ModelFitter, ModelScaleFitter
from sphot.plotting import plot_sphot_results

import multiprocessing as mp
import glob
from warnings import filterwarnings

def run_fit(datafile):
    ######## base fit ########
    # 1. load data & perform initial analysis
    print('Preparing data...', datafile)
    galaxy = load_and_crop(datafile,filters,folder_PSF,
                        base_filter,plot=False)

    # 2. select base filter to fit
    cutoutdata = galaxy.images[base_filter]
    cutoutdata.perform_bkg_stats()
    # cutoutdata.determine_psf_blurring()
    cutoutdata.blur_psf(blur_psf)

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

    # 4. fit 
    print(f'Fitting the base filter {base_filter}...')
    fitter_1.fit(fit_to='data',max_iter=20)
    fitter_psf.fit(fit_to='sersic_residual',plot=False)
    fitter_2.fit(fit_to='psf_sub_data',
                method='iterative_NM',max_iter=30)
    for _ in range(N_mainloop_iter):
        fitter_2.fit(fit_to='psf_sub_data',method='iterative_NM',max_iter=15)
        fitter_psf.fit(fit_to='sersic_residual',plot=False)
        
    # 7. plot the results
    galaxy.save(f'{galaxy.name}_sphot.h5')

    ######## fit each filter ########
    base_params = cutoutdata.sersic_params
    for filt in galaxy.filters:
        print(f'\n*** working on {filt} ***')
        _cutoutdata = galaxy.images[filt]
        
        # basic statistics
        _cutoutdata.perform_bkg_stats()
        # _cutoutdata.determine_psf_blurring()
        _cutoutdata.blur_psf(4)

        # initialize model and fitters
        if fit_complex_model:
            _model = prep_model(_cutoutdata,simple=False)
        else:
            _model = prep_model(_cutoutdata,simple=True)
        _fitter_scale = ModelScaleFitter(_model,_cutoutdata,base_params)
        _fitter_2 = ModelFitter(_model,_cutoutdata)
        _fitter_psf = PSFFitter(_cutoutdata)
        
        # run fitting
        _fitter_scale.fit(fit_to='data')
        _fitter_psf.fit(fit_to='sersic_residual',plot=False)
        _fitter_2.model.x0 = _cutoutdata.sersic_params
        _fitter_2.fit(fit_to='psf_sub_data',max_iter=20)
        for _ in range(N_mainloop_iter):
            _fitter_2.fit(fit_to='psf_sub_data',max_iter=10)
            _fitter_psf.fit(fit_to='sersic_residual',plot=False)
        
    # update the galaxy save data
    galaxy.save(f'{galaxy.name}_sphot.h5')
    
def run_fit_helper(datafile):
    try:
        run_fit(datafile)
    except Exception as e:
        print(f'Error in {datafile}: {e}')
        
if __name__ == '__main__':
    filterwarnings('ignore')
    datafiles = glob.glob(data_folder+'g*.h5')
    print('Working on the following files:',datafiles)
    
    with mp.Pool(mp.cpu_count()) as pool:
        _ = list(pool.imap_unordered(run_fit_helper,datafiles))