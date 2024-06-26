'''run_sphot.py
Y.S.Murakami 2024 @ JHU
Run the basic sphot pipeline on a single galaxy.
This script automatically detects the running environment (e.g., Slurm array job or local machine) and switches the logging output accordingly.

Usage:
    python run_sphot.py data_file.h5  [--out_folder=foldername] # run sphot on a new data file
    python run_sphot.py sphot_file.h5 --scalefit_only # run scalefit on existing sphot file if necessary 
    python run_sphot.py sphot_file.h5 --scalefit_rerun # rerun all scalefit on existing sphot file

'''



from sphot.utils import load_and_crop
from sphot.core import run_basefit, run_scalefit, logger
from sphot.data import read_sphot_h5

import sys
import os 

if __name__ == '__main__':
  
    # default options
    scalefit_only = False
    scalefit_rerun = False
    out_folder = './'

    datafile = sys.argv[1]
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg == '--scalefit_only':
                print('scalefit only option detected')
                scalefit_only = True
            if arg == '--scalefit_rerun':
                print('rerun option detected')
                scalefit_only = True
                scalefit_rerun = True
            elif arg.startswith('--out_folder'):
                out_folder = arg.split('=')[1]
                print('output folder specified:',out_folder)
            else:
                print('unknown options:',arg)     
                
    # switch logging option based on how this file is running
    if "SLURM_JOB_ID" in os.environ:
        from rich.console import Console
        def console_wrapper(func,*args,**kwargs):
            slurm_jobid = os.environ.get("SLURM_ARRAY_JOB_ID")
            slurm_taskid = os.environ.get("SLURM_ARRAY_TASK_ID")
            logfile = f'logs/{slurm_jobid}_{slurm_taskid}.log'
            logger.info(f"Running in Slurm (jobid={slurm_jobid}, taskid={slurm_taskid})")
            logger.info(f'Saving the progress in the log file: {logfile}')
            print(f'Saving the progress in the log file: {logfile}',flush=True)
            with open(logfile, 'w') as log_file:
                # Create a Console instance that writes to the log file
                console = Console(file=log_file, force_terminal=True, force_interactive=True)   
                console.print('test: this should be written to the log file '+logfile)
                kwargs.update(dict(console=console))
                return func(*args,**kwargs)
    else:
        def console_wrapper(func,*args,**kwargs):
            return func(*args,**kwargs)
            
    def run_sphot(**kwargs):
        ''' main commands are put in this dummy function so that the rich output can be forwarded to a log file when running in slurm'''
        filters =  ['F555W','F814W','F090W','F150W','F160W','F277W']
        folder_PSF = 'PSF/'   # a folder that contains filtername.npy (which stores PSF 2D array)
        base_filter = 'F150W' # the name of filter to which the Sersic model is fitted
        blur_psf = dict(zip(filters,[4,5,3.8,3.8,9,9])) # the sigma of PSF blurring in pixels
        iter_basefit = 10  # Number of iterative Sersic-PSF fitting for the base_filter fit
        iter_scalefit = 5 # Number of iterative Sersic-PSF fitting for scale-fit
        fit_complex_model = False # two-Sersic if True, single-Sersic if False
        allow_refit = False # Sersic profile is re-fitted for each filter if True
        custom_initial_crop = 1 # a float between 0 and 1: make this number smaller to manually crop data before analysis
        sigma_guess = 10 # initial guess of galaxy size in pixels (~HWHM of the galaxy profile)

        # 1. load data
        if not scalefit_only:
            galaxy = load_and_crop(datafile,filters,folder_PSF,
                                base_filter = base_filter,
                                plot = False,
                                custom_initial_crop = custom_initial_crop,
                                sigma_guess = sigma_guess)
            out_path = os.path.join(out_folder,f'{galaxy.name}_sphot.h5')
            logger.info(f'Galaxy data loaded: sphot file will be saved as {out_path}')

            # 2. fit Sersic model using the base filter
            run_basefit(galaxy,
                        base_filter = base_filter,
                        fit_complex_model = fit_complex_model,
                        blur_psf = blur_psf[base_filter],
                        N_mainloop_iter = iter_scalefit,
                        **kwargs)
            galaxy.save(out_path)
        else:
            logger.info(f'Loading an existing sphot filt: {datafile}')
            galaxy = read_sphot_h5(datafile)
            out_path=datafile
        
        # 3. Scale Sersic model (if necessary)
        logger.info('Starting Scale fit')
        base_params = galaxy.images[base_filter].sersic_params
        for filt in filters:
            if hasattr(galaxy.images[filt],'psf_sub_data') and not scalefit_rerun:
                # skip scalefit if data exists and not scalefit_rerun==True
                logger.info(f'Filter {filt} already has PSF-subtracted data')
                continue
            try:
                run_scalefit(galaxy,filt,base_params,
                            allow_refit=allow_refit,
                            fit_complex_model=fit_complex_model,
                            N_mainloop_iter=7,
                            blur_psf=blur_psf[filt],
                            **kwargs)
                galaxy.save(out_path)
            except Exception as e:
                logger.info(f'Filter {filt} failed:\n',e)
                continue
        logger.info('Completed Sphot')
        
    console_wrapper(run_sphot)
