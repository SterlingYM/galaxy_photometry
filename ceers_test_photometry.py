import os
import glob
import h5py
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sphot.data import (read_sphot_h5, load_h5data,
                        MultiBandCutout, CutoutData)
from sphot.aperture import IsoPhotApertures
from sphot.aperture import CutoutDataPhotometry


##################
# settings
##################
sphot_folder = 'sphot_out/Jun18_tests'
ceers_filename = 'cutouts_ceers/Field1_bksub.h5'
base_filter = 'F150W'
petro_val_at_aperture = 0.4
psf_oversample = 4
center_mask = 3
filters = ['F555W','F814W','F090W','F150W','F160W','F277W']
out_folder = 'test_results'
logger = logging.getLogger('sphot')

def get_original_galaxy(filename,galaxy_id,filters,PSFs_dict,
                        psf_oversample):
    ''''''
    galaxy = MultiBandCutout(name = galaxy_id)
    with h5py.File(filename,'r') as f:
        for filt in filters:
            psf = PSFs_dict[filt]
            image = f[galaxy_id][filt][:]
            cutoutdata = CutoutData(data = image, 
                                    psf = psf,
                                    psf_oversample = psf_oversample,
                                    filtername = filt)
            galaxy.add_image(filt, cutoutdata)
        f.close()
    return galaxy 

paths = glob.glob(os.path.join(sphot_folder,'*.h5'))
columns = ['galaxy_id',
           *[filt+'_mag_sphot' for filt in filters],
           *[filt+'_err_sphot' for filt in filters],
           *[filt+'_mag_ceers' for filt in filters],
           *[filt+'_err_ceers' for filt in filters],
              'aper_size']
df_results = pd.DataFrame()
for i,path in enumerate(paths):
    logger.info(f'*** Processing the galaxy {i+1}/{len(paths)}: {path} ***')
    
    try:
        # load Sphot processed data
        logger.info('loading the sphot-processed data...')
        galaxy_sphot = read_sphot_h5(path)
        pdf = PdfPages(os.path.join(out_folder,galaxy_sphot.name+'.pdf'))
        galaxy_sphot.plot()
        pdf.savefig(); plt.close()
        
        # load original CEERS cutout & crop in to the same size as Sphot cutouts
        logger.info('loading the original CEERS data...')
        PSFs_dict = dict(zip(galaxy_sphot.filters,
                            [c.psf for c in galaxy_sphot.image_list]))
        galaxy_ceers = get_original_galaxy(
            filename = ceers_filename,
            galaxy_id = galaxy_sphot.name.split('_')[0],
            filters = galaxy_sphot.filters,
            PSFs_dict = PSFs_dict,
            psf_oversample = psf_oversample)
        x0,y0,_ = galaxy_ceers.images[base_filter].init_size_guess()
        galaxy_ceers.crop_in(x0, y0, galaxy_sphot.images[base_filter].data.shape[0])
        galaxy_ceers.plot()
        pdf.savefig(); plt.close()

        # Determine aperture size
        logger.info('Determining the aperture size using the Sphot data...')
        iso_apers = IsoPhotApertures(galaxy_sphot.images[base_filter])
        iso_apers.create_apertures(fit_to='sersic_modelimg',
                                frac_enc=np.linspace(0.05,0.8,100))
        iso_apers.measure_flux(measure_on='psf_sub_data')
        iso_apers.calc_petrosian_indices(bin_size=2)
        iso_apers.plot(x_attr='semi_major_axes')
        aper_sci = iso_apers.get_aper_at(petro=petro_val_at_aperture)
        pdf.savefig(); plt.close()
        
        # Perform photometry on the sphot-processed data
        logger.info('Performing photometry on the sphot-processed data...')
        mags_sphot = {}
        mag_errors_sphot = {}
        for filt in galaxy_sphot.filters:
            try:
                cutoutdata = galaxy_sphot.images[filt]
                aperphot = CutoutDataPhotometry(cutoutdata,aper_sci)
                aperphot.measure_flux(measure_on='psf_sub_data')
                aperphot.measure_sky(measure_on='residual_masked',
                                    center_mask = center_mask,
                                    mode='grid')
                aperphot.calc_mag()
                aperphot.plot()
                mags_sphot[filt] = aperphot.magAB
                mag_errors_sphot[filt] = aperphot.magAB_err
                pdf.savefig(); plt.close()
            except Exception:
                mag_sphot[filt] = np.nan
                mag_errors_sphot[filt] = np.nan
        
        # Perform photometry on the sphot-processed data
        logger.info('Performing photometry on the original data...')
        mags_ceers = {}
        mag_errors_ceers = {}
        for filt in galaxy_ceers.filters:
            try:
                cutoutdata = galaxy_ceers.images[filt]
                aperphot = CutoutDataPhotometry(cutoutdata,aper_sci)
                aperphot.measure_flux(measure_on='_rawdata')
                aperphot.measure_sky(measure_on='_rawdata',
                                    center_mask = center_mask,
                                    mode='grid')
                aperphot.calc_mag()
                aperphot.plot()
                mags_ceers[filt] = aperphot.magAB
                mag_errors_ceers[filt] = aperphot.magAB_err
                pdf.savefig(); plt.close()
            except Exception:
                mag_ceers[filt] = np.nan
                mag_errors_ceers[filt] = np.nan
                    
        # save the results
        result_dict = {'galaxy_id':galaxy_sphot.name,
                        **{filt+'_mag_sphot':mags_sphot[filt] for filt in filters},
                        **{filt+'_err_sphot':mag_errors_sphot[filt] for filt in filters},
                        **{filt+'_mag_ceers':mags_ceers[filt] for filt in filters},
                        **{filt+'_err_ceers':mag_errors_ceers[filt] for filt in filters},
                        'aper_size':max(aper_sci.a,aper_sci.b)}
        df_results = pd.concat([df_results,pd.DataFrame(result_dict,index=[0])],ignore_index=True)
        pdf.close()
        df_results.to_csv(os.path.join(out_folder,'results.csv'))
    except Exception:
        pass
print(df_results)
