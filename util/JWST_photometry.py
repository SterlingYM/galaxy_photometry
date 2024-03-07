import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy.visualization import simple_norm
import astropy.units as u
from photutils import SkyCircularAperture, aperture_photometry
from sedpy.observate import load_filters
import importlib.resources as pkg_resources

from .aperture_phot import interp_aperture
from . import resources

abvega_table_path = pkg_resources.path(resources,
                                'NIRCam_ABVega_offset.txt')
with abvega_table_path as p:
    AB_Vega_offset_table = pd.read_csv(p, names=['filter','pupil','offset'],
                                       delim_whitespace=True).set_index('filter')

def get_AB_zp(hdul):
    pixar_sr = hdul[1].header['PIXAR_SR']
    zp1 = -2.5 * np.log10(pixar_sr * 1e6 / 3631)
    return zp1

def get_AB_Vega_offset(filt):
    return AB_Vega_offset_table.loc[filt,'offset'].astype(float)
    
# import aperture energy table
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-encircled-energy
# EE_UVIS_path = pkg_resources.path(resources,
#                                   'wfc3uvis2_aper_007_syn.csv')
# EE_IR_path = pkg_resources.path(resources,
#                                 'ir_ee_corrections.csv')
# with EE_UVIS_path as p:
#     EE_UVIS_table = pd.read_csv(p).set_index('FILTER')
# with EE_IR_path as p:
#     EE_IR_table = pd.read_csv(p).set_index('FILTER')

def get_interpEE(filt):
    ee_dat_path = pkg_resources.path(resources,filt.lower()+'_ee.dat')
    with ee_dat_path as p:
        headers = pd.read_csv(p, delim_whitespace=True, nrows=0).columns[1:]
        df_ee = pd.read_csv(p,delim_whitespace=True,comment='#',names=headers)
        radii_pix = df_ee['rad'].values
        EE = df_ee['ee'].values
    vals_interp = interpolate.interp1d(radii_pix,EE,kind='linear',fill_value=(0,1),bounds_error=False)
    return vals_interp

# EE_F555W = get_interpEE(EE_UVIS_table.loc['F555W'])
# EE_F814W = get_interpEE(EE_UVIS_table.loc['F814W'])
# EE_F160W = get_interpEE(EE_IR_table.loc['F160W'])
EE_functions = {
    'F090W': get_interpEE('F090W'),
    'F150W': get_interpEE('F150W'),
    'F277W': get_interpEE('F277W')
}

# prepare filter info
# filter_info = {
#     'F555W': load_filters(['wfc3_uvis_f555w'])[0],
#     'F814W': load_filters(['wfc3_uvis_f814w'])[0],
#     'F160W': load_filters(['wfc3_ir_f160w'])[0],
# }

# Vega zero point
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
# vega_zp = {
#     'F555W': 25.838,
#     'F814W': 24.699,
#     'F160W': 24.662
# }


def JWST_aper_phot(RA,DEC,theta,files=None,hdul_list=None,interp=False,return_type='Vega'):
    pos = SkyCoord(RA * u.deg, DEC * u.deg)
    aperture_obj = SkyCircularAperture(pos, theta.to(u.arcsec))
    # do photometry
    filters = []
    values = []

    if hdul_list is None:
        if files is None:
            assert False, "HST image files or HDU objects are not given"
        hdul_list = []
        for fl in files:
            hdul_list.append(fits.open(fl))
            
    for hdul in hdul_list:
        header = hdul[1].header
        wcs = WCS(header = header)
        img = hdul[1].data
        filt = hdul[0].header['FILTER']
        exptime = 1 # data file already accounts for exposure time
        EE_ratio = EE_functions[filt](aperture_obj.to_pixel(wcs).r)

        # unit conversion equivalency prep
#         orig_units = u.erg / u.cm**2 / u.s / u.AA

        # aperture photometry using photutils
        if interp:
            flux_results = interp_aperture(img,wcs,pos,theta.to(u.arcsec))
            total_counts = flux_results['flux']['interp']['total']            
        else:
            phot_table = aperture_photometry(img, aperture_obj, wcs=wcs)#,error=err)
            total_counts = phot_table['aperture_sum'].value[0]

        # unit conversion
#         if return_type == 'flux':
#             val = total_electrons * photflam / exptime * orig_units / EE_ratio
            
#         if return_type == 'maggies':
#             total_flux = total_electrons * photflam / exptime * orig_units / EE_ratio
#             filter_obj = filter_info[filt]
#             eqs = u.spectral_density(filter_obj.wave_effective * u.AA)
#             val = total_flux.to(u.Jy,equivalencies=eqs).value/3631
            
        if return_type == 'Vega':
            EE_offset = 2.5 * np.log10(EE_ratio)
            AB_zp = get_AB_zp(hdul)
            AB_vega_offset = get_AB_Vega_offset(filt)
            AB = -2.5 * np.log10(total_counts / exptime) + AB_zp + EE_offset
            val = AB - AB_vega_offset
            
        # save data
        filters.append(filt)
        values.append(val)
        
    return filters,values
        
#         total_err  = phot_table['aperture_sum_err'].value[0]
#             local_err  = aperture.to_mask().multiply(err)