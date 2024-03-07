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

# import aperture energy table
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-encircled-energy
EE_UVIS_path = pkg_resources.path(resources,
                                  'wfc3uvis2_aper_007_syn.csv')
EE_IR_path = pkg_resources.path(resources,
                                'ir_ee_corrections.csv')
with EE_UVIS_path as p:
    EE_UVIS_table = pd.read_csv(p).set_index('FILTER')
with EE_IR_path as p:
    EE_IR_table = pd.read_csv(p).set_index('FILTER')

def get_interpEE(EE_series):
    keys = EE_series.keys()[1:]
    vals = EE_series.values[1:]
    radii = np.array([float(key.split('#')[1]) for key in keys])
    vals_interp = interpolate.interp1d(radii,vals,kind='linear',
                                       fill_value=(0,1),bounds_error=False)
    return vals_interp

EE_F555W = get_interpEE(EE_UVIS_table.loc['F555W'])
EE_F814W = get_interpEE(EE_UVIS_table.loc['F814W'])
EE_F160W = get_interpEE(EE_IR_table.loc['F160W'])
EE_functions = {
    'F555W': EE_F555W,
    'F814W': EE_F814W,
    'F160W': EE_F160W
}

# prepare filter info
filter_info = {
    'F555W': load_filters(['wfc3_uvis_f555w'])[0],
    'F814W': load_filters(['wfc3_uvis_f814w'])[0],
    'F160W': load_filters(['wfc3_ir_f160w'])[0],
}

# Vega zero point
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-photometric-calibration
# https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/ir-photometric-calibration
vega_zp = {
    'F555W': 25.838,
    'F814W': 24.699,
    'F160W': 24.662
}


def HST_aper_phot(RA,DEC,theta,files=None,hdu_list=None,interp=False,return_type='Vega'):
    pos = SkyCoord(RA * u.deg, DEC * u.deg)
    aperture_obj = SkyCircularAperture(pos, theta.to(u.arcsec))
    # do photometry
    filters = []
    values = []

    if hdu_list is None:
        if files is None:
            assert False, "HST image files or HDU objects are not given"
        hdu_list = []
        for fl in files:
            hdu_list.append(fits.open(fl)[0])
            
    for hdu in hdu_list:
        header = hdu.header
        cs = WCS(header = header)
        img = hdu.data
#         err = get_err(hdu)
        filt = header['FILTER']
        exptime = header['EXPTIME']
        photflam = header['PHOTFLAM']
        EE_ratio = EE_functions[filt](theta.to(u.arcsec).value)

        # unit conversion equivalency prep
        orig_units = u.erg / u.cm**2 / u.s / u.AA

        # aperture photometry using photutils
        if interp:
            flux_results = interp_aperture(img,cs,pos,theta.to(u.arcsec))
            total_electrons = flux_results['flux']['interp']['total']            
        else:
            phot_table = aperture_photometry(img, aperture_obj, wcs=cs)#,error=err)
            total_electrons = phot_table['aperture_sum'].value[0]

        # unit conversion
        if return_type == 'flux':
            val = total_electrons * photflam / exptime * orig_units / EE_ratio
            
        if return_type == 'maggies':
            total_flux = total_electrons * photflam / exptime * orig_units / EE_ratio
            filter_obj = filter_info[filt]
            eqs = u.spectral_density(filter_obj.wave_effective * u.AA)
            val = total_flux.to(u.Jy,equivalencies=eqs).value/3631
            
        if return_type == 'Vega':
            zp = vega_zp[filt]
            val = -2.5 * np.log10(total_electrons / exptime) + zp
            
        if return_type == 'AB':
            
            
        # save data
        filters.append(filt)
        values.append(val)
        
    return filters,values
        
#         total_err  = phot_table['aperture_sum_err'].value[0]
#             local_err  = aperture.to_mask().multiply(err)