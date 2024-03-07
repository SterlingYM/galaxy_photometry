import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd

import astropy.units as u
from astropy.io import fits
from astropy.nddata import CCDData, Cutout2D, block_reduce
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.modeling import models
from astropy.stats import sigma_clip, sigma_clipped_stats
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats

from petrofit.modeling import get_default_sersic_bounds, PSFConvolvedModel2D
from sphot import subtract_background
from sphot import ModelFitter,Vegamag_from_HST
from util.JWST_photometry import get_AB_zp, get_AB_Vega_offset

class Image2D():
    def __init__(self,fitsfile):
        if isinstance(fitsfile,str):
            hdul = fits.open(fitsfile)
        elif isinstance(fitsfile,fits.hdu.hdulist.HDUList):
            hdul = fitsfile
        else:
            raise ValueError("Input fits file has to be HDUList or path to FITS file."+\
                            "Input type:",type(fitsfile))
        self.hdul = hdul
        self.telescope = self.get_telescope_info()
        self.data = self.get_image_hdu().data
        self.wcs = WCS(self.get_image_hdu())        
        self.filter = self.get_filter()
        self.exptime = self.get_exptime()
        
        if self.telescope == 'JWST':
            # JWST image has value of 0.00 instead of NaN for 
            # pixels outside the detector. This causes bias in fitting
            # so we replace it with NaN.
            self.data[self.data == 0.] = np.nan
            
            # set ABmag zeropoint
            pixar_sr = hdul['SCI'].header['PIXAR_SR']
            zp1 = -2.5 * np.log10(pixar_sr * 1e6 / 3631)
            self.zp = zp1
            
        
    def get_telescope_info(self):
        return self.hdul['PRIMARY'].header['TELESCOP']
            
    def get_image_hdu(self):
        if self.telescope == 'HST':
            return self.hdul['PRIMARY']
        elif self.telescope == 'JWST':
            return self.hdul['SCI']
    
    def get_filter(self):
        if self.telescope == 'HST':
            return self.hdul['PRIMARY'].header['FILTER']
        if self.telescope == 'JWST':
            return self.hdul['PRIMARY'].header['FILTER'] 
        
    def get_exptime(self):
        if self.telescope == 'HST':
            return self.hdul['PRIMARY'].header['EXPTIME']
        if self.telescope == 'JWST':
            return self.hdul['SCI'].header['XPOSURE']

    def set_psf(self,PSF,oversample_factor=1):
        PSF /= PSF.sum()
            
        # reduce PSF to the original bpixel scale
        if oversample_factor > 1:
            PSF_reduced = block_reduce(PSF, oversample_factor)
            PSF_reduced /= PSF_reduced.sum()
            self.PSF = PSF_reduced
        else:
            self.PSF = PSF
    
    def plot_psf(self):
        plt.figure(figsize=(7,7))
        plt.imshow(self.PSF, vmin=0, vmax=self.PSF.std()/10)
        plt.show()
        
    def plot(self,percentiles=[1,99],cmap='viridis'):
        vmin,vmax = np.nanpercentile(self.data,percentiles)
        if vmin <= 0:
            offset = -vmin + 1e-1
        else:
            offset = 0
        vmin += offset 
        vmax += offset
        
        norm = LogNorm(vmin=vmin,vmax=vmax)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection=self.wcs)
        ax.imshow(self.data + offset,norm=norm,cmap=cmap)
        return fig,ax
        
class Images2DCutout():
    def __init__(self,objname,df_mask,images,size_factor=1):
        '''
        inputs:
            objname (str): name of the object
            df_mask (pandas.DataFrame): table containing mask info
            images (list): list of 
        '''
        self.objname = objname
        self.df_mask = df_mask
        self.get_mask_info()
        
        self.images = []
        for img in images:
            try:
                img_cutout = self.cutout_image(img,size_factor=size_factor)
                img_cutout.filter = img.filter
                img_cutout.exptime = img.exptime
                img_cutout.PSF = img.PSF
                img_cutout.telescope = img.telescope
                if hasattr(img,'zp'):
                    img_cutout.zp = img.zp
                self.images.append(img_cutout)      
            except Exception as e:
                print(img.filter+' failed:',e)
        
    def get_mask_info(self):
        s = self.df_mask['OBJECT'].str.contains(self.objname)
        cols = ['OBJECT','RA_OBJ','DEC_OBJ','slitLen']
        objname,ra,dec,slitlen = self.df_mask.loc[s,cols].values[0]

        self.mask_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
        self.mask_size = slitlen * u.arcsec*3
        
    def cutout_image(self,img,size_factor):
        image_cutout = Cutout2D(img.data, 
                                position=self.mask_coord,
                                size=self.mask_size * size_factor, 
                                wcs=img.wcs)
        return image_cutout

class Galaxy():
    def __init__(self,objname,cutout):
        '''
        inputs:
            objname (str): name of this object
            cutout (Images2DCutout): cutout images
        '''
        self.objname = objname
        self.images = cutout.images
        self.filters = [img.filter for img in self.images]
        
        for filt in self.filters:
            i = self.filters.index(filt)
            setattr(self,filt,self.images[i])
        
    def plot_images(self,percentiles=[0.1,99.],cmap='viridis',return_figax=False):
        size = len(self.images)
        if size > 3:
            ncols = 3
            nrows = np.floor(size/3).astype(int) + 1
        else: 
            ncols = size
            nrows = 1
        fig = plt.figure(figsize=(5*ncols,5*nrows))
        
        axes=[]
        for i,(img,filt) in enumerate(zip(self.images,self.filters)):
            ax = fig.add_subplot(nrows,ncols,i+1,projection=img.wcs)
            axes.append(ax)
            vmin,vmax = np.nanpercentile(img.data,percentiles)
            if vmin <= 0:
                offset = -vmin + 1e-1
            else:
                offset = 0
            vmin += offset 
            vmax += offset

            norm = LogNorm(vmin=vmin,vmax=vmax)
            ax.imshow(img.data + offset,norm=norm,cmap=cmap)
            ax.set_title(filt,fontsize=18)
        if return_figax:
            return fig,axes
        else:
            plt.show()
        
    def sub_gradient(self,filt=None):
        if filt is None:
            filt = self.filters
        else:
            filt = [np.squeeze(filt)]

        for f in filt:
            i = self.filters.index(f)
            bgsub = subtract_background(self.images[i].data,title=self.images[i].filter)
            self.images[i].bgsub = bgsub
            
    def fit_model(self,filt=None,fixed_params={},**kwargs):
        if filt is None:
            filt = self.filters
        else:
            filt = [np.squeeze(filt)]
        for f in filt:
            i = self.filters.index(f)
            fitter = self._fit_model(self.images[i].filter,
                                     self.images[i].bgsub,
                                     self.images[i].PSF,
                                     fixed_params=fixed_params,**kwargs)
            self.images[i].fitter = fitter
            
    def fit_fixed_model(self,base_filter,filter_to_fit,fixed_params=['r_eff','n','ellip'],fitter_kwargs={}):
        ''' continue fit using known r_eff, n, and ellip to filter_to_fit.
        This function assumes that base_filter image is fitted with fit_model() already.'''
        
        # get values of fixed parameters
        fixed_param_dict = {}
        for key in fixed_params:
            print(key)
            if key == 'r_eff':
                val = self.scale_r_eff_wcs(base_filter,filter_to_fit)
            else:
                val = self.get_bestfit_params_from(base_filter,[key])[0]
            fixed_param_dict[key] = val
            
        # fit
        self.fit_model(filt=filter_to_fit,
                       fixed_params = fixed_param_dict,
                       fitter_kwargs = fitter_kwargs)
        
            
    def _fit_model(self,filtername,img,PSF,fixed_params={},fitter_kwargs={}):
        fitter_default_kwargs = dict(N_iter=4,
                                     rtol_init=1e-0,
                                     rtol_iter=1e-1)
        fitter_updated_kwargs = fitter_default_kwargs
        fitter_updated_kwargs.update(fitter_kwargs)
        
        # initial guesses -- use central regions
        center_loc = (int(img.shape[0]/2),int(img.shape[1]/2))
        cutout_size = (30,30)
        center_cutout = Cutout2D(img, center_loc, cutout_size)
        maxval = np.nanmax(center_cutout.data)
        
        # Standard Sersic model
        sersic_model = models.Sersic2D(
                amplitude = maxval/100, # Intensity at r_eff
                r_eff     = img.shape[0]/10, # Effective or half-light radius
                n         = 4, # Sersic index
                x_0       = img.shape[0]/2, # center of model in the x direction
                y_0       = img.shape[1]/2, # center of model in the y direction
                ellip     = 0.5, # Ellipticity
                theta     = 1.5, #  Rotation angle in radians
                bounds    = get_default_sersic_bounds(), # Parameter bounds
        )
        galaxy_model = sersic_model #+ sersic_model

        # PSF comvolved model
        psf_sersic_model = PSFConvolvedModel2D(galaxy_model, 
                                               psf=PSF, 
                                               oversample=4)

        # Initialize ModelFitter object
        fitter = ModelFitter(data = img,
                             model = psf_sersic_model)

        # initial guess and bounds
        lower_bounds = [-2,0,1e-2,0,0,0,0,0,-1]
        upper_bounds = [np.log10(np.nanmax(img)),2,1,1,1,1,np.pi,90,2]
        bounds = np.vstack([lower_bounds,upper_bounds]).T

        # fit
        fitter.fit_model(bounds=bounds,
                         fixed_params=fixed_params,
                         **fitter_updated_kwargs)
        fitter.plot_result(title=filtername)
        return fitter
    
    def get_bestfit_params_from(self,base_filter,param_names=['n','ellip']):
        ''' a helper function to call specific best-fit values for convenience '''
        base_img = getattr(self,base_filter)
        values = []
        for param_name in param_names:
            values.append(base_img.fitter.bestfit[param_name])
        return values

        
    def scale_r_eff_wcs(self,base_filter,target_filter):
        ''' scale the fitted r_eff in one pixel scale to another
        based on the wcs information.
        inputs:
            base_filter (str): name of the filter to be used as a base
            target_filter (str): name of the filter to scale to
        output:
            r_eff_pix_scaled (float)
        '''
        # get data
        base_img = getattr(self,base_filter)
        
        # best-fit values
        r_eff_pix_base = base_img.fitter.bestfit['r_eff'] 
    
        # convert values to the target_filter image pixel scale
        pix_scale_base = base_img.wcs.proj_plane_pixel_scales()[0]
        r_eff_physical = r_eff_pix_base * pix_scale_base
    
        # iterate through target filters
        target_img = getattr(self,target_filter)
        pix_scale_target = target_img.wcs.proj_plane_pixel_scales()[0]
        r_eff_pix_scaled = (r_eff_physical / pix_scale_target).value
        return r_eff_pix_scaled

    
    def calc_mag(self,method='Sersic',filt=None):
        if filt is None:
            filt = self.filters
        else:
            filt = [np.squeeze(filt)]
            
        if method == 'Sersic':
            for f in filt:
                i = self.filters.index(f)
                img = self.images[i]
                counts = img.fitter.bestfit_img.sum()
                if img.telescope == 'HST':
                    mag = Vegamag_from_HST(counts,f,img.exptime)

                elif img.telescope == 'JWST':
                    AB_vega_offset = get_AB_Vega_offset(img.filter)
                    AB = -2.5 * np.log10(counts) + img.zp # AB mag
                    mag = AB - AB_vega_offset  # Vega mag
                    
                img.mag = mag
                print(f+f': {mag:.2f}')
    
    def plot_photometry(self,filt=None):
        # extra requirement: astro-sedpy
        from sedpy import observate
        
        sedpy_filters = {
            'F555W': 'wfc3_uvis_f555w',
            'F814W': 'wfc3_uvis_f814w',
            'F160W': 'wfc3_ir_f160w',
            'F090W': 'jwst_f090w',
            'F150W': 'jwst_f150w',
            'F277W': 'jwst_f277w',
        }
        
        if filt is None:
            filt = self.filters
        else:
            filt = [np.squeeze(filt)]

        fig = plt.figure(figsize=(10,6))
        ax = fig.add_subplot(4,1,(2,4))
        ax_top = fig.add_subplot(4,1,(1))
        plt.subplots_adjust(hspace=0)
        for f in filt:
            # data
            i = self.filters.index(f)
            img = self.images[i]
            filter_obj = observate.Filter(sedpy_filters[f])
            
            # transmission curve
            wav,trans = filter_obj.wavelength,filter_obj.transmission
            s = trans > 1e-3
            ax_top.plot(wav[s],trans[s])
            ax_top.fill_between(wav[s],0,trans[s])
            
            # photometry
            ax.errorbar(filter_obj.wave_effective,img.mag,
                        yerr=0.5,ms=7,fmt='o')
            
        # prettify
        ax.invert_yaxis()
        ax.tick_params(labelsize=13)
        ax.set_xlabel(r'Wavelength [$\AA$]',fontsize=18)
        ax.set_ylabel(r'mag',fontsize=18)
        ax_top.set_title(self.objname,fontsize=18)
        ax_top.set_ylim(0,)
        ax_top.set_xticklabels([])
        ax.set_xlim(ax_top.get_xlim())
        ax_top.set_yticks([])
        return fig,ax,ax_top

        
#### petrosian photometry
from util.petrophot import subtract_bkg,find_center,calc_petro,plot_petro,calc_r_petro,do_photometry

def do_petrophot(img,plot=True,sigma=3):
    
    # subtract background
    img_bgsub = subtract_bkg(img.data.copy(),sigma=sigma)
    
    # find the center of galaxy
    r_list = np.arange(1,30)
    xylist = [find_center(img.data,r).x for r in r_list]
    xlist,ylist = np.asarray(xylist).T

    # construct apertures
    position = [np.mean(xlist),np.mean(ylist)]
    r_list = np.linspace(1,30)

    # do photometry
    flux_list = []
    area_list = []
    for r in r_list:
        aperture = CircularAperture(position, r=r)
        phot_table = aperture_photometry(img_bgsub, aperture)
        flux_list.append(phot_table['aperture_sum'].value[0])
        area_list.append(aperture.area)
    r_list = np.asarray(r_list)
    flux_list = np.asarray(flux_list)
    area_list = np.asarray(area_list)

    # calculate Petrosian values
    locs, petrovals = calc_petro(r_list,area_list,flux_list)
    if plot:
        plot_petro(locs,petrovals)
    r_petro = calc_r_petro(locs,petrovals,refval=0.2)
    
    # do photometry
    counts_total, bkg_total, effective_total = do_photometry(img_bgsub,r_eff=r_petro,position=position,
                                                             aperture_factor=2.9,r_in_factor=3,r_out_factor=5,
                                                             plot=plot)

    # check the fluctuation of the total count values
    counts_list = []
    factors = np.linspace(0.5,4,30)
    for fac in factors:
        _,_,counts = do_photometry(img_bgsub,r_eff=r_petro,position=position,
                                   aperture_factor=fac,r_in_factor=3,r_out_factor=5,plot=False)
        counts_list.append(counts)
    if plot:
        plt.figure(figsize=(8,6))
        plt.plot(factors,counts_list)

    # calculate magnitude
    if img.telescope == 'JWST':
        AB_vega_offset = get_AB_Vega_offset(img.filter)
        AB = -2.5 * np.log10(counts_total) + img.zp # AB mag
        mag = AB - AB_vega_offset  
        print(img.filter+f': {mag:.2f}')
        img.petromag = mag