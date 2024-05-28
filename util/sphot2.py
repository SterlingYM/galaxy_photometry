import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import logging
from tqdm.auto import tqdm
from scipy.optimize import minimize, leastsq
from scipy import stats
from scipy.stats import multivariate_normal

from photutils import CircularAperture

import astropy.units as u
from astropy.stats import sigma_clip
from astropy.nddata import Cutout2D
from astropy.modeling import models
from astropy.convolution import convolve, Gaussian2DKernel
from petrofit.modeling import PSFConvolvedModel2D, model_to_image

# Configure logging for sphot2
logger = logging.getLogger('sphot2')
logger.setLevel(logging.WARNING)
sphot_formatter = logging.Formatter('[sphot] %(levelname)s: %(message)s (%(funcName)s)')
sphot_handler = logging.StreamHandler()
sphot_handler.setFormatter(sphot_formatter)
logger.addHandler(sphot_handler)

def calc_mag(total_counts_Mjy_per_Sr):
    PIXAR_SR = ((0.03*u.arcsec)**2).to(u.sr).value
    magAB = -6.10 - 2.5 *np.log10(total_counts_Mjy_per_Sr*PIXAR_SR)
    return magAB

def astroplot(data,percentiles=[1,99.5],cmap='viridis',ax=None,
              offset=0,norm=None,figsize=(5,5),title=None,set_bad='r'):
    if (data is None) or np.isnan(data).all():
        raise ValueError('Data is empty!')
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=figsize)

    # auto-normalize data in log scale 
    # (and take care of the negative values if needed)       
    if norm is None:
        vmin,vmax = np.nanpercentile(data,percentiles)
        if vmin <= 0:
            offset = -vmin + 1e-1 # never make it zero
        else:
            offset = 0
        vmin += offset 
        vmax += offset
        norm = LogNorm(vmin=vmin,vmax=vmax)
    else:
        assert offset is not None, 'offset has to be provided if norm is provided'

    # plot
    clipped_data = data.copy() + offset
    clipped_data[clipped_data<=norm.vmin] = norm.vmin
    clipped_data[clipped_data>=norm.vmax] = norm.vmax
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
        cmap.set_bad(set_bad)
    ax.imshow(clipped_data,norm=norm,cmap=cmap,origin='lower')
    ax.set_yticks([])
    ax.set_xticks([])
    if title is not None:
        ax.set_title(title,fontsize=13)
    return norm,offset

class MultiBandCutout():
    ''' a container for CutoutData '''
    def __init__(self,name=None,**kwargs):
        self.name = name
        for key,val in kwargs.items():
            setattr(self,key,val)
        if not hasattr(self,'filters'):
            self.filters = []
    
    def add_image(self,filtername,data):
        if hasattr(self,filtername):
            raise ValueError("The specified filter already exists")
        setattr(self,filtername,data)
        self.filters.append(filtername)
        
    @property
    def images(self):
        return dict(zip(self.filters,[getattr(self,filtername) for filtername in self.filters]))
    
    @property
    def image_list(self):
        return [getattr(self,filtername) for filtername in self.filters]
               
class CutoutData():
    def __init__(self,data,psf,psf_oversample,filtername=None,**kwargs):
        '''
        Args:
            data (2D array): cutout image data
            psf (2D array): PSF data. It can be oversampled (pixel scale could be an integer multiple of the image)
            PSF_oversample (int): oversampling factor of the PSF
            filtername (str): filter name
        '''
        self.data = data
        self.psf = psf
        self.psf_oversample = psf_oversample
        self.filtername = filtername
        for key,val in kwargs.items():
            setattr(self,key,val)
            
        self.fix_psf()
            
    def fix_psf(self):
        ''' make sure the sum of PSF is 1 '''
        self.psf /= np.sum(self.psf)
    
    def plot(self):
        astroplot(self.data,title=self.filtername)
        
    def init_size_guess(self,sigma_guess=10,center_slack = 0.20,
                        plot=False,sigma_kernel=5):
        '''roughly estimate the effective radius using Gaussian profile.
        Args:
            sigma_guess (float): initial guess for the standard deviation of the Gaussian profile (in pixels)
            center_slack (float): the fraction of the image size (from center) within which the center of the galaxy is expected. Default is 5%
        Returns:
            float: rough estimate of the effective radius (in pixels)
        '''

        from scipy.optimize import curve_fit
        from scipy.ndimage import gaussian_filter
        
        if plot:
            fig,axes = plt.subplots(1,2,figsize=(8,3))
        
        centers,sigmas  = [],[]
        for i,axis in enumerate([1,0]):
            if np.isfinite(self.data).sum() == 0:
                means_smooth = gaussian_filter(self.data,sigma=sigma_kernel).mean(axis=axis)
            else:
                kernel = Gaussian2DKernel(x_stddev=sigma_kernel,y_stddev=sigma_kernel)
                smooth_img = convolve(self.data, kernel)
                s1 = np.any(np.isfinite(self.data),axis=axis)
                means_smooth = np.nanmean(smooth_img[s1],axis=axis)
            
            # fit Gaussian to smoothed counts
            axis_pixels = np.arange(self.data.shape[axis-1])[s1]
            shape = len(axis_pixels)
            gaussian = lambda x,mu,sigma,amp,offset: amp*np.exp(-(x-mu)**2/(2*sigma**2))+offset
            bounds = ([shape/2 - center_slack*shape,0,0,-np.inf],
                      [shape/2 + center_slack*shape,shape,np.inf,np.inf])

            popt,_ = curve_fit(gaussian,axis_pixels,means_smooth,
                            p0=[shape/2,sigma_guess,means_smooth.max(),0],
                            bounds=bounds)
            centers.append(popt[0])
            sigmas.append(popt[1])
            
            if plot:
                ax = axes[i]
                ax.plot(axis_pixels,means_smooth,c='k')
                ax.plot(axis_pixels,gaussian(axis_pixels,*popt),c='orange',lw=3)
                ax.axvline(popt[0],c='yellowgreen',ls=':',label=f'center={popt[0]:.1f}')
                ax.axvspan(popt[0]-popt[1],popt[0]+popt[1],color='yellowgreen',alpha=0.3,label=f'sigma={popt[1]:.1f}')
                ax.legend(frameon=False)
                ax.set_xlim(0,self.data.shape[axis-1])
                ax.tick_params(direction='in')

                axes[0].set_xlabel('x (pixels)')
                axes[1].set_xlabel('y (pixels)')
                axes[0].set_ylabel('summed counts')
                fig.suptitle('Initial estimation of galaxy shape')
        
        # check if any fit 'failed':
        if np.all(np.array(sigmas) > np.array(self.data.shape)/1.5):
            raise ValueError('looks like cutout is too small or smoothing parameter is not set correctly')
        elif np.any(np.array(sigmas) > np.array(self.data.shape)/1.5):
            s = np.array(sigmas) < np.array(self.data.shape)/1.5
            size_guess = np.array(sigmas)[s][0]
        else:
            size_guess = np.mean(sigmas)
        print(size_guess)
        self.x0_guess = centers[1]
        self.y0_guess = centers[0]
        self.size_guess = size_guess
        
    def measure_background(self,aperture_size = 3, clip_sigma = 3.5, plot=False,
                           percentile = 20):
        ''' measure background level and std using annulus'''
        if not hasattr(self,'x0_guess'):
            self.init_size_guess()
        r_eff = self.size_guess

        # aperture (to invert)
        aperture = CircularAperture((self.x0_guess,self.y0_guess), aperture_size*self.size_guess)
        aperture_mask = aperture.to_mask(method='center')
        aperture_mask_img = ~aperture_mask.to_image(self.data.shape).astype(bool)
        data_masked = aperture_mask_img * self.data
        data_masked[data_masked==0] = np.nan
        data_masked_ma = ma.array(data_masked, mask=~(aperture_mask_img | np.isfinite(self.data))) # using ma suppresses warnings in sigma_clip

        # sigma-clip
        clip_mask = sigma_clip(data_masked_ma,sigma=clip_sigma,maxiters=5)
        data_masked_clipped = np.ones_like(data_masked)*np.nan
        data_masked_clipped[~clip_mask.mask] = clip_mask.data[~clip_mask.mask]

        # result -- sigma-clipped image outside the aperture
        bkg_mean = np.nanpercentile(data_masked_clipped,percentile)
        bkg_std = np.nanstd(data_masked_clipped)
        
        if plot:
            astroplot(data_masked_clipped, set_bad = 'w',
                  title=f'background estimation\nmean={bkg_mean:.3f}, std={bkg_std:.3f}')
        self.bkg_mean = bkg_mean
        self.bkg_std = bkg_std
        
    def prepare_model(self,**kwargs):
        ''' create initial guesses unless specified '''
        # TODO: change this into default kwargs
        amplitude_divider = kwargs.get('amplitude_divider',100)
        ellip_guess = kwargs.get('ellip_guess',0.5)
        theta_guess = kwargs.get('theta_guess',1.5)
        
        # initial guesses -- use values near the center of the galaxy
        center_cutout = Cutout2D(self.data, (self.x0_guess,self.y0_guess), self.size_guess)
        maxval = np.nanmax(center_cutout.data)
        
        init_guess_dict = dict(
            amplitude = kwargs.get('amp_guess',maxval/amplitude_divider), # Intensity at r_eff
            r_eff     = self.size_guess, # Effective or half-light radius
            n         = kwargs.get('n_guess',4), # Sersic index
            x_0       = self.x0_guess, # center of model in the x direction
            y_0       = self.y0_guess, # center of model in the y direction
            ellip     = kwargs.get('ellip_guess',0.5), # Ellipticity
            theta     = kwargs.get('theta_guess',1.5), #  Rotation angle in radians
            # background = 0
        )
        
        galaxy_model = models.Sersic2D(**init_guess_dict)
        psf_sersic_model = PSFConvolvedModel2D(galaxy_model, 
                                                psf=self.psf, 
                                                psf_oversample = self.psf_oversample,
                                                oversample = self.psf_oversample)
        self.galaxy_model = galaxy_model
        self.psf_sersic_model = psf_sersic_model
        
    def get_fitter_bounds(self,img):
        '''generate the initial guess based on the input image. Note these bounds are in the standarized scale'''
        # ['amplitude' 'r_eff' 'n' 'x_0' 'y_0' 'ellip' 'theta' 'psf_pa' 'background']      
        if (img>0).sum() > 0:
            lower_bounds = [np.log10(np.nanmin(img[img>0])),0,1e-2,0,0,0,0,0]
        else:
            lower_bounds = [np.log10(1e-10),0,1e-2,0,0,0,0,0]
        upper_bounds = [np.log10(np.nanmax(img)),2,1,1,1,1,np.pi,90]
        bounds = np.vstack([lower_bounds,upper_bounds]).T
        return bounds
    
    def fit_model(self,error_model=None,fixed_params={},fitter_kwargs={},verbose=True,reuse_fitter=False,**kwargs):
        fitter_default_kwargs = dict(N_iter=4,
                                     rtol_init=1e-0,
                                     rtol_iter=1e-1)
        fitter_updated_kwargs = fitter_default_kwargs
        fitter_updated_kwargs.update(fitter_kwargs)
        
        # subtract background
        self.data_bksub = self.data - self.bkg_mean
        
        if (not hasattr(self,'fitter')) or (not reuse_fitter):
            # Initialize ModelFitter object
            if not hasattr(self,'psf_sersic_model'):
                self.prepare_model(**kwargs)
            if error_model is None:
                error_model = np.ones_like(self.data) * self.bkg_std
            self.error_model = error_model
            fitter = ModelFitter(data = self.data_bksub,
                                err = error_model,
                                model = self.psf_sersic_model,
                                verbose = verbose)
            self.fitter = fitter
        
        # fit
        self.fitter.fit_model(bounds = self.get_fitter_bounds(self.data_bksub),
                              fixed_params = fixed_params,
                              **fitter_updated_kwargs)
        self.bestfit_img = self.fitter.bestfit_img
        self.residual_total = self.data - self.bestfit_img # background is NOT subtracted
        self.residual_bksub = self.data_bksub - self.bestfit_img # background is subtracted
        self.relative_residual = self.residual_bksub / self.error_model
        plot_result = kwargs.get('plot_result',False)
        if plot_result:
            self.plot_fitresult(title=self.filtername)
        
    def plot_fitresult(self,title=''):
        fig,axes = plt.subplots(1,4,figsize=(16,4))
        plt.subplots_adjust(wspace=0.03,bottom=0.1,right=0.9)
    
        norm,offset = astroplot(self.data_bksub, percentiles=[0.1,99.9], ax=axes[0])
        astroplot(self.error_model, norm=norm, offset=offset, ax=axes[1])
        astroplot(self.bestfit_img, norm=norm, offset=offset, ax=axes[2])
        astroplot(self.residual_bksub, norm=norm, offset=offset, ax=axes[3])

        if title != '':
            axes[0].set_title('Original: '+title)
        else:
            axes[0].set_title('Original image')
        axes[1].set_title('Error model')
        axes[2].set_title('Best-fit model')
        axes[3].set_title('Residual')
        for ax in axes[1:]:
            ax.set_yticklabels([])
        plt.show()

    def update_error_model(self, error_model, residual, modelimg, 
                        min_noise_floor = 0,
                        min_fraction_delta = 0.1):
        ''' update error model based on the residual and the model image '''
        relative_residual = abs(residual/error_model)
        delta_error = (error_model * relative_residual - error_model)
        
        # apply 100% change in error at 0% flux rank, N% change in error at 100% flux rank
        # N is determined by min_fraction_delta
        flux_rank = stats.rankdata(modelimg).reshape(modelimg.shape) 
        deweight_by_flux = 1 - (1 - min_fraction_delta) * flux_rank/flux_rank.max()
        delta_error *= deweight_by_flux
        
        updated_error_model = error_model + delta_error
        updated_error_model[updated_error_model < min_noise_floor] = min_noise_floor
        return updated_error_model

    def calc_mag(self,total_counts_Mjy_per_Sr):
        PIXAR_SR = ((0.03*u.arcsec)**2).to(u.sr).value
        magAB = -6.10 - 2.5 *np.log10(total_counts_Mjy_per_Sr*PIXAR_SR)
        return magAB

    def update_background_estimate(self,error_model,plot=False,return_weight=False):
        _res = self.residual_bksub
        _bestfit_img = self.bestfit_img.copy()
        _bestfit_img[_bestfit_img<=0] = _bestfit_img[_bestfit_img>0].min()
        deweight_center = 1/(_bestfit_img**2) 
        deweight_center /= np.nansum(deweight_center)
        deweight_outlier = 1/error_model**2
        deweight_outlier /= np.nansum(deweight_outlier)

        weights = deweight_center * deweight_outlier
        weights /= np.nansum(weights)
        
        mean = np.nansum(weights * self.data)
        variance = np.nansum(weights * (self.data - mean)**2)
        std = np.sqrt(variance)
        
        if plot:
            plt.figure(figsize=(4,3))
            plt.hist(self.data.flatten(),bins=np.linspace(0,1,100));
            plt.axvline(mean,c='r')
            plt.axvspan(mean-std, mean+std, alpha=0.5, color='red')
        
        if return_weight:
            return mean, std, weights
        return mean, std

    def do_photometry(self,plot=False,mag_tol=0.01, fixed_params = {},
                      N_repeat_fit_max=10,N_repeat_update_error=5,
                      rtol_init = 1e-3, rtol_iter = 1e-3, N_iter=4,
                      uniform_error = False):
        
        if fixed_params.get('r_eff',None) is None:
            self.init_size_guess(plot=plot)
        else:
            self.size_guess = fixed_params['r_eff']
            self.x0_guess = fixed_params['x_0']
            self.y0_guess = fixed_params['y_0']
        self.measure_background(aperture_size = 3.5,clip_sigma = 3,plot=plot)
        logger.info(f'Initial background estimation: {self.bkg_mean:3f} +/- {self.bkg_std:.3f}')
        if plot:
            plt.show()

        # ----------------- initial fit -----------------
        logger.info('Starting initial fit...')
        mags = []
        error_model = np.ones_like(self.data) * self.bkg_std
        self.fit_model(error_model=error_model,plot_result=plot,
                       fitter_kwargs = {'N_iter':N_iter, 'rtol_init':rtol_init, 'rtol_iter':rtol_iter},
                       fixed_params = fixed_params)
        mags.append(self.calc_mag(self.fitter.bestfit_img.sum()))
        logger.info(f'Estimated magnitude = {mags[-1]:.3f}')
            
        # ----------------- iterative fit -----------------
        # keep fitting (updating residuals) until the estimated magnitude converges
        for i_fititer in range(N_repeat_fit_max):
            logger.info(f'Starting photometry iteration #{i_fititer+2}...')
            # update error model and background estimation
            
            for _ in range(N_repeat_update_error):
                # update error model and background estimation
                if uniform_error:
                    updated_error_model = np.ones_like(error_model) * self.bkg_std
                else:
                    updated_error_model = self.update_error_model(error_model = error_model,
                                                                residual = self.fitter.residual,
                                                                modelimg = self.fitter.bestfit_img,
                                                                min_noise_floor = self.bkg_std,
                                                                min_fraction_delta = 0.05)
                mean,std = self.update_background_estimate(updated_error_model)
                self.bkg_mean = mean
                self.bkg_std = std
            logger.debug(f'Updated background mean = {self.bkg_mean:.3f}')
            logger.debug(f'Updated background STD = {self.bkg_std:.3f}')

            # fit again
            self.fit_model(error_model=updated_error_model, 
                           plot_result=plot, 
                           fitter_kwargs =  {'N_iter':N_iter, 'rtol_init':rtol_init, 'rtol_iter':rtol_iter},
                           fixed_params = fixed_params)
            mags.append(self.calc_mag(self.fitter.bestfit_img.sum()))
            logger.info(f'Estimated magnitude = {mags[-1]:.3f}')
            
            # check for convergence
            if np.abs(mags[-1] - mags[-2]) < mag_tol:
                logger.info('Magnitude converged within mag_tol')
                break
        if np.abs(mags[-1] - mags[-2]) > mag_tol:
            logger.warning('Magnitude did not converge within mag_tol')
        self.mag = mags[-1]
        
    def bootstrap_fit(self,N_bootstrap=500,vary_background=True):
        fitter = self.fitter

        def lstsq_calcimg(theta):
            ''' helper function for least square. This function allows background level to vary, 
            which helps marginalizing the uncertainty associated with background level estimation '''
            if vary_background:
                _theta,bkg = theta[:-1],theta[-1]
            else:
                _theta = theta
                
            # generate image
            standardized_params = fitter.standardize_params(_theta)
            _img = fitter.calc_img(standardized_params,*fitter.fitter_args)
            
            if vary_background:
                _img += bkg
            return _img

        def lstsq_func(theta):
            ''' helper function to calculate residual based on lstsq_calcimg. '''
            _img = lstsq_calcimg(theta)
            chi = (fitter.data - _img) / fitter.err
            return chi.ravel()
            

        # least square fit -- estimate covariance of parameters
        x0 = list(fitter.bestfit[fitter.free_params].values)
        if vary_background:
            x0.append(0) # add background level
        popt,pcov,infodict,mesg,ier = leastsq(lstsq_func,x0,full_output=True)

        # scale 
        cost = np.sum(infodict['fvec'] ** 2)
        ysize = len(infodict['fvec'])
        s_sq = cost / (ysize - popt.size)
        pcov = pcov * s_sq

        # result
        for i,p in enumerate(fitter.free_params):
            print(f'{p:10}: {popt[i]:6.3f} +/- {np.sqrt(pcov[i,i]):.3f}')
        if vary_background:
            print(f'{"background":10}: {popt[-1]:.3f} +/- {np.sqrt(pcov[-1,-1]):.3f}')
            
        # generate magnitudes
        p_samples = multivariate_normal(mean=popt,cov=pcov,allow_singular=True).rvs(size=N_bootstrap)
        mags = []
        for p in tqdm(p_samples):
            _img = fitter.calc_img(fitter.standardize_params(p),*fitter.fitter_args)
            mag = self.calc_mag(_img.sum())
            mags.append(mag) 
        print(f'mag: {np.mean(mags):.3f} +/- {np.std(mags):.3f}')
        
        self.mag = np.mean(mags)
        self.mag_err = np.std(mags)
        
        
class ModelFitter():
    def __init__(self,data,err,model,n_norm=10,verbose=True):
        self.data = data
        self.err = err
        self.model = model
        self.img_shape = data.shape
        self.param_names = model._param_names
        self.param_names_arr = np.array(self.param_names)
        self.n_norm = n_norm
        self.verbose = verbose
        
    def standardize_params(self,params):
        ''' Process parameterds to avoide numerical errors
        '''
        standardized_params = []
        for val,name in zip(params,self.free_params):
            if 'amplitude' in name:
                _val = np.log10(val)
            elif 'background' in name:
                _val = val #/ self.bkg_norm
            elif 'x' in name:
                _val = val / self.img_shape[0]
            elif 'y' in name:
                _val = val / self.img_shape[1]
            elif 'n' in name:
                _val = val / self.n_norm
            elif 'r_eff' in name:
                _val = val / self.img_shape[0]
            else:
                _val = val
            standardized_params.append(_val)
        return standardized_params

    def unstandardize_params(self,standardized_params):
        params = []
        for val,name in zip(standardized_params,self.free_params):
            if 'amplitude' in name:
                _val = 10**val
            elif 'background' in name:
                _val = val #* self.bkg_norm
            elif 'x' in name:
                _val = val * self.img_shape[0]
            elif 'y' in name:
                _val = val * self.img_shape[1]
            elif 'n' in name:
                _val = val * self.n_norm
            elif 'r_eff' in name:
                _val = val * self.img_shape[0]
            else:
                _val = val
            params.append(_val)
        return params    

    def fit_helper(self,*args,**kwargs):
        '''function to be minimized -- chi square'''
        chi2 = self.calc_chi2(*args,**kwargs)
        if self.verbose and not kwargs.get('quiet',False):
            print(f'\r  chi2 = {chi2:.5e}'.ljust(10),end='',flush=True)
        return chi2

    def calc_chi2(self,standardized_params,fixed_params,fixed_idx,free_idx,param_template,**kwargs):
        ''' calculate reduced chi square'''
        _img = self.calc_img(standardized_params,fixed_params,fixed_idx,free_idx,param_template)
        # chi2
        if self.err is not None:
            chi2 = np.nansum(((self.data - _img) / self.err)**2) 
        else:
            chi2 = np.nansum((self.data - _img)**2)
        return chi2 / np.multiply(*self.img_shape)
    
    def calc_chi(self,standardized_params,fixed_params,fixed_idx,free_idx,param_template,no_sigma=False,**kwargs):
        ''' a helper function to use scipy.optimize.leastsq 
        returns the relative residual vector (y-yfit)/sigma. '''
        _img = self.calc_img(standardized_params,fixed_params,fixed_idx,free_idx,param_template)
        # chi2
        if (self.err is not None) and not no_sigma:
            chi = (self.data - _img) / self.err
        else:
            chi = self.data - _img
        return chi
    
    def calc_img(self,standardized_params,fixed_params,fixed_idx,free_idx,param_template):
        ''' a helper function to calculate the model image '''
        # unpack params & merge fixed parameters
        params_partial = self.unstandardize_params(standardized_params)
        params = param_template
        params[free_idx] = params_partial
        params[fixed_idx] = list(fixed_params.values())
        
        # set parameter & evaluate model
        self.model.parameters = params
        _img = model_to_image(self.model, size=self.img_shape)
        return _img
            
    def fit_model(self,bounds,N_iter=4,rtol_init=1e-1,rtol_iter=1e-2,xrtol=1,
                  fixed_params={},method='Nelder-Mead'):
        '''
        inputs:
            rtol_init: relative tolerance to the starting value during the initial fit
            rtol_iter: relative tolerance to the previous fit result during iteration
        '''
        # handle free and fixed parameters
        names = self.param_names
        fixed_idx = [names.index(name) for name in fixed_params.keys()]
        free_idx = [name_idx for name_idx, name in enumerate(names) if name not in fixed_params.keys()]
        param_template = np.zeros(len(names)) * np.nan
        bounds = bounds[free_idx]
          
        self.fitter_args = (fixed_params,fixed_idx,free_idx,param_template)
        self.free_params = np.array(names)[free_idx]
        self.fixed_params = fixed_params
        logger.debug('Free params: '+str(self.free_params))
        logger.debug('Fixed params: '+str(self.fixed_params))

        # initial guess
        x0_physical = self.model.parameters[free_idx]
        x0 = self.standardize_params(x0_physical)
        xatol = max(xrtol * np.abs(x0))
        
        # initial fit
        logger.debug('x0 ='+str(x0_physical))
        logger.debug('x0 (standardized) = '+str(x0))
        logger.debug('bounds ='+str(bounds))
        init_chi2 = self.fit_helper(x0,*self.fitter_args,quiet=True)
        init_fatol = init_chi2 * rtol_init
        logger.info(f'* Starting profile-fitting iteration 1/{N_iter}... (fatol={init_fatol:.2e})')
        res = minimize(self.fit_helper,
                       x0 = x0,
                       bounds = bounds,
                       method = method,
                       args = self.fitter_args,
                       options = dict(maxfev=100,fatol=init_fatol,xatol=xatol))

        # iterative fit 
        # this process helps breaking local minima 
        # -- more effective than continuing the initial fit
        for i in range(N_iter - 1):
            # update tolerance and repeat fit
            x0 = res.x
            iter_chi2 = self.fit_helper(x0,*self.fitter_args)
            iter_fatol = iter_chi2 * rtol_iter
            logger.info(f'* Starting profile-fitting iteration {i+2}/{N_iter}... (fatol={iter_fatol:.2e})')
            res = minimize(self.fit_helper,
                           x0 = x0,
                           bounds = bounds,
                           method = method,
                           args = self.fitter_args,
                           options = dict(maxfev=5000,fatol=iter_fatol,xatol=xatol))

        params_partial = self.unstandardize_params(res.x)
        params = param_template
        params[free_idx] = params_partial
        params[fixed_idx] = list(self.fixed_params.values())
        
        self.fit_res = res
        self.bestfit_params = params
        self.bestfit = pd.Series(data=self.bestfit_params,index=self.param_names)
        logger.info('* best-fit parameters:\n'+str(self.bestfit))
            
        # evaluate best-fit image
        self.model.parameters = self.bestfit_params
        self.bestfit_img = model_to_image(self.model,size=self.img_shape)
        self.residual = self.data - self.bestfit_img
    
def plot_sersicfit_result(data,_img):
    ''' plot the sersic profile fit result
    Args:
        data (2d array): the original data
        _img (2d array): the fitted model image
    '''
    _res = data - _img
    fig = plt.figure(figsize=(10,10))
    ax_img = fig.add_axes([0.1,0.1,0.4,0.4])
    ax_top = fig.add_axes([0.1,0.5,0.4,0.25])
    ax_right = fig.add_axes([0.5,0.1,0.25,0.4])
    ax_hist = fig.add_axes([0.4,0.6,0.6,0.2])
    ax_model = fig.add_axes([0.62,0.3,0.12,0.3])

    ax_res = fig.add_axes([0.85,0.1,0.4,0.4])
    ax_top2 = fig.add_axes([0.85,0.5,0.4,0.25])
    ax_left2 = fig.add_axes([0.6,0.1,0.25,0.4])

    norm,offset = astroplot(data,ax=ax_img,percentiles=[0.01,99.9],cmap='viridis')
    astroplot(_res,ax=ax_res,norm=norm,offset=offset,cmap='viridis')
    astroplot(_img,ax=ax_model,norm=norm,offset=offset,cmap='viridis')

    for row in data:
        ax_top.plot(row+1,c='k',alpha=0.2)
    ax_top.plot(np.nanmax(_img+1,axis=0),color='r',lw=2,ls=':')
    ax_top.set_xlim(0,data.shape[0])
    ax_top.set_ylim(np.nanpercentile(data,2)+1,)
    ax_top.set_yscale('log')
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.axis('off')

    for row in data.T:
        ax_right.plot(row+1,np.arange(len(row)),c='k',alpha=0.2)
    ax_right.plot(np.nanmax(_img+1,axis=1),np.arange(len(row)),color='r',lw=2,ls=':')
    ax_right.set_ylim(0,data.shape[0])
    ax_right.set_xlim(np.nanpercentile(data,2)+1,)
    ax_right.set_xscale('log')
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.axis('off')

    count_bins = np.linspace(*np.nanpercentile(_res,[0.1,99.9]),100)
    ax_hist.hist(data_annulus-np.nanmean(data_annulus),bins=count_bins,color='lightgray',lw=2,label='data (annulus)',density=True)
    # ax_hist.plot(count_bins,lik_func(count_bins),c='gray',lw=2,label='likelihood model')
    ax_hist.hist(data.flatten(),bins=count_bins,color='k',histtype='step',lw=2,density=True,label='data (total)');
    ax_hist.hist(_res.flatten(),bins=count_bins,color='tomato',histtype='step',lw=2,density=True,label='fit residual');
    ax_hist.legend(frameon=False,fontsize=13)
    ax_hist.set_yscale('log')
    # ax_hist.set_ylim(1e-3,lik_func(count_bins).max()*1.3)
    ax_hist.tick_params(direction='in')
    ax_hist.set_yticks([])
    ax_hist.tick_params(direction='in',which='both',left=False)
    ax_hist.set_ylabel('count density',fontsize=13)
    ax_hist.set_xlabel('pixel value [MJy/sr]',fontsize=13)

    for row in _res:
        ax_top2.plot(row+1,c='k',alpha=0.2)
    ax_top2.set_xlim(ax_top.get_xlim())
    ax_top2.set_ylim(ax_top.get_ylim())
    ax_top2.set_yscale('log')
    ax_top2.set_xticks([])
    ax_top2.set_yticks([])
    ax_top2.axis('off')

    for row in _res.T:
        ax_left2.plot(row+1,np.arange(len(row)),c='k',alpha=0.2)
    ax_left2.set_ylim(ax_right.get_ylim())
    ax_left2.set_xlim(ax_right.get_xlim())
    ax_left2.invert_xaxis()
    ax_left2.set_xscale('log')
    ax_left2.set_xticks([])
    ax_left2.set_yticks([])
    ax_left2.axis('off')