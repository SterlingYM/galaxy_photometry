import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import logging

from scipy.optimize import minimize
from petrofit.modeling import fit_background, model_to_image

logging.basicConfig(format='%(levelname)s:sphot: %(message)s', level=logging.DEBUG)

def subtract_background(data,sigma=3.0,title='',plot=True):
    ''' Note the main purpose of this is to reduce gradient, not to actually set background level to zero. Constant background level will be fitted with the model '''
    bg_model, fitter = fit_background(data, sigma=sigma)
    bg_image = model_to_image(bg_model, 
                              size=data.shape)
    image_bgsub = data - bg_image

    # plot
    if plot:
        fig, axes = plt.subplots(1,3,figsize=(13,4))
        vmin,vmax = np.nanpercentile(data,[0.1,99.9])
        if vmin <= 0:
            offset = - vmin + 1e-1
            vmin = 1e-1
            vmax += offset
        else:
            offset = 0
        norm = LogNorm(vmin=vmin,vmax=vmax)
        plt.subplots_adjust(wspace=0.03)
        axes[0].imshow(data+offset, norm=norm, origin='lower')
        axes[1].imshow(bg_image+offset, origin='lower')
        axes[2].imshow(image_bgsub+bg_image.mean()+offset, norm=norm, origin='lower')
        axes[0].set_title("Original: "+title)
        axes[1].set_title("Background gradient")
        axes[2].set_title("Background gradient subtracted")
        axes[1].set_yticklabels([])
        axes[2].set_yticklabels([])
        
    return image_bgsub

class ModelFitter():
    def __init__(self,data,err,model,n_norm=10,verbose=True):
        self.data = data
        self.err = err
        self.model = model
        self.img_shape = data.shape
        self.param_names = [*model._param_names,'background']
        self.param_names_arr = np.array(self.param_names)
        self.n_norm = n_norm
        self.verbose = verbose
        self.bkg_norm = np.nanmean(data)
        
    def standardize_params(self,params):
        ''' Process parameterds to avoide numerical errors
        '''
        standardized_params = []
        for val,name in zip(params,self.free_params):
            if 'amplitude' in name:
                _val = np.log10(val)
            elif 'background' in name:
                _val = val / self.bkg_norm
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
                _val = val * self.bkg_norm
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
        if self.verbose:
            print(f'\r  chi2 = {chi2:.5e}'.ljust(10),end='',flush=True)
        return chi2

    def calc_chi2(self,standardized_params,fixed_params,fixed_idx,free_idx,param_template):
        ''' calculate reduced chi square'''
        # unpack params & merge fixed parameters
        params_partial = self.unstandardize_params(standardized_params)
        params = param_template
        params[free_idx] = params_partial
        params[fixed_idx] = list(fixed_params.values())
        
        # set parameter & evaluate model
        self.model.parameters = params[:-1]
        _img = model_to_image(self.model, size=self.img_shape, mode='oversample') + params[-1]

        # chi2
        if self.err is not None:
            chi2 = np.nansum(((self.data - _img) / self.err)**2) 
        else:
            chi2 = np.nansum((self.data - _img)**2)
        return chi2 / np.multiply(*img.shape)

    # def predict_Y(self,standardized_params,fixed_params,fixed_idx,free_idx,param_template):
    #     ''' calculate chi square'''
    #     # unpack params & merge fixed parameters
    #     params_partial = self.unstandardize_params(standardized_params)
    #     params = param_template
    #     params[free_idx] = params_partial
    #     params[fixed_idx] = list(fixed_params.values())
        
    #     # set parameter & evaluate model
    #     self.model.parameters = params[:-1]
    #     _img = model_to_image(self.model, size=self.img_shape, mode='oversample') + params[-1]
    #     return _img       
    
    # def log_prior(self,theta,bounds):
    #     for param,bound in zip(theta,bounds):
    #         if (param < bound[0]) or (param > bound[1]):
    #             return -np.inf
    #     return 0

    # def prep_loglik(self,bounds,fixed_params={}):
    #     # handle free and fixed parameters
    #     names = self.param_names
    #     fixed_idx = [names.index(name) for name in fixed_params.keys()]
    #     free_idx = [name_idx for name_idx, name in enumerate(names) if name not in fixed_params.keys()]
    #     param_template = np.zeros(len(names)) * np.nan
    #     bounds_freeparam = bounds[free_idx]
          
    #     self.fitter_args = (fixed_params,fixed_idx,free_idx,param_template)
    #     self.free_params = np.array(names)[free_idx]
    #     self.fixed_params = fixed_params

    #     # initial guess
    #     x0_physical = [*self.model.parameters[free_idx[:-1]],np.nanmean(self.data)]
    #     x0 = self.standardize_params(x0_physical)

    #     predict_Y = lambda theta: self.predict_Y(theta,*self.fitter_args)
    #     Y = self.data
    #     loglik = lambda theta: -0.5*self.calc_chi2(theta,*self.fitter_args)
    #     logp = lambda theta: self.log_prior(theta,bounds_freeparam)

    #     return predict_Y,Y,loglik,logp, self.free_params,x0,bounds_freeparam

    def fit_model(self,bounds,N_iter=4,rtol_init=1e-1,rtol_iter=1e-2,
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
        logging.info('Free params: '+str(self.free_params))
        logging.info('Fixed params: '+str(self.fixed_params))

        # initial guess
        x0_physical = [*self.model.parameters[free_idx[:-1]],np.nanmean(self.data)]
        x0 = self.standardize_params(x0_physical)
        
        # initial fit
        logging.info(f'Starting iteration 1/{N_iter}...')
        logging.info('x0 ='+str(x0_physical))
        logging.info('x0 (standardized) = '+str(x0))
        logging.info('bounds ='+str(bounds))
        init_chi2 = self.fit_helper(x0,*self.fitter_args)
        init_fatol = init_chi2 * rtol_init
        res = minimize(self.fit_helper,
                       x0 = x0,
                       bounds = bounds,
                       method = method,
                       args = self.fitter_args,
                       options = dict(maxfev=100,fatol=init_fatol))

        # iterative fit 
        # this process helps breaking local minima 
        # -- more effective than continuing the initial fit
        for i in range(N_iter - 1):
            # update tolerance and repeat fit
            logging.info('\n* '+f'Starting iteration {i+2}/{N_iter}...')
            x0 = res.x
            iter_chi2 = self.fit_helper(x0,*self.fitter_args)
            iter_fatol = iter_chi2 * rtol_iter
            res = minimize(self.fit_helper,
                           x0 = x0,
                           bounds = bounds,
                           method = method,
                           args = self.fitter_args,
                           options = dict(maxfev=5000,fatol=iter_fatol))

        params_partial = self.unstandardize_params(res.x)
        params = param_template
        params[free_idx] = params_partial
        params[fixed_idx] = list(self.fixed_params.values())
        
        self.fit_res = res
        self.bestfit_params = params
        self.bestfit = pd.Series(data=self.bestfit_params,index=self.param_names)
        logging.info('bestfit:'+str(self.bestfit))
            
        # evaluate best-fit image
        self.model.parameters = self.bestfit_params[:-1]
        self.bestfit_img = model_to_image(self.model,size=self.img_shape)
        self.bestfit_img_total = self.bestfit_img + self.bestfit_params[-1]
        self.residual = self.data - self.bestfit_img
    
    def plot_result(self,vmin=None,vmax=None,cmap='viridis',title=''):
        if vmin is None and vmax is None:
            vmin,vmax = np.nanpercentile(self.data,[0.1,99.])
        elif vmin is None:
            vmin = self.data.min()
        elif vmax is None:
            vmax = self.bestfit_img.max()
            
        # sanity check
        if vmin <= 0:
            offset = - vmin + 1e-1
        else:
            offset = 0
        vmin += offset
        vmax += offset
        norm = LogNorm(vmin=vmin,vmax=vmax,clip=True)            
            
        
        fig,axes = plt.subplots(1,3,figsize=(15,5))
        plt.subplots_adjust(wspace=0.03,bottom=0.1)
        
        im1 = axes[0].imshow(self.data+offset,
                             norm=norm,origin='lower',
                             cmap=cmap)
        im2 = axes[1].imshow(self.bestfit_img_total+offset,
                             norm=norm,origin='lower',
                             cmap=cmap)
        im3 = axes[2].imshow(self.residual+offset,
                             norm=norm,origin='lower',
                             cmap=cmap)
        
        # for ax,im in zip(axes,[im1,im2,im3]):
        #     bounds = ax.get_position().bounds
        #     cax = fig.add_axes([bounds[0],0,bounds[2],0.05])
        #     plt.colorbar(im,cax=cax,orientation='horizontal')
        if title != '':
            axes[0].set_title('Original: '+title)
        else:
            axes[0].set_title('Original image')
        axes[1].set_title('Best-fit model')
        axes[2].set_title('Residual')
        for ax in axes[1:]:
            ax.set_yticklabels([])
            
        plt.show()
        return fig,axes
    
def Vegamag_from_HST(total_electrons,filt,exptime):
    vega_zp = {
    'F555W': 25.838,
    'F814W': 24.699,
    'F160W': 24.662
    }
    zp = vega_zp[filt]
    vegamag = -2.5 * np.log10(total_electrons / exptime) + zp
    return vegamag


