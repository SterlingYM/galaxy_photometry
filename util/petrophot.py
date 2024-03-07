from astropy.stats import sigma_clip, sigma_clipped_stats
import numpy as np
from scipy.optimize import minimize
from photutils.aperture import aperture_photometry, CircularAperture, CircularAnnulus, ApertureStats
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def subtract_bkg(img,sigma=3):
    try:
        clipped_img = sigma_clip(img, sigma=sigma)
        img_bgsub = img - np.nanpercentile(clipped_img,50)
    except Exception as e:
        print('ERROR: clipped image is invalid')
        plt.imshow(clipped_img,origin='lower')
        plt.show()
        raise e
    plt.imshow(clipped_img,origin='lower')
    plt.show()
    return img_bgsub
        
def find_center(img,r):
    ''' minimize negative sum (i.e., maximize sum) within r.'''
    x0 = [img.shape[0]/2, img.shape[1]/2]
    bounds = [[0,img.shape[0]],[0,img.shape[1]]]
    
    def helper(params):
        x,y = params
        indices = np.indices(img.shape)
        x_indices = indices[1]
        y_indices = indices[0]
        s = (x_indices - x)**2 + (y_indices - y)**2 <= r
        return -1 * np.nansum(img[s])
        
    res = minimize(helper,x0=x0,bounds=bounds,method='Nelder-Mead')
    return res

def calc_petro(r_list,area_list,flux_list):
    center_list = (r_list[1:]+r_list[:-1])/2
    sb_annulus = (flux_list[1:]-flux_list[:-1])/(area_list[1:] - area_list[:-1])
    sb_encircled = (flux_list[1:]+flux_list[:-1]) / 2 / (np.pi*center_list**2)
    petrosian_list = sb_annulus / sb_encircled
    return center_list, petrosian_list

def plot_petro(locs,petrovals,refval=0.2):
    r_petro = calc_r_petro(locs,petrovals,refval)
    plt.plot(locs,petrovals,c='k',lw=3)
    plt.axhline(refval,color='yellowgreen',ls='--',lw=3)
    plt.axvline(r_petro,color='dodgerblue',ls=':',lw=3,label=f'Petrosian radius = {r_petro:.2f}')

    plt.ylabel('Petrosian ratio',fontsize=16)
    plt.xlabel('radius [pix]',fontsize=16)
    plt.legend(frameon=False,fontsize=15)

def calc_r_petro(locs,petrovals,refval=0.2):
    r_petro = locs[np.argmin(np.abs(petrovals-refval))]
    return r_petro

def do_photometry(img,r_eff,position,aperture_factor=2,r_in_factor=3,r_out_factor=5,plot=True):
    # handle negative values
    if plot:
        vmin,vmax = np.nanpercentile(img.data,[0.1,99.9])
        if vmin <= 0:
            offset = -vmin + 1e-1
            vmin += offset
            vmax += offset
        else:
            offset = 0
    
    # Petrosian radius
    _aperture = CircularAperture(position, r=r_eff)

    # aperture
    aperture = CircularAperture(position, r=r_eff*aperture_factor)
    aper_stats = ApertureStats(img, aperture, sigma_clip=None)

    # background
    annulus_aperture = CircularAnnulus(position, 
                                       r_in = r_eff * r_in_factor, 
                                       r_out = r_eff * r_out_factor)
    bkg_stats = ApertureStats(img, annulus_aperture)
    bkg_mean = bkg_stats.mean
    
    # plot
    if plot:
        fig,ax = plt.subplots(1,1,figsize=(8,8))
        ax.imshow(img.data+offset,norm=LogNorm(vmin=vmin,vmax=vmax),origin='lower')
        _aperture.plot(ax=ax,color='k',label='Petrosian radius')
        aperture.plot(ax=ax,color='w',label='Aperture')
        annulus_aperture.plot(ax=ax,color='r',label='Background Annulus')
        plt.legend(fontsize=15)
    
    # calculate total counts
    bkg_total = aperture.area * bkg_mean
    counts_total = aper_stats.sum
    effective_total = counts_total - bkg_total
    return counts_total, bkg_total, effective_total