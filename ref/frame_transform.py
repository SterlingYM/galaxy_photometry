from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import interp1d
import pylab as pl
import matplotlib.pyplot as plt

def frame_tfrm_rd(mchdir, f1, f2, pair, ra_factor, figdir, log, error_cut = 0.25, init_dr=0., init_dd=0., ycut0=1.5, alpha=0.05, alpha2=0.1, symsize=1, add_scatter=False):
    log.add_line('From '+f1.replace('.rd','')+' to '+f2.replace('.rd','')+':')
    ## log is log file object define by the class in the last few lines of this file
    d2s = 3600.
    mrd1 = np.loadtxt(mchdir+f1)
    ## mrd1 & mrd2 file format
    #  id     x          y       mag      err         ra               dec
 #    1762 3237.192   13.251   16.931    0.171   184.691706601    47.393590007
 # 3003928 3234.911   32.839   16.554    0.146   184.691856479    47.393398150
 #    1761 3228.444   13.003   15.230    0.066   184.691587129    47.393537705
 # 3000135 1143.083    0.179   16.021    0.121   184.663540620    47.380646673
 # 3006468 1388.171   58.237   16.822    0.134   184.667357195    47.381649365

    mrd2 = np.loadtxt(mchdir+f2)
    idx = mrd1[:,4] < error_cut
    mrd1 = mrd1[idx]
    idx = mrd2[:,4] < error_cut
    mrd2 = mrd2[idx]
    mrd10 = copy.copy(mrd1)
    mrd20 = copy.copy(mrd2)
    med_ra = np.round(np.median(mrd1[:,5]), 5)
    med_dec = np.round(np.median(mrd1[:,6]), 5)
    ## (1) find rough initial shift
    cat1 = SkyCoord(ra = (mrd1[:,5]+init_dr)*u.degree, dec = (mrd1[:,6]+init_dd)*u.degree)
    cat2 = SkyCoord(ra = mrd2[:,5]*u.degree, dec = mrd2[:,6]*u.degree)
    idx, d2d, d3d = cat2.match_to_catalog_sky(cat1)
    mrd1 = mrd1[idx]
    dra = (mrd2[:,5] - mrd1[:,5]) * d2s
    ddec = (mrd2[:,6] - mrd1[:,6]) * d2s
    x = mrd1[:,6] - med_dec
    foo, bar, k, b = ridge_dense(x, dra, box_h=0.08/ra_factor, ycut=ycut0/ra_factor, yshift=init_dr*d2s)
    bdr0 = np.round(b, 2) / d2s
    x = mrd1[:,5] - med_ra
    foo, bar, k, b = ridge_dense(x, ddec, ycut=ycut0, yshift=init_dd*d2s)
    bdd0 = np.round(b, 2) / d2s
    ## (2) find the rough first-order solution
    mrd1 = copy.copy(mrd10)
    mrd2 = copy.copy(mrd20)
    ra1 = mrd1[:,5] + bdr0
    dec1 = mrd1[:,6] + bdd0
    cat1 = SkyCoord(ra = ra1*u.degree, dec = dec1*u.degree)
    cat2 = SkyCoord(ra = mrd2[:,5]*u.degree, dec = mrd2[:,6]*u.degree)
    idx, d2d, d3d = cat2.match_to_catalog_sky(cat1)
    mrd1 = mrd1[idx]
    dra = (mrd2[:,5] - mrd1[:,5]) * d2s
    ddec = (mrd2[:,6] - mrd1[:,6]) * d2s
    fig, ax = plt.subplots(2,1)
    a = ax[0]
    x = mrd1[:,6] - med_dec
    y = dra
    a.scatter(x, y, s=symsize, alpha=alpha)
    a.set_ylim(-1./ra_factor+bdr0*3600, 1./ra_factor+bdr0*3600)
    a.set_xlabel('Decl+offset (J2000)')
    a.set_ylabel(r'$\Delta$ RA')
    xs, ys, k, b = ridge_dense(x, y, box_h=0.08/ra_factor, ycut=ycut0/ra_factor, yshift=bdr0*d2s)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    a.plot(xc, yc, '-.', color='magenta', linewidth=1)
    kdr1 = k
    bdr1 = b
    a = ax[1]
    x = mrd1[:,5] - med_ra
    y = ddec
    a.scatter(x, y, s=symsize, alpha=alpha)
    a.set_ylim(-1.+bdd0*3600, 1.+bdd0*3600)
    a.set_xlabel('RA+offset (J2000)')
    a.set_ylabel(r'$\Delta$ Decl [sec]')
    xs, ys, k, b = ridge_dense(x, y, ycut=ycut0, yshift=bdd0*d2s)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    a.plot(xc, yc, '-.', color='magenta', linewidth=1)
    kdd1 = k
    bdd1 = b
    pl.tight_layout()
    f_fig = figdir + 'tfrm_' + pair + '_t1.png'
    pl.savefig(f_fig)
    pl.close()
    log.add_figure(f_fig, width='48%', linebreak=False)
    ## (3) fit for a second iteration
    mrd1 = copy.copy(mrd10)
    mrd2 = copy.copy(mrd20)
    ra1 = mrd1[:,5]
    dec1 = mrd1[:,6]
    mrd1[:,5] = ra1 + (kdr1 * (dec1-med_dec) + bdr1) / d2s
    mrd1[:,6] = dec1 + (kdd1 * (ra1-med_ra) + bdd1) / d2s
    cat1 = SkyCoord(ra = mrd1[:,5]*u.degree, dec = mrd1[:,6]*u.degree)
    cat2 = SkyCoord(ra = mrd2[:,5]*u.degree, dec = mrd2[:,6]*u.degree)
    idx, d2d, d3d = cat2.match_to_catalog_sky(cat1)
    mrd1 = mrd1[idx]
    dra = (mrd2[:,5] - mrd1[:,5]) * d2s
    ddec = (mrd2[:,6] - mrd1[:,6]) * d2s
    fig, ax = plt.subplots(2,1)
    a = ax[0]
    x = mrd1[:,6] - med_dec
    y = dra
    a.scatter(x, y, s=symsize, alpha=alpha)
    ylim = 0.3
    ylim2 = np.round(ylim / ra_factor,1) ## turn in to the deprojected units
    a.set_ylim(-ylim2, ylim2)
    a.set_xlabel('Decl+offset (J2000)')
    a.set_ylabel(r'$\Delta$ RA')
    xs, ys, k, b = ridge_dense(x, y, ycut=0.2, box_h=0.02/ra_factor)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    a.plot(xc, yc, '-.', color='magenta', linewidth=1)
    kdr2 = k
    bdr2 = b
    a = ax[1]
    x = mrd1[:,5] - med_ra
    y = ddec
    a.scatter(x, y, s=symsize, alpha=alpha)
    a.set_ylim(-ylim, ylim)
    a.set_xlabel('RA+offset (J2000)')
    a.set_ylabel(r'$\Delta$ Decl [sec]')
    xs, ys, k, b = ridge_dense(x, y, ycut=0.2, box_h=0.02)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    a.plot(xc, yc, '-.', color='magenta', linewidth=1)
    kdd2 = k
    bdd2 = b
    pl.tight_layout()
    f_fig = figdir + 'tfrm_' + pair + '_t2.png'
    pl.savefig(f_fig)
    pl.close()
    log.add_figure(f_fig, width='48%')
    ## (4) Final round 1, stepped clip fit similar to DAOMASTER, along perpendicular direction
    mrd1 = copy.copy(mrd10)
    mrd2 = copy.copy(mrd20)
    ra1 = mrd1[:,5]
    dec1 = mrd1[:,6]
    ra = ra1 + (kdr1 * (dec1-med_dec) + bdr1) / d2s
    dec = dec1 + (kdd1 * (ra1-med_ra) + bdd1) / d2s
    mrd1[:,5] = ra + (kdr2 * (dec-med_dec) + bdr2) / d2s
    mrd1[:,6] = dec + (kdd2 * (ra-med_ra) + bdd2) / d2s
    cat1 = SkyCoord(ra = mrd1[:,5]*u.degree, dec = mrd1[:,6]*u.degree)
    cat2 = SkyCoord(ra = mrd2[:,5]*u.degree, dec = mrd2[:,6]*u.degree)
    idx, d2d, d3d = cat2.match_to_catalog_sky(cat1)
    mrd1 = mrd1[idx]
    dra = (mrd2[:,5] - mrd1[:,5]) * d2s * ra_factor ## in the units of second
    ddec = (mrd2[:,6] - mrd1[:,6]) * d2s
    idx = (np.abs(dra) < 0.12) & (np.abs(ddec) < 0.12) ## 3 optical pixel width
    mrd1 = mrd1[idx]
    mrd2 = mrd2[idx]
    fig, ax = plt.subplots(2,1)
    a = ax[0]
    x = mrd1[:,6] - med_dec
    y = (mrd2[:,5] - mrd1[:,5]) * d2s * ra_factor
    a.scatter(x, y, s=symsize, alpha=alpha2)
    a.set_xlabel('Decl+offset (J2000)')
    a.set_ylabel(r'$\Delta$ X (RA) [sec]')
    xs, ys, k, b, dd = ridge_clip(x, y)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    kdr3 = k
    bdr3 = b
    a.plot(xc, yc, '-.', color='black', linewidth=1)
    a.plot(xc, yc+dd, '-.', color='black', linewidth=0.3)
    a.plot(xc, yc-dd, '-.', color='black', linewidth=0.3)
    a = ax[1]
    x = mrd1[:,5] - med_ra
    y = (mrd2[:,6] - mrd1[:,6]) * d2s
    a.scatter(x, y, s=symsize, alpha=alpha2)
    a.set_xlabel('RA+offset (J2000)')
    a.set_ylabel(r'$\Delta$ Y (Dec) [sec]')
    xs, ys, k, b, dd = ridge_clip(x, y)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    kdd3 = k
    bdd3 = b
    a.plot(xc, yc, '-.', color='black', linewidth=1)
    a.plot(xc, yc+dd, '-.', color='black', linewidth=0.3)
    a.plot(xc, yc-dd, '-.', color='black', linewidth=0.3)
    pl.tight_layout()
    f_fig = figdir + 'tfrm_' + pair + '_t3.png'
    pl.savefig(f_fig)
    pl.close()
    log.add_figure(f_fig, width='33%', linebreak=False)
    ## (5) Final round 2, stepped clip fit similar to DAOMASTER, along parallel direction
    mrd1 = copy.copy(mrd10)
    mrd2 = copy.copy(mrd20)
    ra1 = mrd1[:,5]
    dec1 = mrd1[:,6]
    ra = ra1 + (kdr1 * (dec1-med_dec) + bdr1) / d2s
    dec = dec1 + (kdd1 * (ra1-med_ra) + bdd1) / d2s
    ra1 = ra + (kdr2 * (dec-med_dec) + bdr2) / d2s
    dec1 = dec + (kdd2 * (ra-med_ra) + bdd2) / d2s
    mrd1[:,5] = ra1 + (kdr3 * (dec1-med_dec) + bdr3) / d2s / ra_factor
    mrd1[:,6] = dec1 + (kdd3 * (ra1-med_ra) + bdd3) / d2s
    cat1 = SkyCoord(ra = mrd1[:,5]*u.degree, dec = mrd1[:,6]*u.degree)
    cat2 = SkyCoord(ra = mrd2[:,5]*u.degree, dec = mrd2[:,6]*u.degree)
    idx, d2d, d3d = cat2.match_to_catalog_sky(cat1)
    mrd1 = mrd1[idx]
    dra = (mrd2[:,5] - mrd1[:,5]) * d2s * ra_factor ## in the units of second
    ddec = (mrd2[:,6] - mrd1[:,6]) * d2s
    idx = (np.abs(dra) < 0.12) & (np.abs(ddec) < 0.12) ## 3 optical pixel width
    mrd1 = mrd1[idx]
    mrd2 = mrd2[idx]
    fig, ax = plt.subplots(2,1)
    a = ax[0]
    x = mrd1[:,5] - med_ra
    y = (mrd2[:,5] - mrd1[:,5]) * d2s * ra_factor
    a.scatter(x, y, s=symsize, alpha=alpha2)
    a.set_xlabel('RA+offset (J2000)')
    a.set_ylabel(r'$\Delta$ X (RA) [sec]')
    xs, ys, k, b, dd = ridge_clip(x, y)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    kdr4 = k
    bdr4 = b
    a.plot(xc, yc, '-.', color='black', linewidth=1)
    a.plot(xc, yc+dd, '-.', color='black', linewidth=0.3)
    a.plot(xc, yc-dd, '-.', color='black', linewidth=0.3)
    a = ax[1]
    x = mrd1[:,6] - med_dec
    y = (mrd2[:,6] - mrd1[:,6]) * d2s
    a.scatter(x, y, s=symsize, alpha=alpha2)
    a.set_xlabel('Decl+offset (J2000)')
    a.set_ylabel(r'$\Delta$ Y (Dec) [sec]')
    xs, ys, k, b, dd = ridge_clip(x, y)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    kdd4 = k
    bdd4 = b
    a.plot(xc, yc, '-.', color='black', linewidth=1)
    a.plot(xc, yc+dd, '-.', color='black', linewidth=0.3)
    a.plot(xc, yc-dd, '-.', color='black', linewidth=0.3)
    pl.tight_layout()
    f_fig = figdir + 'tfrm_' + pair + '_t4.png'
    pl.savefig(f_fig)
    pl.close()
    log.add_figure(f_fig, width='33%', linebreak=False)
    
    fmt_sol = '%5i' + '%15.9f'*4 + '%9s%10.4f%13.6f%13.6f\n'
    f_sol = mchdir + 'tfrm_' + pair + '.dat'
    h_sol = open(f_sol, 'w')
    h_sol.write('  Iter        kdr            bdr            kdd           bdd        units   ra_factor   med_ra       med_dec\n')
    h_sol.write(fmt_sol % (0, 0, bdr0, 0, bdd0, '0', ra_factor, med_ra, med_dec))
    h_sol.write(fmt_sol % (1, kdr1, bdr1, kdd1, bdd1, 'proj', ra_factor, med_ra, med_dec))
    h_sol.write(fmt_sol % (2, kdr2, bdr2, kdd2, bdd2, 'proj', ra_factor, med_ra, med_dec))
    h_sol.write(fmt_sol % (3, kdr3, bdr3, kdd3, bdd3, 'deproj', ra_factor, med_ra, med_dec))
    h_sol.write(fmt_sol % (4, kdr4, bdr4, kdd4, bdd4, 'deproj', ra_factor, med_ra, med_dec))
    h_sol.close()
    
    ## (5) Final check
    mrd1 = copy.copy(mrd10)
    mrd2 = copy.copy(mrd20)
    cat2 = SkyCoord(ra = mrd2[:,5]*u.degree, dec = mrd2[:,6]*u.degree)
    sol = ascii.read(f_sol)
    mrd1[:,5], mrd1[:,6] = frame_tfrm_apply(mrd1[:,5], mrd1[:,6], sol)
    cat1 = SkyCoord(ra=mrd1[:,5]*u.degree, dec=mrd1[:,6]*u.degree)
    idx, d2d, d3d = cat2.match_to_catalog_sky(cat1)
    mrd1 = mrd1[idx]
    dra = (mrd2[:,5] - mrd1[:,5]) * d2s * ra_factor ## in the units of second
    ddec = (mrd2[:,6] - mrd1[:,6]) * d2s
    idx = (np.abs(dra) < 0.12) & (np.abs(ddec) < 0.12) ## 3 optical pixel width
    mrd1 = mrd1[idx]
    mrd2 = mrd2[idx]
    fig, ax = plt.subplots(2,1)
    a = ax[0]
    x = mrd1[:,5]
    y = (mrd2[:,5] - mrd1[:,5]) * d2s * ra_factor
    a.scatter(x, y, s=symsize, alpha=alpha2)
    a.set_xlabel('RA (J2000)')
    a.set_ylabel(r'$\Delta$ X (RA) [sec]')
    xs, ys, k, b, dd = ridge_clip(x, y)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    a.plot(xc, yc, '-.', color='black', linewidth=1)
    a.plot(xc, yc*0., '-', color='red', linewidth=0.5, alpha=0.7)
    a.plot(xc, yc+dd, '-.', color='black', linewidth=0.3)
    a.plot(xc, yc-dd, '-.', color='black', linewidth=0.3)
    if add_scatter:
        from astropy.stats import sigma_clipped_stats
        st_mean, st_median, st_std = sigma_clipped_stats(y, maxiters=20, sigma=2.6)
        std_pix = st_std / 0.04
        a.text(0.02, 0.9, 'Scatter = '+str(np.round(std_pix,2))+' pixel, 0.04"',
                   transform = a.transAxes, color='red')
        a.plot(xc, xc*0+st_std, color='red')
        a.plot(xc, xc*0-st_std, color='red')
    a = ax[1]
    x = mrd1[:,6]
    y = (mrd2[:,6] - mrd1[:,6]) * d2s
    a.scatter(x, y, s=symsize, alpha=alpha2)
    a.set_xlabel('Dec (J2000)')
    a.set_ylabel(r'$\Delta$ Y (Dec) [sec]')
    xs, ys, k, b, dd = ridge_clip(x, y)
    xc = np.linspace(np.min(x), np.max(x), 5)
    yc = k * xc + b
    a.plot(xc, yc, '-.', color='black', linewidth=1)
    a.plot(xc, yc*0., '-', color='red', linewidth=0.5, alpha=0.7)
    a.plot(xc, yc+dd, '-.', color='black', linewidth=0.3)
    a.plot(xc, yc-dd, '-.', color='black', linewidth=0.3)
    if add_scatter:
        from astropy.stats import sigma_clipped_stats
        st_mean, st_median, st_std = sigma_clipped_stats(y, maxiters=20, sigma=2.6)
        std_pix = st_std / 0.04
        a.text(0.02, 0.9, 'Scatter = '+str(np.round(std_pix,2))+' pixel, 0.04"',
                   transform = a.transAxes, color='red')
        a.plot(xc, xc*0+st_std, color='red')
        a.plot(xc, xc*0-st_std, color='red')
    pl.tight_layout()
    f_fig = figdir + 'tfrm_' + pair + '_check.png'
    pl.savefig(f_fig)
    pl.close()
    log.add_figure(f_fig, width='33%', linebreak=False)
    log.add_table(sol)
    return(sol)

def frame_tfrm_apply(ra1, dec1, sol):
    med_ra = sol['med_ra'][0]
    med_dec = sol['med_dec'][0]
    d2s = 3600.
    ra = ra1 + (sol['kdr'][1] * (dec1-med_dec) + sol['bdr'][1]) / d2s
    dec = dec1 + (sol['kdd'][1] * (ra1-med_ra) + sol['bdd'][1]) / d2s
    ra1 = ra + (sol['kdr'][2] * (dec-med_dec) + sol['bdr'][2]) / d2s
    dec1 = dec + (sol['kdd'][2] * (ra-med_ra) + sol['bdd'][2]) / d2s
    ra = ra1 + (sol['kdr'][3] * (dec1-med_dec) + sol['bdr'][3]) / d2s / sol['ra_factor'][3]
    dec = dec1 + (sol['kdd'][3] * (ra1-med_ra) + sol['bdd'][3]) / d2s
    ra1 = ra + (sol['kdr'][4] * (ra-med_ra) + sol['bdr'][4]) / d2s / sol['ra_factor'][3]
    dec1 = dec + (sol['kdd'][4] * (dec-med_dec) + sol['bdd'][4]) / d2s
    return(ra1, dec1)

def ridge_dense(x, y, box_w = 0.0015, box_h = 0.08, nsamp = 100, nsampy = 100, ycut=1.0, yshift=0.):
    import numpy as np
    idx = np.abs(y-yshift) < ycut
    x = x[idx]
    y = y[idx]
    x1 = np.min(x) + box_w * 2.
    x2 = np.max(x) - box_w * 2.
    xs = np.linspace(x1, x2, nsamp+1)
    xmids = (xs[0:nsamp] + xs[1:(nsamp+1)]) * 0.5
    ymids = xmids * 0.
    for i in range(nsamp):
        xmid = xmids[i]
        idx = (x >= xmid - box_w / 2.) & (x <= xmid + box_w / 2.)
        if sum(idx) > 0:
            yp = y[idx]
            density = np.zeros(nsampy)
            y1 = np.min(yp)
            y2 = np.max(yp)
            ys = np.linspace(y1, y2, nsampy+1)
            for j in range(nsampy):
                idx = (yp > ys[j] - box_h/2.) & (yp < ys[j] + box_h/2.)
                density[j] = len(np.where(idx)[0])
            idx = np.argmax(density)
            ymids[i] = ys[idx]
        else:
            ymids[i] = -99.99
    idx = ymids != -99.99
    X = np.zeros((sum(idx), 2))
    X[:,0] = xmids[idx]
    X[:,1] = 1
    Y = ymids[idx]
    tX = np.transpose(X)
    beta = np.dot(np.linalg.inv(np.dot(tX, X)), np.dot(tX, Y))
    return(xmids, ymids, beta[0], beta[1])

def ridge_clip(x, y):
    by = -0.05
    pixel_clip = np.arange(3, 0.5+by, by)
    z = np.polyfit(x, y, 1)
    res = np.abs(y - z[0]*x - z[1])
    for rad in pixel_clip:
        rad_deg = rad * 0.04 ## optical pixel scale
        idx = res < rad_deg
        x = x[idx]
        y = y[idx]
        z = np.polyfit(x, y, 1)
        res = np.abs(y - z[0]*x - z[1])
    return(x, y, z[0], z[1], rad_deg)
 
class HtmlLog:
    """Writing a log file in HTML format"""
    def __init__(self, logdir, LogName):
        self.logdir = logdir
        self.LogName = LogName
        return(None)

    def add_line(self, t, AppendLog=True):
        t = '<h3>' + t + ' <font color="gray" size=2>('
        t = t + datetime.datetime.now().strftime('Local %Y-%m-%d, %H:%M:%S')
        t = t + ')</font></h3>'
        f = self.logdir + self.LogName
        operation = 'a' if AppendLog else 'w'
        h = open(f, operation)
        h.write(t)
        h.close()
        return(0)

    def add_text(self, t, AppendLog=True):
        f = self.logdir + self.LogName
        operation = 'a' if AppendLog else 'w'
        h = open(f, operation)
        h.write(t+' <br>')
        h.close()
        return(0)

    def add_table(self, t, AppendLog=True, max_lines=20, full=False):
        if full:
            max_lines = len(t) + 10
        f = self.logdir + self.LogName
        t2 = t.pformat(html=True, max_lines=max_lines, max_width=5e3)
        t3 = str(t2).replace('[','').replace(']','').replace("\',",'').replace("\'",'')
        t3 = t3.replace('<td>','<td style="padding: 0 8 0 8;text-align:center;">')
        t3 = t3.replace('<table','<table style="border:1px solid black; border-collapse:collapse;"')
        operation = 'a' if AppendLog else 'w'
        h = open(f, operation)
        h.write(t3)
        h.close()
        return(0)

    def add_figure(self, f_fig, AppendLog=True, width='50%', linebreak=True):
        sf_fig = os.path.basename(f_fig)
        t = '<img src="./figs/'+sf_fig+'" width='+width+'>'
        if linebreak: t = t + '<br>'
        f = self.logdir + self.LogName
        operation = 'a' if AppendLog else 'w'
        h = open(f, operation)
        h.write(t)
        h.close()
        return(0)
