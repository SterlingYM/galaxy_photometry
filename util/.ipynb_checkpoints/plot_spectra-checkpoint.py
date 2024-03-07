def plot_spectra(model,obs,sps,pbest,ax=None):
    # flatten bestfit SED
    wav = obs['wavelength']
    wav_sim = np.linspace(wav.min(),wav.max(),10000)
    _obs = dict(
        wavelength  = wav_sim, 
        spectrum    = np.ones_like(wav_sim)*1, 
        unc         = np.ones_like(wav_sim), 
        redshift    = 0.00666,
    )
    spec_sim,_,_ = model.predict(pbest, obs=_obs, sps=sps);
    spec_sim -= 1

    # flatten data
    s = obs['spectrum']>0
    spec_data = obs['spectrum'][s]
    specerr_data = obs['unc'][s]
    wav_s = wav[s]
    _obs = dict(
        wavelength  = wav_s, 
        spectrum    = np.ones_like(wav_s), 
        unc         = np.ones_like(wav_s), 
        redshift    = 0.00666,
    )
    speccal_data = model.spec_calibration(obs=_obs, spec=spec_data)
    spec_data_flat = spec_data * speccal_data - 1
    specerr_data_flat = specerr_data * speccal_data

    # plot
    if ax is None:
        fig = plt.figure(figsize=(8,3),dpi=200)
        fig.set_facecolor('w')
        ax = fig.add_axes(0.1,0.1,0.8,0.8)
    ax.plot(wav_s,spec_data_flat,c='k',lw=1,label='data')
    ax.fill_between(wav_s,spec_data_flat-specerr_data_flat,spec_data_flat+specerr_data_flat,color='k',alpha=0.5)
    ax.plot(wav_sim,spec_sim,c='yellowgreen',ls='-',lw=2,alpha=0.9,label='bestfit model')

    ax.set_yticks([])
    ax.set_ylabel('flattened flux')
    ax.set_xlabel(r'$\lambda [\AA]$')
    ax.tick_params(direction='in')
    ax.legend(loc='upper left',frameon=False,ncol=2)