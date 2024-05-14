import matplotlib.pyplot as plt
import numpy as np
from astropy.visualization import simple_norm
from photutils.datasets import make_noise_image, make_test_psf_data
from photutils.detection import DAOStarFinder
from photutils.psf import IntegratedGaussianPRF, PSFPhotometry

psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
psf_shape = (9, 9)
nsources = 10
shape = (101, 101)
data, true_params = make_test_psf_data(shape, psf_model, psf_shape,
                                       nsources, flux_range=(500, 700),
                                       min_separation=10, seed=0)
noise = make_noise_image(data.shape, mean=0, stddev=1, seed=0)
data += noise
error = np.abs(noise)

psf_model = IntegratedGaussianPRF(flux=1, sigma=2.7 / 2.35)
fit_shape = (5, 5)
finder = DAOStarFinder(6.0, 2.0)
psfphot = PSFPhotometry(psf_model, fit_shape, finder=finder,
                        aperture_radius=4)
phot = psfphot(data, error=error)

resid = psfphot.make_residual_image(data, (9, 9))

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
norm = simple_norm(data, 'sqrt', percent=99)
ax[0].imshow(data, origin='lower', norm=norm)
ax[1].imshow(data - resid, origin='lower', norm=norm)
im = ax[2].imshow(resid, origin='lower')
ax[0].set_title('Data')
ax[1].set_title('Model')
ax[2].set_title('Residual Image')
plt.tight_layout()