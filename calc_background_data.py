import matplotlib.pylab as plt
import numpy as np
import nullspace_solver as ns

"""
A script to calculate the background data. 
"""

from argparse import ArgumentParser
import numpy as np
import matplotlib.pylab as plt
import colorcet as cc
import nullspace_solver as ns
import nullspace_plot as npt
from forward_calculation_utils import *
from pathlib import Path
import nullspace_utils as nu

# The package is installed via "pip install ."
from tomofast_utils import *
import tomofast_reading_utils as tr

from input_params import *

import matplotlib
matplotlib.use('Qt5Agg')

# Read input parameters.
par = read_input_parameters('parfiles/Parfile.txt')

# Initialize model parameters class.
gpars = ns.GridParameters()

use_tomofast_sensit = par.use_tomofast_sensit
tomofast_sensit_nbproc = par.tomofast_sensit_nbproc
use_rotation_matrix = par.use_rotation_matrix
write_tomofast_grids = par.write_tomofast_grids
unit_conv = par.unit_conv
use_mask_domain = par.use_mask_domain
use_mask_location = par.use_mask_location
weight_prior_model = par.weight_prior_model

# Set the dimensions of the model (nz, nx, ny).
gpars.dim = (par.nz, par.nx, par.ny)
gpars.vert_size = par.vert_size

# Init. model variables
mvars = ns.ModelsVariables()  # This is not used until a bit later... move this down?

# ----------------------------------------------------------------------------------
# Pre-processing parameters.
# Index of rock unit for perturbation and null space analysis.
ind_unit_mask = par.ind_unit_mask  # Index of rock unit (by increasing density value) to define the mask on perturbations. 9 = Mantle.
# Distance max in number of cells away from the outline of rock unit considered.
distance_max = par.distance_max  # 4, 8 in tests shown in Pyrenees paper.

# ----------------------------------------------------------------------------------
# Read the model and model grid.
model_filename = par.model_filename
m_ref, gpars = tr.read_tomofast_model(model_filename, gpars)

# ----------------------------------------------------------------------------------
# Reading the gravity data.

grav_data = nu.GravData()

data_vals_filename = par.data_vals_filename
# TODO: recalculate data_background with Tomofast: could be that it came from the 'old' sensit!!
data_background_filename = par.data_background_filename  # TODO: recalculate, could be coming from old sensit!

tr.read_tomofast_data(grav_data, data_vals_filename, data_type='field')
tr.read_tomofast_data(grav_data, data_background_filename, data_type='background')

# ----------------------------------------------------------------------------------
if use_tomofast_sensit:
    # Load sensitivity kernel from Tomofast-x.
    sensit_path = "../Tomofast-x/output/geomos/SENSIT"
    sensit_tomo = load_sensit_from_tomofastx(sensit_path, nbproc=tomofast_sensit_nbproc, verbose=False)

    sensit = sensit_tomo


def calc_background_model(dens_crust, dens_mantle, depth_mantle, z):
    """
    Defines the a 2-layer model with depths lower than depth_mantle assigned with dens_crust and dens_mantle for depth
    higher than depth_mantle. Useful for the calculation of Bouguer anomaly.

    :param dens_crust:
    :param dens_mantle:
    :param depth_mantle:
    :param z:
    :return:
    """

    ind_crust_background = np.where(z < depth_mantle)
    ind_mantle_background = np.where(z > depth_mantle)

    # create model with background values
    model_background = np.zeros(z.shape)
    model_background[ind_crust_background] = dens_crust
    model_background[ind_mantle_background] = dens_mantle

    return model_background


# Calculate a two-layer background_model used for calculation of the Bouguer anomaly.
dens_background = calc_background_model(dens_crust=2670, dens_mantle=3260, depth_mantle=30.000, z=gpars.z)

# Grav data background as of 'old' sensit.
grav_data_background_old = grav_data.background

# Calculate the forward data of the background.
grav_data_background_new = calc_fwd(sensit, dens_background, unit_conv, use_csr_matrix=use_tomofast_sensit)

# Calculate the difference.
diff = grav_data_background_old - grav_data_background_new

# Calculate the RMS.
RMS_diff = np.sqrt(np.mean(diff**2) / np.shape(diff)[0])

# ----------------------------- PLOTS.

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharex=True)

ax1.set_aspect('equal', adjustable='box')
scatter = ax1.scatter(grav_data.x_data, grav_data.y_data, s=15, c=grav_data_background_new, cmap='jet')
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Color Scale')
ax1.set_title('grav_data_background_new')

ax2.set_aspect('equal', adjustable='box')
scatter = ax2.scatter(grav_data.x_data, grav_data.y_data, s=15, c=grav_data_background_old, cmap='jet')
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label('Color Scale')
ax2.set_title('grav_data_background_old')

ax3.set_aspect('equal', adjustable='box')
scatter = ax3.scatter(grav_data.x_data, grav_data.y_data, s=15, c=diff, cmap='jet')
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Color Scale')
ax3.set_title('Difference')

plt.show()

# Save test background value.
np.save('grav_data_background_new.npy', grav_data_background_new)
np.savetxt('grav_data_background_new.txt', grav_data_background_new, fmt='%f', delimiter='\t')

