"""
Algorigthm adapted from Fichtner A, Zunino A. Hamiltonian Nullspace Shuttles. Geophys Res
Letters 2019 Jan 28;46(2):644-651. doi: 10.1029/2018GL080931, 2019.

This script was used for, and is assocated to, results shown in: Giraud, J., Ford, M., Caumon, G., Ogarko, V., Grose,
L., Martin, R., and Cupillard, P.: Geologically constrained geometry inversion and null-space navigation to explore
alternative geological scenarios: a case study in the Western Pyrenees, Geophysical Journal International,
https://doi.org/10.1093/gji/ggae192, 2024.

# This script:
# 1) Calculates the perturbation for null space shuttles, applied only to certain parts of the model
# 2) Does null space shuttles navigation
# 3) Plots results

# Important parameters to tune: a) importance of eps b) importance of delta_t c) scale p so that H remains constant.

# Note: distances are converted to kilometers when read from the input file. This is used only for plots, but can cause
# plots to look inconsistent expected units are, e.g., meters.
"""

from argparse import ArgumentParser
import numpy as np
import matplotlib
import colorcet as cc
from pathlib import Path

# The package is installed via "pip install ."
from tomofast_utils import *
import tomofast_reading_utils as tr
from input_params import *
import nullspace_utils as nu
from forward_calculation_utils import *
import nullspace_solver as ns
import nullspace_plot as npt
import time

# matplotlib.use('Qt5Agg')

# Possible improvements:
# 1. find a more efficient way to unpack values from the <par> argument in the solve function.
# 2. add the capability to also load a sensitivity matrix calculated with UBC codes or Simpeg.
# 3. give more flexibility to the mask (i.e.: add the possibility to load a mask externally).


def solve(par):
    """
    This function is called to run null space navigation using parameters contained in the class par.
    :param par: a class containing file paths to input data and parameters for the solver.
    """

    # Unpacking parameter class 'par'.

    sensit_path = par.sensit_path
    tomofast_sensit_nbproc = par.tomofast_sensit_nbproc
    use_rotation_matrix = par.use_rotation_matrix
    unit_conv = par.unit_conv
    use_mask_domain = par.use_mask_domain
    weight_prior_model = par.weight_prior_model
    sensit_type = par.sensit_type
    rotation_mat_filename = par.rotation_mat_filename
    geol_model_path = par.geol_model_path
    data_outline_filename = par.data_outline_filename
    data_vals_filename = par.data_vals_filename
    data_background_filename = par.data_background_filename
    perturbation_filename = par.perturbation_filename

    # Initialize model parameters class.
    gpars = ns.GridParameters()

    # Initialise model variables class.
    mvars = ns.ModelsVariables()

    # Set the dimensions of the model (nz, nx, ny).
    gpars.dim = (par.nz, par.nx, par.ny)

    # ----------------------------------------------------------------------------------
    # Read the model and model grid.
    model_filename = par.model_filename
    # mvars.m_beg: starting point for the nullspace navigation (unperturbed model).
    mvars.m_beg, gpars = tr.read_tomofast_model(model_filename, gpars)

    # Geological model for plots and comparison (not used in computation: only for plots).
    mvars.m_geol_orig, _ = tr.read_tomofast_model(geol_model_path, gpars)

    # ----------------------------------------------------------------------------------
    # Setup for saving plots.
    save_plots = par.save_plots

    # Create output folder for plots.
    if save_plots:
        folder_path = Path(par.path_output)
        folder_path.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------------------
    # Pre-processing parameters: definition of mask.
    # Index of rock unit for perturbation and null space analysis (rocks indexed by increasing density).
    ind_unit_mask = par.ind_unit_mask  # Index used for the calculation of the mask. In Pyrenees paper: 9 = Mantle.
    # Distance max in number of cells away from the outline of rock unit with index 'ind_unit_mask'.
    distance_max = par.distance_max  # In Pyrenees paper: 8 in tests shown.
    # ----------------------------------------------------------------------------------
    # Calculate the domain mask: cells where the solver is allowed to change the model values.
    # The domain mask is used to reduce the part of the model that can be affected by null space navigation.
    # We define the compute domain with masks on distance to the outline of selected unit (if applicable).
    mvars.domain_mask = nu.get_masked_domain(use_mask_domain, distance_max, ind_unit_mask, mvars.m_beg)

    # ----------------------------------------------------------------------------------
    # Reading the gravity data.

    # Initialise gravity data class.
    grav_data = nu.GravData()

    tr.read_tomofast_data(grav_data, data_vals_filename, data_type='field')
    tr.read_tomofast_data(grav_data, data_background_filename, data_type='background')

    # ----------------------------------------------------------------------------------
    # Load sensitivity kernel from Tomofast-x.
    sensit = load_sensit_from_tomofastx(sensit_path, nbproc=tomofast_sensit_nbproc, type=sensit_type, verbose=False)

    # ----------------------------------------------------------------------------------

    # Initialise solver parameters parameters.
    shpars = ns.SolverParameters(eps=par.eps,
                                 max_change=par.max_change,
                                 num_epochs=par.num_epochs,
                                 time_step=par.time_step,
                                 weight_prior=weight_prior_model)  
    # Load model perturbation: the model change we want to impose (the final model should have this change).
    mvars.delta_m_orig, _ = tr.read_tomofast_model(perturbation_filename, gpars)

    # ----------------------------------------------------------------------------------
    # Initialization of nullspace navigation.
    # ----------------------------------------------------------------------------------
    # Null space shuttle: perturbation to apply during null space exploration.
    mvars.delta_m = mvars.delta_m_orig

    # Initialising a class setting up the nullspace shuttle using Fichtner and Zunino (2019).
    nsvars = ns.NullSpaceNavigationVars(mvars.delta_m, shpars.eps)

    # Initialise models for nullspace navigation.
    mvars.m_nullspace_orig = mvars.m_beg.flatten().copy()
    mvars.m_nullspace_subs = mvars.m_nullspace_orig

    mvars.m_beg = mvars.m_nullspace_subs.copy()  # Starting point for nullspace navigation.
    mvars.m_curr = mvars.m_nullspace_orig.copy()  # Current model.
    # ----------------------------------------------------------------------------------

    # %% ----------------------------------------------------------------------------------
    # Do the nullspace navigation. Stops when the perturbation reaches par.max_change or after par.num_epochs.
    grav_data, misfit_data, kinetic_energy, model_misfit, mvars = ns.nullspace_navigation(shpars, mvars, nsvars,
                                                                                          sensit, grav_data)

    # Print final data RMS error.
    misfit_data_final = calc_data_rms(grav_data)
    print(str("Data misfit after null space navigation:"), format(misfit_data_final, ".2f"), str(' mGal'))

    # Calculate response of perturbation mvars.delta_m_orig and its RMS.
    grav_data_deltam = calc_fwd(sensit, mvars.delta_m_orig.flatten(), unit_conv, use_csr_matrix=True)
    rms_pert = calc_data_rms(grav_data_deltam)
    print(str("RMS of data of original model perturbation:"), format(rms_pert, ".2f"), str(' mGal'))

    # Calculate response of the difference between the original (starting) and final model and its RMS.
    m_diff = nu.calc_model_diff(mvars.m_curr, mvars.m_nullspace_orig)
    grav_data_mdiff = calc_fwd(sensit, m_diff, unit_conv, use_csr_matrix=True)
    rms_mdiff = calc_data_rms(grav_data_mdiff)
    print(str("RMS data of diff between orig - final model:"), format(rms_mdiff, ".2f"), str(' mGal'))

    # %% ===============================================================================================
    # Setting up plots and visual analysis.
    # Models to plot - 2x2 subplots.
    # ===============================================================================================
    # Set plot properties.
    npt.set_plotprops()

    # Read and the outline of core area of modelling (area covered by gravity data).
    npt.read_data_outline(grav_data, data_outline_filename)

    # Calculate Hamiltonian and its different terms for analysis.
    hamiltonian_quantities = ns.HamiltonQuantities(kinetic_energy, misfit_data, model_misfit, unit_conv,
                                                   weight_prior_model)

    # Plotting parameters.
    ppars = npt.PlotParameters(dens_tresh=30, colm=250, slice_x=20, slice_z=10)  # slice_z=12 also interesting.

    # Define plot parameters.
    ppars = npt.prepare_plots(gpars.dim, mvars, m_diff, ppars, grav_data.outline_coords)

    # Find where differences above a certain threshold are located -- used for the scatter plot.
    ind_bigdiff_all, ind_bigdiff_slice = npt.get_large_diff(gpars, ppars, mvars.m_curr, mvars.m_nullspace_orig)

    # Get rotation matrix used to define the mesh for inversion: needed to relocate data in their geographical location.
    rotation_matrix = nu.get_rotation_matrix(use_rotation_matrix, rotation_mat_filename)

    # %% ===============================================================================================
    # Do the plotting.
    # ===============================================================================================

    # Plot a depth slice with before/after/differences.
    fig_depthslice = npt.plot_navigation_depthslice(gpars, ppars, rotation_matrix, ind_bigdiff_all,
                                                    grav_data.outline_coords)
    # Plot a vertical cross section with before/after/differences.
    fig_xsection = npt.plot_navigation_xsection(gpars, ppars, ind_bigdiff_slice)
    # Plot modelling metrics.
    fig_gravpert = npt.plot_grav_perturbation(misfit_data, grav_data, grav_data_deltam, rotation_matrix,
                                              hamiltonian_quantities)

    # Save the plots.
    npt.save_plot(fig=fig_depthslice, filename=par.path_output + '/Differences_DepthSlice', ext='.png', save=save_plots)
    npt.save_plot(fig=fig_xsection, filename=par.path_output + '/Differences_VertSlice', ext='.png', save=save_plots)
    npt.save_plot(fig=fig_gravpert, filename=par.path_output + '/Metrics', ext='.png', dpi=300, save=save_plots)


# =========================================================================================================
def main(parfile_path):
    """
    Main function of the Nullspace navigation script. It reads a parameter file (parfile) that contains the following:
    - file paths,
    - solver parameters,
    that are required run the modelling and which can be changed by the user.

    :param parfile_path: relative path to the parfile.
    :return:
    """
    print('Started nullspace main')

    # Read input parameters.
    par = read_input_parameters(parfile_path)

    # Record the start time
    start_time = time.time()

    # Run the solver.
    solve(par)

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print('RUN TIME: ', elapsed_time, 'sec')


# =============================================================================
if __name__ == "__main__":
    # Read command line arguments.
    parser = ArgumentParser()
    parser.add_argument("-p", "--parfile", dest="parfile_path",
                        help="path to the parameters file", default="parfiles/Parfile_paper.txt")

    # Get the information from the parameter file.
    args = parser.parse_args()

    # Run the main program.
    main(args.parfile_path)
    print('Completed.')
