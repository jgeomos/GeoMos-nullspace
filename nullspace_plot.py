from dataclasses import dataclass
import matplotlib.pylab as plt
import numpy as np
import colorcet as cc  # Used only for colormaps.
import random as rd
from typing import Optional
from forward_calculation_utils import rotate_mesh
import nullspace_utils as nu
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable


@dataclass
class PlotParameters:
    """
    Parameters for plots of null space navigation outputs.
    at the moment: plots only in the x direction.
    """

    # Threshold for identification of density differences between models before/after navigation.
    dens_tresh: float = 30
    # Value of density contrast defining range of colors min and max.
    colm: float = 250
    # Slices for plots.
    slice_x: int = -1
    slice_y: int = -1
    slice_z: int = -1
    plot_models: Optional[tuple] = None
    # Titles for plots.
    plot_titles: Optional[tuple] = None
    # Titles for color bars.
    cbar_titles: Optional[tuple] = None
    # Limits for colours on plot.
    clims: Optional[tuple] = None
    # Colour schemes for plots.
    colorschemes: Optional[tuple] = None
    # Ticks for colorbar.
    cbar_ticks: Optional[tuple] = None
    # limits in x and y directions for plots
    xlims: Optional[np.array] = None
    ylims: Optional[np.array] = None


def add_grid(ax):
    """
    Add grid to existing plot axes.
    """

    # Add grid.
    ax.grid()
    ax.minorticks_on()
    ax.grid(visible=True, which='minor', color='0.2', linestyle='--', alpha=0.1)
    ax.grid(visible=True, which='major', color='0.6', linestyle='--', alpha=1)


def set_plotprops():
    """
    Set default plot properties to use for all plots in the script.
    """

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 13})


def plot_addticks_cbar(cbar_title, cbar_ticks, shrink_perc):
    """
    Add colorbar with specified title and ticks.

    :param cbar_title: title for the colorbar.
    :param cbar_ticks: location of the ticks on the colorbar.
    :return: colobar handle.
    """

    cbar = plt.colorbar(shrink=shrink_perc, ticks=cbar_ticks)
    # cbar.set_label(cbar_title, labelpad=-20, y=-0.015, rotation=0, fontfamily='serif')
    # cbar.set_label(cbar_title, labelpad=-20, x=1.15, y=-0.02, rotation=0)
    cbar.set_label(cbar_title, labelpad=-20, x=1.10, y=1.25, rotation=0)

    ## Changing the font of ticks.
    # for i in cbar.ax.yaxis.get_title():
    #     i.set_family("Comic Sans MS")

    return cbar


def calc_plot_coordinates(mpars, ppars):
    """
    Get the distance along a profile (oblique or not) crossing the modelling mesh.

    :param ppars: PlotParameters object.
    :param mpars: ModelParameters object.
    :return: dist_profile: 2D ndarray containing, z_plot: 2D ndarray of the corresponding depth.
    """

    # For Pyrenees case.
    # x_plot = mpars.x.reshape(mpars.dim)[:, ppars.slice_x, 9:-10]
    # y_plot = mpars.y.reshape(mpars.dim)[:, ppars.slice_x, 9:-10]
    # z_plot = -mpars.z.reshape(mpars.dim)[:, ppars.slice_x, 9:-10]

    # For homogenous example.
    x_plot = mpars.x.reshape(mpars.dim)[:, ppars.slice_x, :]
    y_plot = mpars.y.reshape(mpars.dim)[:, ppars.slice_x, :]
    z_plot = mpars.z.reshape(mpars.dim)[:, ppars.slice_x, :]

    x_min = np.min(x_plot)
    y_min = np.min(y_plot)

    dist_profile = np.round(np.sqrt((x_plot - x_min) ** 2 + (y_plot - y_min) ** 2))

    return dist_profile, z_plot


def plot_core_outline(outline_coords):
    """
    Plots the outline of the area covered by gravity data.

    :param gravity_data: gravity data class
    :return: None.
    """

    if outline_coords is not None:

        x_outline = outline_coords[:, 0]
        y_outline = outline_coords[:, 1]

        plt.plot(x_outline, y_outline, 'r--', linewidth=0.75)


def plot_grav_perturbation(misfit_evolution, gravity_data, grav_data_diff, rotation_matrix, hamiltonian):
    """
    Plot the evolution of gravity data misfit and the misfit due to the anomaly assessed using nullspace shuttle

    :param misfit_evolution: 1D array,  The evolution of gravity data misfit.
    :param gravity_data: GetGravData object, The gravity data class contains the gravity data + coordinates.
    :param grav_data_diff: 1D array, The forward gravity data of the due to the anomaly assessed using nullspace shuttle
    :param rotation_matrix: 2D array, The rotation matrix used to rotate the data.
    :param hamiltonian: HamiltonQuantities class containing the quantities to calculate the Hamiltonian.
    :return: None, only plots the data.
    """

    def get_accuracy(values):

        # Convert each value to a string.
        string_values = [f"{v:.16f}" for v in values]

        # Iterate over each digit after the decimal point.
        for i in range(1, len(string_values[0])):
            # Check if all characters up to this point are the same.
            if all(s[i] == string_values[0][i] for s in string_values):
                continue
            else:
                digit_num = i - 4  # -4 because it accounts for the 1st digit and the dot.
                return digit_num

    # Rounding the total energy so that matplotlib is not affect by numerical inaccuracy to plot nearly constant values.
    n_digits = get_accuracy(hamiltonian.total_energy)
    hamiltonian.total_energy = np.round(hamiltonian.total_energy, n_digits)  # Rounding to n_digits, (matplolib pb?).

    gravity_data.x_data, gravity_data.y_data = rotate_data(gravity_data, rotation_matrix)

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(10, 7), constrained_layout=True)

    # fig.tight_layout()

    ax = fig.add_subplot(5, 1, 1)
    ax.plot(misfit_evolution[misfit_evolution > 0])
    ax.set_title('(a) Data misfit during null space navigation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Data misfit (mGal)')
    add_grid(ax)

    # fig = plt.figure(rd.randint(0, int(1e6)), figsize=(8, 8))
    ax = fig.add_subplot(5, 1, 2)
    ax.plot(hamiltonian.total_energy)
    ax.set_title('(b) Artificial Hamiltonian')
    ax.set_ylabel('Total Energy')
    ax.set_xlabel('Epochs')
    add_grid(ax)

    ax = fig.add_subplot(5, 1, 3)
    ax.plot(hamiltonian.kinetic_energy)
    ax.set_title('(c) Kinetic energy')
    ax.set_ylabel('Kinetic Energy')
    ax.set_xlabel('Epochs')
    add_grid(ax)

    ax = fig.add_subplot(5, 1, 4)
    ax.plot(hamiltonian.potential_energy)
    ax.set_title('(d) Potential energy')
    ax.set_ylabel('Potential Energy')
    ax.set_xlabel('Epochs')
    add_grid(ax)

    ax = fig.add_subplot(5, 1, 5)
    sct = ax.scatter(gravity_data.x_data / 1e3,
                gravity_data.y_data / 1e3, 50, c=grav_data_diff)  # edgecolors='black')
    # plot_addticks_cbar('mGal', np.linspace(-10, 10, 11))
    # plt.colorbar(extend='both', orientation='horizontal')

    divider1 = make_axes_locatable(ax)
    cax1 = divider1.append_axes("right", size="1.0%", pad=-0.5)
    cbar = fig.colorbar(sct, cax=cax1)
    cbar.set_label('$kg.m^{-3}$', labelpad=-20, x=1.10, y=1.25, rotation=0)


    plot_core_outline(gravity_data.outline_coords)
    add_grid(ax)
    # ax.set_aspect('equal'),
    ax.set_aspect('equal', 'box')
    ax.set_title('(e) Forward Bouguer anomaly of perturbation')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Distance (km)')

    plt.show()

    return fig


def plot_model(ax, mesh_dim1, mesh_dim2, mod, slice_plot, title_string, cmap, clim):
    """
    Plots a 2D section of gridded model along a specific slice.

    :param ax: The axes object to plot onto.
    :param mesh_dim1: numpy.ndarray, 2D, mesh along the 1st dimension to plot.
    :param mesh_dim2: numpy.ndarray, 2D, mesh along the 2nd dimension to plot.
    :param mod: numpy.ndarray, 2D, The slice to plot.
    :param slice_plot: int, indices of the slice in the 3D model.
    :param title_string: str, The title of the plot.
    :param cmap: matplotlib.colors.LinearSegmentedColormap, The color map to use for the plot.
    :param clim: tuple, The color limits to use for the plot.
    :return: matplotlib.collections.QuadMesh
    """

    color_min = clim[0]
    color_max = clim[1]

    # TODO: make use of the PlotParameters class here

    # For Pyrenees case.
    # handle = plt.pcolormesh(mesh_dim1, mesh_dim2, mod[:, slice_plot, 9:-10], cmap=cmap, vmin=color_min, vmax=color_max,
    #                         label='test1')

    # For homogenous model example.
    # matrix_to_plot = np.asfortranarray(mod[:, slice_plot, :])
    matrix_to_plot = mod[:, slice_plot, :]
    handle = plt.pcolormesh(mesh_dim1, mesh_dim2, matrix_to_plot, cmap=cmap, vmin=color_min, vmax=color_max,
                            label='test1')

    # plt.text(np.min(mesh_dim1) - 7, np.max(mesh_dim2) + 3, 'A', fontsize=13)
    # plt.text(np.max(mesh_dim1) + 1, np.max(mesh_dim2) + 3, 'B', fontsize=13)

    add_grid(ax)

    ax.set_aspect('equal', 'box'),
    ax.set_xlabel('Distance along profile (km)')
    ax.set_ylabel('Depth (km)')

    plt.box(on=bool(1))
    plt.title(title_string)

    return handle


def get_large_diff(mpars, ppars, model1, model2):
    """
    Identifies cells from 3D volume along slice with indices ppars.slice_x where the value of teh differences between
     model1 and model 2 is superior to ppars.dens_tresh, using the mask mask_location.

    :param ppars: PlotParameters object.
    :param mpars: ModelParameters object.
    :param model1: numpy.ndarray, 1D array containing the model to calculate the difference / update to plot.
    :param model2: numpy.ndarray, 1D array containing the model to calculate the difference / update to plot.
    :return: A tuple containing two arrays. The first array contains the indices of all cells with values greater than
             ppars.dens_tresh, and the second array contains the indices of cells with values greater than
             ppars.dens_tresh along the slice with indices ppars.slice_x.
    """

    # Find where mask was not applied.
    # mask_model_nodiff = 1 - mask_location.reshape(mpars.dim)
    # Get part of the model where mask was not applied, ie where it could evolve during null space navigation.
    # model_diff_masked = nu.apply_mask_diff(mask_model_nodiff, model1, model2).reshape(mpars.dim)

    # Calculate the differences between models.
    model_diff = nu.calc_model_diff(model1, model2).reshape(mpars.dim)
    # Restrict analysis to subdomain.
    model_diff_masked_slice = model_diff[:, ppars.slice_x, :]
    # Identify indices of cells with values superior to a predefined threshold on the slice.
    ind_bigdiff_all = np.where(np.abs(model_diff) > ppars.dens_tresh)
    # Identify indices of cells with values superior to a predefined threshold on the slice.
    ind_bigdiff_slice = np.where(np.abs(model_diff_masked_slice) > ppars.dens_tresh)

    return ind_bigdiff_all, ind_bigdiff_slice


def plot_navigation_xsection(mpars, ppars, ind_scatter):
    """
    Plots four 2D sections of the gridded model, along a specific slice, with a scatter plot on top for the last one.

    :param ind_scatter: tuple of length 2 with numpy.ndarray of indices along selected profile to use for scatter plot.
    :param mpars: ModelParameters object containing parameters for the model.
    :param ppars: PlotParameters object containing parameters for the plot.
    :return: None.
    """

    plot_models = ppars.plot_models
    colorschemes = ppars.colorschemes
    clims = ppars.clims
    cbar_ticks = ppars.cbar_ticks
    cbar_titles = ppars.cbar_titles
    plot_titles = ppars.plot_titles
    slice_x = ppars.slice_x

    # Get the coordinates for plotting.
    # Used for the Pyrenees. 
    # dist_profile, z_plot = calc_plot_coordinates(mpars, ppars)

    z_plot = mpars.z.reshape(mpars.dim)[:, ppars.slice_x, :]
    x_plot = mpars.x.reshape(mpars.dim)[:, ppars.slice_x, :]

    n_subplots = 4
    n_row_subplots = 4
    n_columns_subplot = 1

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(13, 7))

    for i in range(0, n_subplots):

        ax = fig.add_subplot(n_row_subplots, n_columns_subplot, int(i + 1))
        # Used for the Pyrenees.
        # plot_model(ax, mesh_dim1=dist_profile, mesh_dim2=z_plot, mod=plot_models[i],
        #            slice_plot=slice_x, title_string=plot_titles[i], cmap=colorschemes[i], clim=clims[i])
        plot_model(ax, mesh_dim1=x_plot, mesh_dim2=z_plot, mod=plot_models[i],
                   slice_plot=slice_x, title_string=plot_titles[i], cmap=colorschemes[i], clim=clims[i])
        plot_addticks_cbar(cbar_titles[i], cbar_ticks[i], shrink_perc=1.)


        
        plt.xlim((ppars.xlims[0], ppars.xlims[1]))

        ax.invert_yaxis()

        # Adding the plot of black dots showing differences above a threshold specified in the parameter file.
        # 2nd panel.
        # if i == 1:
        #     # For Pyrenees.
        #     # plt.scatter(dist_profile[ind_scatter[0], ind_scatter[1]-9], z_plot[ind_scatter[0], ind_scatter[1]-9],
        #     #             alpha=0.5, s=1, c='black', label='Values superior to threshold')
        #     plt.scatter(x_plot[ind_scatter[0], ind_scatter[1]], z_plot[ind_scatter[0], ind_scatter[1]],
        #                 alpha=0.5, s=1, c='black', label='Values superior to threshold')
        # # 4th panel.
        # if i == 3:
        #     # For Pyrenees.
        #     # plt.scatter(dist_profile[ind_scatter[0], ind_scatter[1]-9], z_plot[ind_scatter[0], ind_scatter[1]-9],
        #     #             alpha=0.5, s=1, c='black', label='Values superior to threshold')
        #     plt.scatter(x_plot[ind_scatter[0], ind_scatter[1]], z_plot[ind_scatter[0], ind_scatter[1]-9],
        #                 alpha=0.5, s=1, c='black', label='Values superior to threshold')

    # For Pyrenees. 
    # plt.text(np.min(dist_profile) - 7, np.max(z_plot) + 3, 'A', fontsize=13)
    # plt.text(np.max(dist_profile) + 1, np.max(z_plot) + 3, 'B', fontsize=13)
    # ax.annotate("Original perturbation", xy=(135, -24), xycoords='data', xytext=(15, -25), textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.legend()
    fig.tight_layout()
    plt.show()

    return fig


def rotate_data(gravity_data, rotation_matrix):
    # TODO: move this function somewhere? 
    """
    Rotates the location of gravity data in gravity_data using rotation_matrix

    :param gravity_data: GetGravData object, The gravity data class contains the gravity data + coordinates.
    :param rotation_matrix: numpy.ndarray, a 2x2 rotation matrix
    :return: rotated x_data and y_data, tuple of numpy.ndarray
    """

    x_data = gravity_data.x_data
    y_data = gravity_data.y_data

    coord = np.matmul(rotation_matrix[0:2, 0:2], np.array([x_data, y_data]))
    x_data = coord[0, :]
    y_data = coord[1, :]

    return x_data, y_data


def plot_navigation_depthslice(mpars, ppars, rotation_matrix, indice_scatter, outline_coords):
    """
    Plots four depth slices of geophysical models using rotation_matrix to rotate locations in x, y, z coordinates,
    with a scatter plot on top for the last one.

    :param mpars: ModelParameters object containing parameters for the model.
    :param ppars: PlotParameters object containing parameters for the plot.
    :param rotation_matrix: numpy.ndarray, a 3x3 rotation matrix.
    :param indice_scatter: An array containing the indices of scatter points to be plotted.
    :param outline_coords: Data outline.
    :return: None
    """

    n_subplots = 4
    n_row_subplots = 2
    n_columns_subplot = 2

    plot_models = ppars.plot_models

    coord_x, coord_y, coord_z = rotate_mesh(mpars, rotation_matrix)

    fig = plt.figure(rd.randint(0, int(1e6)), figsize=(8, 8))

    for i in range(0, n_subplots):
        ax = fig.add_subplot(n_row_subplots, n_columns_subplot, i + 1)
        plt.pcolormesh(coord_x[ppars.slice_z, :, :], coord_y[ppars.slice_z, :, :],
                       plot_models[i][ppars.slice_z, :, :],
                       cmap=ppars.colorschemes[i], clim=ppars.clims[i])
        plot_core_outline(outline_coords)
        add_grid(ax)
        # ax.set_aspect('equal'),
        ax.set_aspect('equal', 'box')
        plt.title(ppars.plot_titles[i])
        plot_addticks_cbar(ppars.cbar_titles[i], ppars.cbar_ticks[i], shrink_perc=0.2)
        ax.set_xlabel('Easting (km)')
        ax.set_ylabel('Northing (km)')
        plt.xlim(ppars.xlims)
        plt.ylim(ppars.ylims)

    plt.scatter(coord_x[indice_scatter[0], indice_scatter[1], indice_scatter[2]],
                coord_y[indice_scatter[0], indice_scatter[1], indice_scatter[2]],
                alpha=0.15, s=5, c='black', marker='o', label='Values superior to threshold')
    # For Pyrenees field application.
    # ax.annotate("Original perturbation", xy=(633, 4792), xycoords='data', xytext=(600, 4820), textcoords='data',
    #             arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
    # plt.legend()
    fig.tight_layout()
    plt.show()

    return fig


def prepare_plots(dim, mvars, m_diff, ppars, outline_coords, xlims=None, ylims=None):
    """
    Define plot parameters: models to plot, titles, limits etc.
    """

    print('\nPlot parameters are hardcoded in function', prepare_plots.__name__, "in file",  os.path.basename(__file__))
    print('\n')

    # Models to plot in the 2x2 subplot.
    # First subplot.
    m1 = mvars.m_nullspace_orig.reshape(dim)
    # Second subplot.
    m2 = mvars.m_curr.reshape(dim)
    # Third subplot.
    m3 = mvars.m_geol_orig.reshape(dim)
    # Fourth subplot.
    m4 = m_diff.reshape(dim)

    ppars.plot_models = (m1, m2, m3, m4)

    # Color limits for the subplots.
    # For Pyrenees field case.
    # ppars.clims = (np.array([m1.min(), m1.max()]),  # In example shown in paper: m3 is the starting model.
    #                np.array([m1.min(), m1.max()]),
    #                np.array([m1.min(), m1.max()]),
    #                np.array([-200, 200]))
    # For homogenous model example.
    ppars.clims = (np.array([-300, 300]),
                   np.array([-300, 300]),
                   np.array([-300, 300]),
                   np.array([-300, 300]))

    # Colormaps for each subplot.
    ppars.colorschemes = (cc.cm.CET_R4,
                          cc.cm.CET_R4,
                          cc.cm.CET_R4,
                          'seismic')
    # Titles for each subplot.
    ppars.plot_titles = ('(a) Start of navigation',
                         '(b) End of navigation',
                         '(c) Reference model',
                         '(d) Difference: End - Start')

    # Titles for each colorbar attached to the subplots.
    ppars.cbar_titles = ('$kg.m^{-3}$',
                         '$kg.m^{-3}$',
                         '$kg.m^{-3}$',
                         '$kg.m^{-3}$')

    # Ticks for each colorbar.
    # For Pyrenees field case.
    # ppars.cbar_ticks = ([2400, 2600, 2800, 3000, 3200],
    #                     [2400, 2600, 2800, 3000, 3200],
    #                     [2400, 2600, 2800, 3000, 3200],
    #                     [-200, -100, 0, 100, 200])
    # For homogenous model example.
    ppars.cbar_ticks = ([-200, -100, 0, 100, 200],
                        [-200, -100, 0, 100, 200],
                        [-200, -100, 0, 100, 200],
                        [-200, -100, 0, 100, 200])

    # Limits for the plot of top view of depth slices: outline of the area covered by the data plus a buffer around it.
    if outline_coords is not None:
        ppars.xlims = np.array(
            [-2.5 + outline_coords[:, 0].min(), 2.5 + outline_coords[:, 0].max()])
        ppars.ylims = np.array(
            [-2.5 + outline_coords[:, 1].min(), 2.5 + outline_coords[:, 1].max()])
    else:
        # 1) In the absence of an shape around the area covered by the data, let the xlim and ylim be set automatically.
        # ppars.xlims = None
        # ppars.ylims = None
        # 2) Or do it by hand (unit: kilometers):
        # For Pyrenees example.
        # ppars.xlims = np.array([600, 710])
        # ppars.ylims = np.array([4740, 4850])
        # For homogenous model example.
        ppars.xlims = xlims
        ppars.ylims = ylims

    return ppars


def read_data_outline(grav_data, file_path_data_outline):
    """
    Read data outline.
    """

    # Read the matrix containing the values for the locations of the outline of the real world area covered by gravity
    # data.
    # TODO: add a condition about whether 'core_outline_points' exists. Could be we don't need it.

    if os.path.exists(file_path_data_outline):
        grav_data.outline_coords = np.loadtxt(file_path_data_outline)
    else:
        print(f"File '{file_path_data_outline}' not found: not using the outline of area covered by data.")
        grav_data.outline_coords = None


def save_plot(fig=None, filename='myplot', ext='.png', dpi=300, save=False):
    """
    Save the current figure to file or the figure provided in argument.

    :param: filename (str): The name of the output file.
    :param: dpi (int): Dots per inch (resolution) of the saved image (default: 300).
    :param: format (str): The format of the output file (default: 'png').
    :param: save (bool): Flag to indicate whether to save the plot (default: True).

    Returns: None
    """

    filename = filename + ext

    if save:

        # Check that the extension provided is OK.
        _, ext = os.path.splitext(filename)
        ext_lower = ext.lower()

        valid_extensions = ['.png', '.jpg', '.jpeg', '.svg', '.pdf']
        if ext_lower not in valid_extensions:
            raise ValueError("Invalid file extension. Supported extensions are: " + ", ".join(valid_extensions))

        if fig is None:
            # Get the current figure.
            fig = plt.gcf()

        # Do the saving;
        fig.savefig(filename, dpi=dpi, format=ext_lower[1:])
