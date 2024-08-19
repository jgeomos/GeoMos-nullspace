import numpy as np
import skfmm
import warnings
from dataclasses import dataclass

# COMMENTS: checked on 12/06/2024.


@dataclass
class GravData:
    """
    A dataclass containing the calculated, measured and background data, and the geographical location of said data.
    """

    data_calc: np.array = None
    data_field: np.array = None
    background: np.array = None

    x_data: np.array = None
    y_data: np.array = None
    z_data: np.array = None

    # For plotting only.
    outline_coords: np.array = None


def get_masked_domain(use_mask_domain, distance_max, ind_unit_mask, dens_model, mask_first_layer=True):
    """
    Calculates the signed distances to unit with index 'ind_unit_mask' to create a mask based on the signed distance
    values (e.g., can be used to mask changes further away from a certain distances to interface to selected rock unit).

    :param use_mask_domain: boolean to control whether the mask will be calculated and used. If False, not calculated.
    :param distance_max: maximum distance away from 'ind_unit_mask' in terms of number of cells.
    :param ind_unit_mask: index of the considered unit in the sorted densities (index 0: least dense unit, index 1:
    second least dense unit, etc.)
    :param dens_model: density model.
    :param mask_first_layer: boolean controlling whether first layer is masked.
    # TODO: make this coherent with the use of calc_mask_layer.
    :return: mask.
    """

    def calc_signed_distances(mod_dens_const, drho0, cell_size=None, narrow=False):
        """
        Calculates the signed distances from the given parameters using the Fast Marching Method (FMM).

        Parameters:
        --------------
        :param mod_dens_const: array_like, density model, 3D matrix.
            Discrete density contrast model
        :param drho0 : array_like
            Density contrast vector (gm/m^3)
        :param cell_size : array_like
            Physical dimension of model cells, optional (default is [1, 1, 1])
        :param narrow : bool
            Narrow band flag, optional (default is False)

        :return:
        phi : array_like.
            Array of the signed distances after calculation using the FMM
        """

        assert mod_dens_const.ndim == 3, "Density constant model array should not be flat. It should be a 3D array."

        signdist = np.zeros((drho0.size, mod_dens_const.shape[0], mod_dens_const.shape[1], mod_dens_const.shape[2]))

        # Set the cell size to default [1, 1, 1] if not provided
        if cell_size is None:
            cell_size = [1, 1, 1]

        # Loop over each density contrast value.
        for i in range(0, drho0.size):

            signdist[i][mod_dens_const != drho0[i]] = -1
            signdist[i][mod_dens_const == drho0[i]] = 1

            # Use FMM with narrow or wide banding.
            # arg. 'periodic' controls boundary conditions.
            if narrow:

                signdist[i] = skfmm.distance(signdist[i], cell_size, order=2, periodic=False, narrow=2 * cell_size[0])
                print('Calculating BAND LIMITED signed-dist. (2*dim[0] band)')
            else:
                signdist[i] = skfmm.distance(signdist[i], cell_size, order=2, periodic=False)

        return signdist

    def calc_mask_dist(phi_, distance_max_, unit_index_):
        """
        Set the mask on the distance to the interface of selected rock unit.
        The two shallowest layers of `phi_mask`
        are then set to 0 to prevent changes in the shallowest layers, and the rest are set to 1.

        By default, this function also masks the shallowest two units of the model (0th and 1st layer at the surface),
        calling the function 'calc_mask_layer', under the assumption that they are already well constrained.

        :param phi_: signed distance for rock units in the studied area
        :param distance_max_: distance threshold for calculation of mask
        :param unit_index_: index of rock unit to calculate distance from
        :return: mask with zeros farther than 'distance_max' from the interface of rock number 'unit_index'
        """

        phi_mask = phi_[unit_index_][:, :, :].copy()
        phi_mask[phi_[unit_index_][:, :, :] > - distance_max_] = 1.
        phi_mask[phi_[unit_index_][:, :, :] < - distance_max_] = 0.

        return phi_mask

    def calc_mask_layer(masked_layer, horizontal_layer_index):
        """
        Set the layer of model cells with index 'horizontal_layer_index' equal to 0.

        :param masked_layer: 3D numpy array representing the model, with shape:
        (n_vertical_layers, n_horizontal_layers_dir1, n_horizontal_layers_dir2).
        :param horizontal_layer_index: int, index of the depth layer to be masked.
        :return: None
        """

        masked_layer[horizontal_layer_index, :, :] = 0.

    assert dens_model.ndim == 3, "Density contrast model array should not be flat. It should be a 3D array."

    # Calculates the mask based on distance to outline of selected units.
    if use_mask_domain:
        # Calculate the signed distances using the density model: used to define mask controlling areas that can change.
        phi = calc_signed_distances(dens_model, np.unique(dens_model), cell_size=None, narrow=False)

        # Mask on cells farther than a certain distance to the units with index ind_unit_mask
        # In the paper example: masks values further than a certain distance away from the mantle.
        mask_modelling_domain = calc_mask_dist(phi, distance_max, unit_index_=ind_unit_mask)

        # Apply mask by default to the 2 shallowest units: assuming they are well constrained.
        calc_mask_layer(mask_modelling_domain, horizontal_layer_index=0)
        calc_mask_layer(mask_modelling_domain, horizontal_layer_index=1)

    else:
        mask_modelling_domain = np.ones_like(dens_model)
        
    if mask_first_layer:
        # TODO: mal it "first_layers"
        # Used for the example, not the Pyrenees case.
        mask_modelling_domain[0, :, :] = 0.  # First layer.
        # mask_modelling_domain[1, :, :] = 0.  # Second layer. 

    return mask_modelling_domain.flatten()


def get_rotation_matrix(use_rotation_matrix, matrix_file=None):
    """
    Reads the rotation matrix from file or defines it as identity if it is not to be used.

    :param use_rotation_matrix:
    :param matrix_file:
    :return:
    """

    if use_rotation_matrix:
        if matrix_file is None:
            raise "To use a rotation matrix, a file containing it should be provided!"
        else:
            rotation_matrix = np.loadtxt(matrix_file)
    else:
        rotation_matrix = np.identity(3)

    return rotation_matrix


def print_progressbar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Function found in (Last accessed: 11/06/2024):
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters/13685020
    Call in a loop to create terminal progress bar

    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete.
    if iteration == total:
        print()

    return


def calc_model_diff(model1, model2):
    """
    Calculates the difference between two models that should have the same number of elements.

    :param model1:
    :param model2:
    :return: The element-wise difference between model1 and model2.
    :Raises: AssertionError: If the shapes of the input arrays are different.
    """

    if model1.shape != model2.shape:
        model1 = model1.flatten()
        model2 = model2.flatten()
        warnings.warn("The arrays should have the same dimensions. We flattened them", UserWarning)
    assert model1.shape == model2.shape, "models should have the same number of elements"

    return model1 - model2


def stopping_check_progress(mvars, shpars, current_iteration_number):
    """
    Prints a progress bar and determines whether the null space navigation should stop.

    :param mvars: ModelsVariables object.
    :param shpars: SolverParameters object.
    :param current_iteration_number: int.
    :return: boolean True/False depending on the condition for stopping the process.
    """

    # Print the progress bar with some info.
    print_progressbar(current_iteration_number + 1, shpars.num_epochs,
                      prefix='Progress over the maximum number of iterations',
                      suffix='... Running ...', length=50)

    # Minimal sanity check on the values in model.
    assert np.isreal(mvars.m_nullspace_subs).all(), "Error: not all values in m_nullspace are real!"
    assert not (mvars.m_nullspace_subs == 0).all(), "Error: All values in m_nullspace are 0!"

    # Determine whether the process should continue or not, based on the max magnitude of changes in the updated model.
    if np.abs(mvars.m_beg - mvars.m_nullspace_subs).max() >= shpars.max_change:

        # Backtrack to previous model if changes exceed the maximum allowed change.
        model_downdate(mvars)
        print('Objective dfference with original reached: stopping at iteration ' + str(current_iteration_number))
        return False
    else:
        return True


def model_downdate(mvars):
    """
    Substracts the nullspace navigation perturbation delta_m to the current model m_nullspace.
    Useful to go back one model.

    :param mvars: A ModelsVariables object.
    :return: no value.
    """

    assert mvars.m_nullspace_subs.shape == mvars.delta_m.shape, "Arrays should have the same shape"

    mvars.m_nullspace_subs = mvars.m_nullspace_subs - mvars.delta_m


def assert_values(array_to_check):
    """ Series of sanity checks on input variables. """

    assert not np.isnan(array_to_check).any(), "Nan problem"
    assert not np.isinf(array_to_check).any(), "Inf problem"
    # assert not np.all(array_to_check == 0), "Zero-values only problem"


def assert_sanity(var_class):
    """
    Sanity checks on variables in class 'var_class': checks that no non-function field remains None.
    """

    # Get fields from the class.
    attrs = vars(var_class)

    # Get fields from the class that are not functions.
    non_functions = {k: v for k, v in attrs.items() if not callable(v)}

    # Loop through non-function fields for sanity check.
    for attr in non_functions:
        if not callable(getattr(var_class, attr)):
            tmp = getattr(var_class, attr)
            # Check if field is None.
            assert tmp is not None, f"{attr} is not defined!"
            # For arrays, run sanity check on Nan, Inf and determine if it is zeros-only.
            if isinstance(tmp, np.ndarray):
                assert_values(getattr(var_class, attr))

    return

