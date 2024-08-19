from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy import sparse
import forward_calculation_utils as fw
import nullspace_utils as nu
from collections import Counter

# TODO: typing the variables, eg use typing.Union[int,float,None]


@dataclass
class ModelsVariables:
    """
    A class containing the different models and subsets of models used in the null space navigation.
    """

    # Full unperturbed initial model for beginning of null space navigation.
    m_nullspace_orig: np.array = None
    # Current model of null space nagivation, all model-cells.
    m_curr: np.array = None
    # Full model with perturbation added of complete space navigation.
    m_nullspace_last: np.array = np.array((-1., -1.))

    # Geological or other reference model used for plots.
    m_geol_orig: np.array = None  # TODO: make it optional??

    # Mask of the indices of subset within full model that are considered for null space navigation.
    domain_mask: np.array = None

    # Subset of model used for perturbation.
    m_nullspace_subs: np.array = None
    # Initial model before perturbation.
    m_beg: np.array = None

    # First perturbation of model model.
    delta_m_orig: np.array = None

    # Perturbation of current model.
    delta_m: np.array = None

    # Prevent the creation of new variables in the class after it is declared.
    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)


@dataclass
class GridParameters:
    """
    A class that initialises the parameters of the model in regular voxet form
    """

    # Coordinates of centres of model cells.
    x: np.ndarray = field(default_factory=lambda: np.array([1]))
    y: np.ndarray = field(default_factory=lambda: np.array([1]))
    z: np.ndarray = field(default_factory=lambda: np.array([1]))

    # Number of cells in x, y, and z direction.
    dim: np.ndarray = field(default_factory=lambda: np.array([-1, -1, -1]))  # or make it tuple since dim doesnt change?

    # Total number of elements calculated from dim.
    @property
    def n_el(self):
        return np.prod(self.dim)


@dataclass
class SolverParameters:
    """
    Class to store hyperparameters for the solver of null space shuttles.

    Attributes:
    eps: float,
        Tolerance on misfit variations during null space navigation.
    max_change: float,
        Maximum difference between the first model of navigation and the current model.
    num_epochs: int,
        Number of time steps. Default is 100.
    time_step: float,
        Length of a time step (scales the perturbation at each iteration). Default is 1.
    """

    # Tolerance on misfit variations during null space navigation.
    eps: float = 1.
    # Maximum difference between first model of navigation and current model.
    max_change: float = np.inf
    # Number of time steps.
    num_epochs: int = 100
    # Length of a time step (scales the perturbation at each iteration).
    time_step: float = 1
    # Prior model weight.
    weight_prior: float = 0.

    def __init__(self, eps, max_change, num_epochs, time_step, weight_prior):
        self.eps = eps
        self.max_change = max_change
        self.num_epochs = num_epochs
        self.time_step = time_step
        self.weight_prior = weight_prior

    # Prevent the creation of new variables in the class after it is declared.
    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)


class HamiltonQuantities:
    """
    A class containing and calculating the three terms of artificial Hamiltonian and a function to calculate it from
    quantities used during null space navigation.
    """

    total_energy: np.array = None
    potential_energy: np.array = None
    kinetic_energy: np.array = None

    def __init__(self, kinetic_energy, data_misfit, model_misfit, unit_conv, weight_prior):
        """
        Calculates the Hamiltonian defined as the sum of the data misfit term from geophysics' cost function and the
        kinetic energy of the particle (perturbation) used for the null space shuttles.
        Here, the data misfit term is analogous to potential energy.

        :param kinetic_energy: nd.array, kinetic energy calculated during null space shuttle navigation.
        :param data_misfit: nd.array, containing the RMS data misfit.
        :param unit_conv: flag for unit conversion SI/cgs.
        :return: hamiltonian, data misfit term as in the geophysics cost function and the kinetic energy term.
        """

        # Hamiltonian.
        self.total_energy: np.array = None
        # Data misfit term
        self.potential_energy: np.array = None
        # Kinetic energy term
        self.kinetic_energy: np.array = None

        fact = 1.
        if unit_conv:
            fact = 1e-2

        self.kinetic_energy = kinetic_energy[kinetic_energy != 0.]  # use np.nonzero ?

        # Case when no prior model accounted for.
        if weight_prior == 0:
            self.potential_energy = 0.5 * data_misfit[data_misfit != 0.] ** 2 * fact
        # Case when prior model is accounted for (weith_prior != 0).
        else:
            self.potential_energy = 0.5 * data_misfit[data_misfit != 0.] ** 2 * fact \
                                  + 0.5 * model_misfit[model_misfit != 0.]

        self.total_energy = self.kinetic_energy + self.potential_energy 
        # Could it be that the zeros are not at the same place in potential_energy data_mistfit and kinetic_energy ? 

    # Prevent the creation of new variables in the class after it is declared.
    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)


@dataclass
class SolverParameters:
    """
    Class to store hyperparameters for the solver of null space shuttles.

    Attributes:
    eps: float,
        Tolerance on misfit variations during null space navigation.
    max_change: float,
        Maximum difference between the first model of navigation and the current model.
    num_epochs: int,
        Number of time steps. Default is 100.
    time_step: float,
        Length of a time step (scales the perturbation at each iteration). Default is 1.
    """

    # Tolerance on misfit variations during null space navigation.
    eps: float = 1.
    # Maximum difference between first model of navigation and current model.
    max_change: float = np.inf
    # Number of time steps.
    num_epochs: int = 100
    # Length of a time step (scales the perturbation at each iteration).
    time_step: float = 1
    # Prior model weight.
    weight_prior: float = 0.

    def __init__(self, eps, max_change, num_epochs, time_step, weight_prior):
        self.eps = eps
        self.max_change = max_change
        self.num_epochs = num_epochs
        self.time_step = time_step
        self.weight_prior = weight_prior

    # Prevent the creation of new variables in the class after it is declared.
    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)


class NullSpaceNavigationVars:
    """
    A class that defines and calculates the different terms used for solving the Hamiltonian equations for null space
    navigation

    Attributes:
        delta_m (np.ndarray) representing the difference between the current and reference model,
        eps (float) defining the size of the perturbation to be applied.

    Methods:
        calc_mass_inv: Calculates the inverse mass matrix in the particular case where it is the identity matrix.
        calc_gamma: Calculates the gamma exponent (TODO see equations of null space navigation).
        calc_momentum: Calculates the momentum using Fichtner and Zunino 2019.
        calc_init: Runs calculation to get the different terms used at the first iteration of null space shuttles.
    """

    # Set default type of mass matrix.
    mass_matrix_type = 'identity'

    delta_m: np.array = None
    eps: float = None
    mass_inv = None
    gamma: float = None
    momentum: np.array = None

    def __init__(self, delta_m: np.ndarray, eps: float):
        self.delta_m = delta_m.flatten()
        self.eps = eps

        self.calc_init()  # TODO: is doing this OK?

    def calc_mass_inv(self):
        """
        Calculates the inverse mass matrix in the particular case where it is the identity matrix.

        :return: inverse mass matrix
        """

        if self.mass_matrix_type == 'identity':
            return csr_matrix(diags(np.ones_like(self.delta_m)))
        else:
            raise "Mass matrix supported atm: only identity or diagonal"

    def calc_gamma(self):
        """
        Calculates the gamma exponent of equation [TODO], see Giraud et al. [TODO] eq. NN

        :return: scalar value of gamma
        """

        val = np.matmul(self.delta_m.T, self.mass_inv.T.dot(self.delta_m))
        val = np.sqrt(2 * self.eps / val)

        return val

    def calc_momentum(self):
        """
        Calculates the momentum using equation [TODO] in Fichtner and Zunino 2019

        :return: momentum vector
        """

        return self.gamma * sparse.csr_matrix.dot(self.mass_inv, self.delta_m)

    def calc_init(self):
        """
        Run calculation to get the different terms used at the first iteration of null space shuttles.

        """
        self.mass_inv = self.calc_mass_inv()
        self.gamma = self.calc_gamma()
        self.momentum = self.calc_momentum()

    # Prevent the creation of new variables in the class after it is declared.
    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        super().__setattr__(name, value)


def update_momentum(momentum, leapfrog_step):
    """
    Calculate the updated momentum in eq. 6 of Fichtner and Zunino 2019

    :return: momentum for new iteration.
    """

    momentum[:] = momentum + leapfrog_step


def calc_model_perturbation(shpars, nsvars, mvars):
    """
    Calculate the term model perturbation delta_m for null space navigation from eq. 7 in Fichtner and Zunino 2019

    :param mvars: ModelsVariables object.
    :param nsvars: NullSpaceNavigationVars object.
    :param shpars: object with hyperparameters of the solver.
    :return:
    """

    mass_inv = nsvars.mass_inv
    momentum = nsvars.momentum

    mvars.delta_m = shpars.time_step * mass_inv.dot(momentum)


def model_update(mvars):
    """
    The function updates the current null space model by adding the perturbation delta_m to it.
    It raises an assertion error if the shapes of the two arrays, m_nullspace_subs and delta_m, do not match.
    It does not return any value.

    :param mvars: object of NullSpaceNavigationVars object containing the perturbation and model.
    :return:
    """

    assert mvars.m_nullspace_subs.shape == mvars.delta_m.shape, "Arrays should have the same shape"

    mvars.m_nullspace_subs = mvars.m_nullspace_subs + mvars.delta_m

    mvars.m_curr = mvars.m_nullspace_subs  # TODO: could be deleted in later versions, added for new


def calc_leapfrog_step(grav_data, sensit, mvars, shpars):
    """
    Calculate the leap from half step.

    :param grav_data: gravity data.
    :param sensit: sensitivity matrix without depth weighting.
    :param mvars: A ModelsVariables object.
    :param shpars: SolverParameters object.
    :return:
    """

    # Calculate gravity data.
    fw.calc_grav_data(grav_data, sensit, mvars.m_curr,
                         bouguer_anomaly=True, use_csr_matrix=True)

    # Calculate derivative of data cost wrt model parameters.
    grad_data_misfit = fw.calc_grad_misfit(sensit, grav_data, use_csr_matrix=True)

    # Derivative of prior model term wrt model parameters.
    grad_prior_model = mvars.m_curr.flatten()

    # Apply domain mask.
    grad_prior_model[mvars.domain_mask == 0.] = 0.
    grad_data_misfit[mvars.domain_mask == 0.] = 0.

    # Sanity check.
    nu.assert_values(grad_data_misfit)

    leapfrog_step = - 0.50 * shpars.time_step * (grad_data_misfit + shpars.weight_prior*grad_prior_model)

    return leapfrog_step


def nullspace_navigation(shpars, mvars, nsvars, sensit, grav_data):
    """
    Performs null space navigation: iterative update of the model using gradient of cost function to apply small
    perturbations of the model to maintain data misfit within a certain tolerance.

    TODO: docstring

    :param shpars: SolverParameters object.
    :param mvars: A ModelsVariables object.
    :param nsvars: NullSpaceNavigationVars object.
    :param sensit: sensitivity matrix without depth weighting.
    :param grav_data: gravity data.
    :return:
    """

    # Prior model term could be added to gradient calc.

    # Intialise quantities used to monitor the process.
    misfit_data = np.zeros(shpars.num_epochs)
    misfit_model = np.zeros(shpars.num_epochs)
    kinetic_energy = np.zeros(shpars.num_epochs)

    # ------------------------------------------------------------------------------------
    # Loop through the number of null space model perturbation, using the Leapfrog method.
    for i in range(0, shpars.num_epochs):

        # First leapfrog step.
        if i == 0:
            # Could this if statement not be removed? 
            leapfrog_step = calc_leapfrog_step(grav_data, sensit, mvars, shpars)

        # Calculate the data new momentum.
        update_momentum(nsvars.momentum, leapfrog_step)

        # Calculate perturbation of model for null space navigation.
        calc_model_perturbation(shpars, nsvars, mvars)

        # Update model using perturbation.
        model_update(mvars)

        # leapfrog step.
        leapfrog_step = calc_leapfrog_step(grav_data, sensit, mvars, shpars)

        # Calculate the data new momentum.
        update_momentum(nsvars.momentum, leapfrog_step)

        # ------------------ Convergence control and monitoring
        # Calculate gravity data misfit to monitor the process.
        misfit_data[i] = fw.calc_data_rms(grav_data)

        # Calculate term called 'kinetic energy' as in the Fichtner and Zunino 2019 paper.
        kinetic_energy[i] = calc_kinetic_energy(nsvars)

        # Calculate model misfit to monitor the process.
        misfit_model[i] = shpars.weight_prior * np.matmul(mvars.m_curr.T, mvars.m_curr)  # TODO: refactor.

        # Stopping criteria based on the magnitude of changes: stop if above certain threshold.
        if not nu.stopping_check_progress(mvars, shpars, i):
            break

        # Sanity check.
        nu.assert_sanity(mvars)
        nu.assert_sanity(nsvars)

    return grav_data, misfit_data, kinetic_energy, misfit_model, mvars


def calc_kinetic_energy(nsvars):
    """
    Using the terminology of Fichtner and Zunino 2019, Calculates the kinetic energy of a particle given its momentum
    and inverse mass matrix.

    :param nsvars: NullSpaceNavigationVars object, containing variables to define null space navigation.
    :return kinetic_energy: a float value representing the kinetic energy of the particle with momentum nsvars.momentum.
    """

    momentum = nsvars.momentum
    inverse_mass_matrix = nsvars.mass_inv

    kinetic_energy = 1 / 2 * np.matmul(momentum.T, inverse_mass_matrix.dot(momentum))

    return kinetic_energy
