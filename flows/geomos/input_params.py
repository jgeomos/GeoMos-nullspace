import shutil
import os

from onecode import Project, file_input, checkbox, dropdown, number_input, slider


class InputParameters:
    """
    A class to contains all input parameters.
    """
    def __init__(self):
        # -------------------------------
        # Section 'FilePaths'.
        # -------------------------------
        self.model_filename = file_input(
            key='model_filename',
            value='models/model_grid.txt',
            label="Model of the area investigated for the case study"
        )
        print(self.model_filename)
        # self.model_filename = 'models/model_grid.txt'
        
        self.perturbation_filename = file_input(
            key='perturbation_filename',
            value='models/delta_m_orig.txt',
            label="Perturbation that will be added to the model"
        )
        self.perturbation_filename = 'models/delta_m_orig.txt'

        # Geophysical data, e.g., Bouguer anomaly.
        # self.data_vals_filename = file_input(
        #     key='data_vals_filename',
        #     value='gravity_data/data_vals.txt',
        #     label="File containing the gravity data"
        # )
        self.data_vals_filename = 'gravity_data/data_vals.txt'

        # Value of the background model used, e.g., in the calculation of the Bouguer anomaly.
        # self.data_background_filename = file_input(
        #     key='data_background_filename',
        #     value='gravity_data/data_background.txt',
        #     label="File containing the density response of the background model"
        # )
        self.data_background_filename = 'gravity_data/data_background.txt'

        # sensit_files = file_input(
        #     key='sensit_files',
        #     value=[
        #         'data/SENSIT/sensit_grav_5_0',
        #         'data/SENSIT/sensit_grav_5_1',
        #         'data/SENSIT/sensit_grav_5_2',
        #         'data/SENSIT/sensit_grav_5_3',
        #         'data/SENSIT/sensit_grav_5_4',
        #         'data/SENSIT/sensit_grav_5_meta.dat',
        #         'data/SENSIT/sensit_grav_5_weight',
        #         'data/SENSIT/sensit_grav_meta.txt',
        #         'data/SENSIT/sensit_grav_nnz',
        #         'data/SENSIT/sensit_grav_weight',
        #     ],
        #     multiple=True,
        #     label="Files the sensitivity matrix (input ALL files)",
        # )
        sensit_files = [
                '/data/SENSIT/sensit_grav_5_0',
                '/data/SENSIT/sensit_grav_5_1',
                '/data/SENSIT/sensit_grav_5_2',
                '/data/SENSIT/sensit_grav_5_3',
                '/data/SENSIT/sensit_grav_5_4',
                '/data/SENSIT/sensit_grav_5_meta.dat',
                '/data/SENSIT/sensit_grav_5_weight',
                '/data/SENSIT/sensit_grav_meta.txt',
                '/data/SENSIT/sensit_grav_nnz',
                '/data/SENSIT/sensit_grav_weight',
            ]
        
        self.sensit_path = Project().get_output_path('SENSIT')
        os.makedirs(self.sensit_path, exist_ok=True)
        for s_file in sensit_files:
            shutil.copyfile(
                s_file,
                os.path.join(self.sensit_path, os.path.basename(s_file))
            )

        # self.rotation_mat_filename = file_input(
        #     key='rotation_mat_filename', 
        #     value='rotation_matrix.txt',
        #     label="Path to rotation matrix",
        #     optional=True
        # )
        self.rotation_mat_filename = 'rotation_matrix.txt'

        # self.geol_model_path = file_input(
        #     key='geol_model_path',
        #     value='models/m_geol_orig.txt',
        #     label="Path to geological or other reference model, used for plots only, to add a reference more for comparison",
        #     optional=True
        # )
        self.geol_model_path = 'models/m_geol_orig.txt'

        # self.data_outline_filename = file_input(
        #     key='data_outline_filename',
        #     value='gravity_data/ouline_core_area_dots.txt',
        #     label="Path to file containing the outline of the geophysical data ",
        #     optional=True
        # )
        self.data_outline_filename = 'gravity_data/ouline_core_area_dots.txt'

        # -------------------------------
        # Section 'SolverParameters'.
        # -------------------------------
        # Flag defining if we import sensitivity kernel from Tomofast-x.
        # self.use_tomofast_sensit = checkbox(
        #     key='use_tomofast_sensit',
        #     value=True,
        #     label="Use Tomofast sensitivity"
        # )
        self.use_tomofast_sensit = True

        # String of characters determining the type of inversion / sensitivity matrix ('grav' or 'magn').
        self.sensit_type = dropdown(
            key='sensit_type',
            value='grav',
            options=['grav', 'magn'],
            label='Type of geophysical data'
        )

        # Number of procs used to calculate the sensitivity kernel with Tomofast.
        # self.tomofast_sensit_nbproc = number_input(
        #     key='tomofast_sensit_nbproc',
        #     value=5,
        #     min=1,
        #     step=1,
        #     label="Number of processors used to calculate the sensitivity kernel (with Tomofast-x)"
        # )
        self.tomofast_sensit_nbproc = 5

        # Flag defining whether we rotate the data for plotting.
        # self.use_rotation_matrix = checkbox(
        #     key='use_rotation_matrix',
        #     value=True,
        #     label="Flag defining whether we rotate the data (for plotting only)"
        # )
        self.use_rotation_matrix = True

        # Flag on unit conversion
        # self.unit_conv = checkbox(
        #     key="unit_conv",
        #     value=True,
        #     label="Unit conversion of the input gravity data (apply a 1e2 factor to do mGal)"
        # )
        self.unit_conv = True

        # Flag on whether we use a mask to reduce the domain where modifications of the model are allowed.
        self.use_mask_domain = checkbox(
            key="unit_use_mask_domainconv",
            value=True,
            label="Use a mask to control where modifications are allowed"
        )

        # Weight of prior model term. (<0: larger variations, >0: smaller variations. Depth weight can go here)
        # - 2.5e-12 for plunging crust
        # 1.e-11 for mantle chunk in axial zone.
        self.weight_prior_model = slider(
            key='weight_prior_model',
            value=-5.e-12,
            min=-1.e-11,
            max=1.e-11,
            step=1e-12
        )

        # Tolerance on misfit variations during null space navigation.
        self.eps = slider(
            key='eps',
            value=0.23,
            min=0.,
            max=1.,
            step=0.01,
            label="Tolerance on misfit variations during null space navigation"
        )

        # Maximum difference between the first model of navigation and the current model.
        self.max_change = number_input(
            key="max_change",
            value=440,
            min=0,
            step=10,
            label="Maximum difference between the first model and the current model (in kg/m^3)"
        )

        # Number of time steps.
        self.num_epochs = slider(
            key="num_epochs",
            value=350,
            min=10,
            max=1000,
            step=10,
            label="Maximum number of time steps (= modifications of the model)"
        )

        # Length of a time step (scales the perturbation at each iteration).
        self.time_step = number_input(
            key="time_step",
            value=100,
            min=0,
            step=10,
            label="Length of a time step (scales the perturbation at each iteration: small time_step = small perturbation)"
        )

        # -------------------------------
        # Section 'GridParameters'.
        # -------------------------------
        # Dimensions of the mesh
        # self.nx = number_input(
        #     key='nx',
        #     value=54,
        #     min=2,
        #     step=1,
        #     label="Mesh Dimension (Nx)"
        # )
        # self.ny = number_input(
        #     key='ny',
        #     value=68,
        #     min=2,
        #     step=1,
        #     label="Mesh Dimension (Ny)"
        # )
        # self.nz = number_input(
        #     key='nz',
        #     value=31,
        #     min=2,
        #     step=1,
        #     label="Mesh Dimension (Nz)"
        # )
        self.nx = 54
        self.ny = 68
        self.nz = 31
        # ------------------------------------
        # Section 'PreProcessingParameters'.
        # ------------------------------------
        # Index of rock unit (by increasing density value) to define the mask on perturbations. 9 = Mantle.
        self.ind_unit_mask = number_input(
            key="ind_unit_mask",
            value=9,
            min=1,
            step=1,
            label="Mask: Index of rock unit (by increasing density value) to define the mask on perturbations (in paper: 9 = Mantle)"
        )

        # Distance max in number of cells away from the outline of rock unit considered.
        # 8 in tests shown in Pyrenees paper.
        self.distance_max = number_input(
            key="distance_max",
            value=4,
            min=1,
            step=1,
            label="Mask: Distance max in number of cells away from masking rock unit"
        )

        # ------------------------------------
        # Section 'SaveOutput'.
        # ------------------------------------
        # self.save_plots = checkbox(
        #     key="save_plots",
        #     value=True,
        #     label="Will plots be saved"
        # )
        self.save_plots = True


# =============================================================================
def read_input_parameters():
    return InputParameters()
