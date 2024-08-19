# shpars = ns.SolverParameters(eps=0.02, max_change=590, num_epochs=150, time_step=int(1e2), weight_prior=weight_prior_model))  # For tests.

# ----------------------- FROM THE nullspace_main.py file.
# - Mantle chunk in axial zone.
# shpars = ns.SolverParameters(eps=0.15, max_change=590, num_epochs=150, time_step=int(1e2), weight_prior=weight_prior_model)  # In paper, 590 for mantle.
# mvars.delta_m_orig, _ = tr.read_tomofast_model('data/models/delta_m_orig.txt', gpars)   # remove a little bit of mantle
# - Plunging slab.

# shpars = ns.SolverParameters(eps=0.25, max_change=200, num_epochs=350, time_step=int(1e2), weight_prior=weight_prior_model))  # Not In paper, 440 for axial zone.
# To remove the exhumed mantle.
# mvars.delta_m_orig, _ = tr.read_tomofast_model('data/models/delta_m_orig3.txt', gpars)
# To change the density of the axial zone.
# mvars.delta_m_orig, _  = tr.read_tomofast_model('data/models/delta_m_orig4.txt', gpars)
# To change the density of the axial zone and exhumed mantle
# mvars.delta_m_orig, _ = tr.read_tomofast_model('data/models/delta_m_orig5.txt', gpars)


# ----------------------- FROM THE nullspace_utils.py module.
# def get_cell_centers(line_data, physical_unit):
#     # TODO: check that this function is still used.
#     """
#     From line_data in the same format as tomofast grid (x1, x2, y1, y2, z1, z2), returns the coordinates of prismatic
#     cell centers of the corresponding mesh.
#
#     :param line_data: A numpy array of shape (N, 6) representing the coordinates of the edges of the cells in the format
#                       (x1, x2, y1, y2, z1, z2).
#     :param physical_unit:  A string representing the Systeme International unit of the coordinates, either 'km' or 'm'.
#     :return: Three numpy arrays of shape (N,) representing the x, y, and z coordinates of the cell centers.
#     """
#
#     if physical_unit == 'km':
#         factor = 1e-3
#     elif physical_unit == 'm':
#         factor = 1.
#     else:
#         raise Exception("Dimension problem")
#
#     x = (line_data[:, 0] + line_data[:, 1]) / 2. * factor
#     y = (line_data[:, 2] + line_data[:, 3]) / 2. * factor
#     z = (line_data[:, 4] + line_data[:, 5]) / 2. * factor
#
#     return x, y, z


# def get_rock_indices(signed_distances):
#     """
#     Takes in a np.array signed_distances and returns the indices of rock units with highest signed distances.
#
#     :param signed_distances: A numpy array containing the signed distances.
#     :return: rock_indices: A numpy array containing the indices of rock units with highest signed distances.
#     """
#
#     rock_indices = np.argmax(signed_distances, 0).copy()
#
#     return rock_indices


# def mask_domain(indices_domain: np.array, array_to_mask: np.array):  # TODO: is it good practice to add ": np.array"
#     """
#     Extracts a subset of a vector model or sensitivity matrix by masking out unwanted indices.
#
#     :param indices_domain: numpy.array, A 1D array of indices corresponding to the cells of the geological model or a 2D
#                            array of indices corresponding to the cells of the sensitivity matrix where modifications of
#                            the model by null space navigation are allowed.
#     :param array_to_mask: numpy.array, The 1D vector or 2D sensitivity matrix to mask.
#     :return: numpy.array, The model or sensitivity matrix restricted to model cells with indices `indices_domain`.
#     :raises ValueError: If the dimension of `array_to_mask` is not 1 or 2.
#     """
#
#     if array_to_mask.ndim == 2:
#         matrix_subset = array_to_mask[:, indices_domain]
#     elif array_to_mask.ndim == 1:
#         matrix_subset = array_to_mask[indices_domain]
#     else:
#         raise "Dimension problem"
#
#     return matrix_subset


# def mask_unit_diff(m0, m1, petro_vals, ind_unit_mask):
#     """
#     Gets the differences between two models `m0` and `m1` for the geological unit characterised by density
#     `dens_unit_comp`. The output is a binary mask where 1s represent locations where the two models are different for the
#     given unit.
#
#     :param m0: np.ndarray, 1D array corresponding to the first model.
#     :param m1: np.ndarray, 1D array corresponding to the second model.
#     :param petro_vals: np.ndarray, density valueS characterising for the geological units.
#     :param ind_unit_mask: int, indices of the unit to calculate the mask.
#     :return: ind_unit_mask :, Binary-valued mask of the same shape as `m0` and `m1` with 1s in locations
#     where `m0` and `m1` differ for the given unit.
#     """
#
#     dens_unit_comp = petro_vals[ind_unit_mask]
#
#     m0 = m0.flatten()
#     m1 = m1.flatten()
#
#     mask_diff_location = np.zeros_like(m0)
#
#     m0[m0 != dens_unit_comp] = 0
#     m0[m0 == dens_unit_comp] = 1
#
#     m1[m1 != dens_unit_comp] = 0
#     m1[m1 == dens_unit_comp] = 1
#
#     # Identify differences between models for mantle unit differ.
#     mask_diff_location[m0 != m1] = 1
#
#     return mask_diff_location


# def calc_mask_location(use_mask_location, mask_pert_location, mpars, ymin_mask, ymax_mask, zmin_mask):
#     """
#     TODO: can also decide a geographical extension for mask in x direction. Now only in z and y.
#
#     Sets to zeros values of mask_pert_location for values of y: mpars.y < ymax_mask  and z: mpars.z > zmin_mask
#     If Assign zeros to parts of the model outside the zone of interest defined by ymax_mask and zmin_mask
#
#     :param use_mask_location: flag controlling application of mask based on location.
#     :param mask_pert_location: numpy array containing the mask to be modified.
#     :param mpars: ModelParameters object.
#     :param ymin_mask: float, maximum y value to preserve in mask.
#     :param ymax_mask: float, maximum y value to preserve in mask.
#     :param zmin_mask: float, minimum z value to preserve in mask.
#     :return: None.
#     """
#
#     if use_mask_location:
#         # Reduce lateral extension in y.
#         mask_pert_location[mpars.y < ymin_mask] = 0
#         mask_pert_location[mpars.y > ymax_mask] = 0
#
#         # Reduce depth extension in z.
#         mask_pert_location[mpars.z > zmin_mask] = 0


# --------------------------- unpack variables from a class
# - Issue encountered: when called wihtin a function, does not work with local variables (works with global, but tricky)
# def get_field_names(obj):
#     """
#     Function to get the names of non-callable attributes of an object, excluding special methods.
#
#     :param: obj,The object to inspect.
#     :return: list of str, names of the object's non-callable, non-special attributes.
#     """
#
#     name_fields = [attr for attr in dir(obj) if not callable(getattr(obj, attr)) and not attr.startswith("__")]
#
#     return name_fields
#
#
# def set_local_variables(obj, names):
#     local_vars = {}
#     for name in names:
#         local_vars[name] = getattr(obj, name)
#
#     return local_vars
#
#
# # Get the names of the different fields.
# names = get_field_names(par)
#
# # print(names)
#
# locals_ = set_local_variables(par, names)
#
# for name, value in locals_.items():
#     exec(f"{name} = locals_['{name}']")
#     # locals()[f"{name}"] = locals_[name]
#
#     print(type(f"{name}"))
#     print('\n')
#     print('')
#


