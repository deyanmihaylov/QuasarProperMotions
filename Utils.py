import os
import sys
import numpy as np

def is_pathname_valid(path_name: str) -> bool:
    """
    True if the passed path_name is a valid path_name for the current OS;
    False otherwise.
    """

    try:
        if not isinstance(path_name, str) or not path_name:
            return False

        _, path_name = os.path.splitdrive(path_name)

        root_dirname = os.environ.get('HOMEDRIVE', 'C:') if sys.platform == 'win32' else os.path.sep
        assert os.path.isdir(root_dirname)

        root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep

        for pathname_part in path_name.split(os.path.sep):
            try:
                os.lstat(root_dirname + pathname_part)
            except OSError as exc:
                if hasattr(exc, 'winerror'):
                    if exc.winerror == ERROR_INVALID_NAME:
                        return False
                elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                    return False

    except TypeError as exc:
        return False

    else:
        return True

def assert_config_params(params):
    """
    Perform checks that parameters are valid.
    """

    # output_dir should be a valid path in the user's OS
    assert is_pathname_valid(params['General']['output_dir']) == True, sys.exit("output_dir needs to be a valid path")

    # verbose should be a non-negative integer
    assert isinstance(params['General']['verbose'], int), sys.exit("verbose takes integer values")
    assert params['General']['verbose'] >= 0, sys.exit("verbose takes non-negative values")

    # plotting should be a non-negative integer
    assert isinstance(params['General']['plotting'], int), sys.exit("plotting takes integer values")
    assert params['General']['plotting'] >= 0, sys.exit("plotting takes non-negative values")

    # Lmax should be a positive integer
    assert isinstance(params['Analysis']['Lmax'], int), sys.exit("Lmax takes integer values")
    assert params['Analysis']['Lmax'] >= 1, sys.exit("Lmax takes non-negative values")

    # N_obj should be a positive integer
    assert isinstance(params['Analysis']['N_obj'], int), sys.exit("N_obj takes integer values")
    assert params['Analysis']['N_obj'] >= 1, sys.exit("N_obj takes non-negative integer values")

    # positions should be an integer between 1 and 5
    assert isinstance(params['Analysis']['positions'], int), sys.exit("positions takes integer values")
    assert params['Analysis']['positions'] >= 1 and params['Analysis']['positions'] <= 5, sys.exit("positions takes an integer value between 1 and 5")

    # positions_method should be one of ["uniform", "bunched"]
    assert params['Analysis']['positions_method'] in ["uniform", "bunched"], sys.exit("positions_method takes values \"uniform\" or \"bunched\"")

    # bunch_size_polar should be a non-negative number
    assert isinstance(params['Analysis']['bunch_size_polar'], float) or isinstance(params['Analysis']['bunch_size_polar'], int), sys.exit("bunch_size_polar takes numerical values")
    assert params['Analysis']['bunch_size_polar'] >= 0., sys.exit("bunch_size_polar takes non-negative values")

    # bunch_size_azimuthal should be a non-negative number
    assert isinstance(params['Analysis']['bunch_size_azimuthal'], float) or isinstance(params['Analysis']['bunch_size_azimuthal'], int), sys.exit("bunch_size_azimuthal takes numerical values")
    assert params['Analysis']['bunch_size_azimuthal'] >= 0., sys.exit("bunch_size_azimuthal takes non-negative values")

    # proper_motions should be an integer between 1 and 5
    assert isinstance(params['Analysis']['proper_motions'], int), sys.exit("proper_motions takes integer values")
    assert params['Analysis']['proper_motions'] >= 1 and params['Analysis']['proper_motions'] <= 5, sys.exit("proper_motions takes an integer value between 1 and 5")

    # proper_motions_method should be one of ["zero", "dipole", "multipole"]
    assert params['Analysis']['proper_motions_method'] in ["zero", "dipole", "multipole"], sys.exit("proper_motions_method takes values \"zero\", \"dipole\" or \"multipole\"")

    # dipole should be a non-negative number
    assert isinstance(params['Analysis']['dipole'], float) or isinstance(params['Analysis']['dipole'], int), sys.exit("dipole takes numerical values")
    assert params['Analysis']['dipole'] >= 0., sys.exit("dipole takes non-negative values")

    # multipole should be a list of non-negative numbers with length equal to Lmax
    assert isinstance(params['Analysis']['multipole'], list), sys.exit("multipole takes a list of numbers")
    assert len(params['Analysis']['multipole']) == params['Analysis']['Lmax'], sys.exit("The size of multipole needs to match Lmax")
    for x in params['Analysis']['multipole']: assert isinstance(x, float) or isinstance(x, int), sys.exit("multipole takes a list of numbers")

    # proper_motion_errors should be an integer between 1 and 5
    assert isinstance(params['Analysis']['proper_motion_errors'], int), sys.exit("proper_motion_errors takes integer values")
    assert params['Analysis']['proper_motion_errors'] >= 1 and params['Analysis']['proper_motion_errors'] <= 5, sys.exit("proper_motion_errors takes an integer value between 1 and 5")

    # proper_motion_errors_method should be one of ["zero", "noise"]
    assert params['Analysis']['proper_motion_errors_method'] in ["zero", "noise"], sys.exit("proper_motion_errors_method takes values \"zero\", \"noise\"")

    # proper_motion_noise should be a non-negative number
    assert isinstance(params['Analysis']['proper_motion_noise'], float) or isinstance(params['Analysis']['proper_motion_noise'], int), sys.exit("proper_motion_noise takes numerical values")
    assert params['Analysis']['proper_motion_noise'] >= 0., sys.exit("proper_motion_noise takes non-negative values")

    # vsh_basis should be one of ["vsh", "mod"]
    assert params['Analysis']['vsh_basis'] in ["vsh", "mod"], sys.exit("vsh_basis takes values \"normal\" or \"mod\"")

    # ll_method should be one of ["permissive", "quadratic"]
    assert params['MCMC']['ll_method'] in ["permissive", "quadratic"], sys.exit("llmethod takes values \"permissive\" or \"quadratic\"")

    # nthreads should be a positive integer
    assert isinstance(params['MCMC']['nthreads'], int), sys.exit("nthreads takes integer values")
    assert params['MCMC']['nthreads'] >= 1, sys.exit("nthreads takes positive values")

    # nlive should be a positive integer
    assert isinstance(params['MCMC']['nlive'], int), sys.exit("nlive takes integer values")
    assert params['MCMC']['nlive'] >= 1, sys.exit("nlive takes positive values")

    # maxmcmc should be a positive integer
    assert isinstance(params['MCMC']['maxmcmc'], int), sys.exit("maxmcmc takes integer values")
    assert params['MCMC']['maxmcmc'] >= 1, sys.exit("maxmcmc takes non-negative values")

    # prior_bounds should be a positive number
    assert isinstance(params['MCMC']['prior_bounds'], float) or isinstance(params['MCMC']['prior_bounds'], int), sys.exit("prior_bounds takes numerical values")
    assert params['MCMC']['prior_bounds'] > 0., sys.exit("prior_bounds takes positive values")

def covariant_matrix(errors, corr):
    """
    Function for computing the covariant matrix from errors and correlations.
    """
    covariant_matrix = np.einsum('...i,...j->...ij', errors, errors)
    covariant_matrix[...,0,1] = covariant_matrix[...,1,0] = np.multiply(covariant_matrix[...,1,0], corr.flatten())
    return covariant_matrix

def deg_to_rad(degree_vals):
    """
    Does what it says on the tin
    """
    return np.deg2rad(degree_vals)

def normalize_matrix(matrix, L=None):
    """
    Normalize the overlap matrix so that the diagonals are of order 1e0.

    matrix: numpy.ndarray
        the matrix to be normalized
    """
    if Lmax is None:
        norm_exponent = 1
    else:
        norm_exponent = 1. / (2.*L*(L+2))

    norm_matrix = (1. / np.linalg.det(matrix)**norm_exponent) * matrix

    return norm_matrix


    
    

# def int_length(i):
#     return len("%i" % i)





