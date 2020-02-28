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

    assert is_pathname_valid(params['General']['output_dir']) == True, sys.exit("output_dir needs to be a valid path")

    assert isinstance(params['General']['verbose'], int), sys.exit("verbose takes integer values")

    assert params['General']['verbose'] >= 0, sys.exit("verbose takes non-negative values")

    assert isinstance(params['General']['plotting'], int), sys.exit("plotting takes integer values")

    assert params['General']['plotting'] >= 0, sys.exit("plotting takes non-negative values")

    assert isinstance(params['Analysis']['Lmax'], int), sys.exit("Lmax takes integer values")

    assert params['Analysis']['Lmax'] >= 1, sys.exit("Lmax takes non-negative values")

    assert isinstance(params['Analysis']['positions'], int), sys.exit("positions takes integer values")

    assert params['Analysis']['positions'] >= 0 and params['Analysis']['positions'] <= 5, sys.exit("positions takes an integer value between 1 and 5")

    assert isinstance(params['Analysis']['injection'], int), sys.exit("injection takes integer values")

    assert params['Analysis']['injection'] >= 0 and params['Analysis']['injection'] <= 7, sys.exit("injection takes an integer value between 1 and 7")

    assert isinstance(params['Analysis']['pm_errors'], int), sys.exit("pm_errors takes integer values")

    assert params['Analysis']['pm_errors'] >= 0 and params['Analysis']['pm_errors'] <= 6, sys.exit("pm_errors takes an integer value between 1 and 6")

    assert isinstance(params['Analysis']['N_obj'], int), sys.exit("N_obj takes integer values")

    assert params['Analysis']['N_obj'] >= 1, sys.exit("N_obj takes non-negative integer values")

    assert isinstance(params['Analysis']['bunch_size'], float) or isinstance(params['Analysis']['bunch_size'], int), sys.exit("bunch_size takes numerical values")

    assert params['Analysis']['bunch_size'] >= 0., sys.exit("bunch_size takes non-negative values")

    assert isinstance(params['Analysis']['pm_noise'], float) or isinstance(params['Analysis']['pm_noise'], int), sys.exit("pm_noise takes numerical values")
    
    assert params['Analysis']['pm_noise'] >= 0., sys.exit("pm_noise takes non-negative values")

    assert params['Analysis']['vsh_basis'] == "vsh" or params['Analysis']['vsh_basis'] == "mod", sys.exit("vsh_basis takes values \"normal\" or \"mod\"")

    assert params['MCMC']['llmethod'] == "permissive" or params['MCMC']['llmethod'] == "quadratic", sys.exit("llmethod takes values \"permissive\" or \"quadratic\"")

    assert isinstance(params['MCMC']['nthreads'], int), sys.exit("nthreads takes integer values")

    assert params['MCMC']['nthreads'] >= 1, sys.exit("nthreads takes non-negative values")

    assert isinstance(params['MCMC']['nlive'], int), sys.exit("nlive takes integer values")

    assert params['MCMC']['nthreads'] >= 1, sys.exit("nlive takes non-negative values")

    assert isinstance(params['MCMC']['maxmcmc'], int), sys.exit("maxmcmc takes integer values")

    assert params['MCMC']['maxmcmc'] >= 1, sys.exit("maxmcmc takes non-negative values")

    assert isinstance(params['MCMC']['prior_bounds'], float) or isinstance(params['MCMC']['prior_bounds'], int), sys.exit("prior_bounds takes numerical values")

    assert params['MCMC']['prior_bounds'] >= 0., sys.exit("prior_bounds takes non-negative values")

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

# def int_length(i):
#     return len("%i" % i)





