import os
import sys
import numpy as np
import errno
import ast

from scipy.stats import chi2
from scipy.optimize import broyden1

import AstrometricData as AD

def is_pathname_valid(
        path_name: str
    ) -> bool:
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

def assert_param_group_exists(
    params:dict,
    group_name: str,
) -> None:
    assert (
        group_name in params
    ), sys.exit(f"'{group_name}' parameters group is missing")

def assert_param_exists(
    params: dict,
    group_name: str,
    param_name: str,
) -> None:
    assert (
        param_name in params[group_name]
    ), sys.exit(f"'{param_name}' parameter is missing")

def assert_valid_path(
    params: dict,
    group_name: str,
    param_name: str,
) -> None:
    assert (
        is_pathname_valid(params[group_name][param_name]) == True
    ), sys.exit(f"'{param_name}' needs to be a valid path")

def assert_int(
    params: dict,
    group_name: str,
    param_name: str,
) -> None:
    assert isinstance(
        params[group_name][param_name],
        int
    ), sys.exit(f"'{param_name}' takes integer values")

def assert_numerical(
    params: dict,
    group_name: str,
    param_name: str,
) -> None:
    assert isinstance(
        params[group_name][param_name],
        float
    ), sys.exit(f"'{param_name}' takes numerical values")

def assert_non_negative(
    params: dict,
    group_name: str,
    param_name: str,
) -> None:
    assert (
        params[group_name][param_name] >= 0
    ), sys.exit(f"'{param_name}' takes non-negative values")

def assert_positive(
    params: dict,
    group_name: str,
    param_name: str,
) -> None:
    assert (
        params[group_name][param_name] > 0
    ), sys.exit(f"'{param_name}' takes positive values")

def assert_greater_than_or_equal(
    params: dict,
    group_name: str,
    param_name: str,
    val: float,
) -> None:
    assert (
        params[group_name][param_name] >= val
    ), sys.exit(f"'{param_name}' takes values greater than or equal to {val}")

def assert_greater_than(
    params: dict,
    group_name: str,
    param_name: str,
    val: float,
) -> None:
    assert (
        params[group_name][param_name] > val
    ), sys.exit(f"'{param_name}' takes values greater than {val}")

def assert_less_than_or_equal(
    params: dict,
    group_name: str,
    param_name: str,
    val: float,
) -> None:
    assert (
        params[group_name][param_name] <= val
    ), sys.exit(f"'{param_name}' takes values less than or equal to {val}")

def assert_less_than(
    params: dict,
    group_name: str,
    param_name: str,
    val: float,
) -> None:
    assert (
        params[group_name][param_name] < val
    ), sys.exit(f"'{param_name}' takes values less than {val}")

def assert_in_list(
    params: dict,
    group_name: str,
    param_name: str,
    val_list: list,
) -> None:
    assert (
        params[group_name][param_name] in val_list
    ), sys.exit(f"'{param_name}' takes one of [{', '.join(val_list)}] as value")

def assert_config_params(
    params: dict,
) -> None:
    """
    Perform checks that parameters are valid.
    """
    
    # General parameters
    assert_param_group_exists(params, 'General')

    # output_dir should be a valid path in the user's OS
    assert_param_exists(params, 'General', 'output_dir')
    assert_valid_path(params, 'General', 'output_dir')

    # verbose should be a non-negative integer
    if 'verbose' in params['General']:
        assert_int(params, 'General', 'verbose')
        assert_non_negative(params, 'General', 'verbose')
    else:
        params['General']['verbose'] = 0

    # Data parameters
    assert_param_group_exists(params, 'Data')

    # Lmax should be a positive integer
    assert_param_exists(params, 'Data', 'Lmax')
    assert_int(params, 'Data', 'Lmax')
    assert_non_negative(params, 'Data', 'Lmax')

    # positions should be an integer between 1 and 5
    assert_param_exists(params, 'Data', 'positions')
    assert_int(params, 'Data', 'positions')
    assert_greater_than_or_equal(params, 'Data', 'positions', 1)
    assert_less_than_or_equal(params, 'Data', 'positions', 5)

    if params['Data']['positions'] == 1:
        # N_obj should be a positive integer
        assert_param_exists(params, 'Data', 'N_obj')
        assert_int(params, 'Data', 'N_obj')
        assert_greater_than_or_equal(params, 'Data', 'N_obj', 1)

        # positions_seed should be an int
        if 'positions_seed' in params['Data']:
            assert_int(params, 'Data', 'positions_seed')
        else:
            params['Data']['positions_seed'] = None

        # positions_method_polar should be one of ["uniform", "bunched"]
        if 'positions_method_polar' in params['Data']:
            assert_in_list(params, 'Data', 'positions_method_polar',
                ["uniform", "bunched"]
            )
        else:
            params['Data']['positions_method_polar'] = "uniform"

        # positions_method_azimuthal should be one of ["uniform", "bunched"]
        if 'positions_method_azimuthal' in params['Data']:
            assert_in_list(params, 'Data', 'positions_method_azimuthal',
                ["uniform", "bunched"]
            )
        else:
            params['Data']['positions_method_azimuthal'] = "uniform"

        # bunch_size_polar should be a non-negative number
        if 'bunch_size_polar' in params['Data']:
            assert_numerical(params, 'Data', 'bunch_size_polar')
            assert_non_negative(params, 'Data', 'bunch_size_polar')
        else:
            params['Data']['bunch_size_polar'] = None
            
        # bunch_size_azimuthal should be a non-negative number
        if 'bunch_size_azimuthal' in params['Data']:
            assert_numerical(params, 'Data', 'bunch_size_azimuthal')
            assert_non_negative(params, 'Data', 'bunch_size_azimuthal')
        else:
            params['Data']['bunch_size_azimuthal'] = None

    # proper_motions should be an integer between 1 and 5
    assert_param_exists(params, 'Data', 'proper_motions')
    assert_int(params, 'Data', 'proper_motions')
    assert_greater_than_or_equal(params, 'Data', 'proper_motions', 1)
    assert_less_than_or_equal(params, 'Data', 'proper_motions', 5)

    if params['Data']['proper_motions'] == 1:
        # proper_motions_seed should be an int
        if 'proper_motions_seed' in params['Data']:
            assert_int(params, 'Data', 'proper_motions_seed')
        else:
            params['Data']['proper_motions_seed'] = None

        # injection should be a dict
        if 'injection' in params['Data']:
            try:
                assert isinstance(
                    params['Data']['injection'],
                    dict
                ), sys.exit(f"'injection' takes a dictionary as value")
            except (SyntaxError, ValueError, AssertionError):
                sys.exit(f"'injection' dictionary could not be parsed")
    
    # proper_motion_noise should be a non-negative number
    if 'proper_motion_noise' in params['Data']:
        assert_numerical(params, 'Data', 'proper_motion_noise')
        assert_non_negative(params, 'Data', 'proper_motion_noise')

        # proper_motion_noise_seed should be an int
        if 'proper_motion_noise_seed' in params['Data']:
            assert_int(params, 'Data', 'proper_motion_noise_seed')
        else:
            params['Data']['proper_motion_noise_seed'] = None

    # proper_motion_errors should be an integer between 1 and 5
    assert_param_exists(params, 'Data', 'proper_motion_errors')
    assert_int(params, 'Data', 'proper_motion_errors')
    assert_greater_than_or_equal(params, 'Data', 'proper_motion_errors', 1)
    assert_less_than_or_equal(params, 'Data', 'proper_motion_errors', 5)

    if params['Data']['proper_motion_errors'] == 1:
        # proper_motion_errors_method should be one of ["flat", "adaptive"]
        if 'proper_motion_errors_method' in params['Data']:
            assert_in_list(params, 'Data', 'proper_motion_errors_method',
                ["flat", "adaptive"]
            )
        else:
            params['Data']['proper_motion_errors_method'] = "flat"

        # proper_motion_errors_std should be a positive number
        assert_param_exists(params, 'Data', 'proper_motion_errors_std')
        assert_numerical(params, 'Data', 'proper_motion_errors_std')
        assert_positive(params, 'Data', 'proper_motion_errors_std')

        # proper_motion_errors_corr should be a number in the interval (-1, 1)
        if 'proper_motion_errors_corr' in params['Data']:
            assert_numerical(params, 'Data', 'proper_motion_errors_corr')
            assert_greater_than(params, 'Data', 'proper_motion_errors_corr', -1)
            assert_less_than(params, 'Data', 'proper_motion_errors_corr', 1)
        else:
            params['Data']['proper_motion_errors_corr'] = 0

    # dimensionless_proper_motion_threshold should be positive
    if 'dimensionless_proper_motion_threshold' in params['Data']:
        assert_numerical(params, 'Data', 'dimensionless_proper_motion_threshold')
        assert_positive(params, 'Data', 'dimensionless_proper_motion_threshold')
    else:
        params['Data']['dimensionless_proper_motion_threshold'] = None

    # vsh_basis should be one of ["vsh", "orthogonal"]
    if 'basis' in params['Data']:
        assert_in_list(params, 'Data', 'basis', ["vsh", "orthogonal"])
    else:
        params['Data']['basis'] = "vsh"

    # MCMC parameters
    assert_param_group_exists(params, 'MCMC')

    # logL_method should be one of ["quadratic", "permissive", "2Dpermissive", "goodandbad"]
    assert_in_list(params, 'MCMC', 'logL_method',
        ["quadratic", "permissive", "2Dpermissive", "goodandbad"]
    )

    # nthreads should be a positive integer
    assert_int(params, 'MCMC', 'nthreads')
    assert_positive(params, 'MCMC', 'nthreads')

    # nlive should be a positive integer
    assert_int(params, 'MCMC', 'nlive')
    assert_positive(params, 'MCMC', 'nlive')

    # maxmcmc should be a positive integer
    assert_int(params, 'MCMC', 'maxmcmc')
    assert_positive(params, 'MCMC', 'maxmcmc')

    # prior_bounds should be a positive number
    assert isinstance(
        params['MCMC']['prior_bounds'], (int, float)
    ), sys.exit("prior_bounds takes numerical values")

    assert (
        params['MCMC']['prior_bounds'] > 0.
    ), sys.exit("prior_bounds takes positive values")

def logger(
    message: str,
):
    print(message)

def covariant_matrix(
        errors,
        corr
    ):
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

def normalize_matrix(
        matrix,
        L=None
    ):
    """
    Normalize the overlap matrix so that the diagonals are of order 1e0.

    matrix: numpy.ndarray
        the matrix to be normalized
    """
    if L is None:
        norm_exponent = 1
    else:
        norm_exponent = 1 / (2 * L * (L+2))

    norm_matrix = (1 / np.linalg.det(matrix)**norm_exponent) * matrix
    return norm_matrix

# def chi_squared_limit(k, P):
#     """
#     Find the P-percent certainty limit of the chi-squared distribution

#     INPUTS
#     ------
#     k: int
#         number of dimensions of the distribution

#     P: float
#         certainty of the distrubtion, in percents

#     RETURNS
#     -------
#     limit: float
#         limit of the distribution
#     """
#     def CDF(x):
#         return chi2.cdf(x, k) - P/100.

#     limit = broyden1(CDF, k, f_tol=1e-10)

#     return limit

# def generalized_chi_squared_limit(k, A, P, N=1000000):
#     """
#     Find the P-percent certainty limit of the generalized chi-squared distribution

#     INPUTS
#     ------
#     k: int
#         number of dimensions of the distribution

#     A: np.array shape=(k,k)

#     P: float
#         certainty of the distrubtion, in percents

#     N: int
#         number of random draws

#     RETURNS
#     -------
#     limit: float
#         limit of the distribution

#     TO DO: rewrite this with a CDF instead of this brute force
#     """
#     z = np.random.normal(size=(N,k))

#     samples = np.einsum("...i,...ij,...j->...", z, A, z)

#     limit = np.percentile(samples, P)

#     return limit

def export_data(
    ADf: AD.AstrometricDataframe,
    limit: float,
    output: str,
):
    
    positions_file_name = os.path.join(output, 'positions.dat')
    np.savetxt(positions_file_name, ADf.positions)

    proper_motions_file_name = os.path.join(output, 'proper_motions.dat')
    np.savetxt(proper_motions_file_name, ADf.proper_motions)

    overlap_matrix_file_name = os.path.join(output, 'overlap_matrix.dat')
    np.savetxt(overlap_matrix_file_name, ADf.overlap_matrix)

    limit_file_name = os.path.join(output, 'limit.dat')
    np.savetxt(limit_file_name, np.array([limit]))
