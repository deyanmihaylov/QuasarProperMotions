import os
import sys
import numpy as np
import errno
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

def assert_config_params(
        params: dict
    ) -> None:
    """
    Perform checks that parameters are valid.
    """

    # output_dir should be a valid path in the user's OS
    assert (
        is_pathname_valid(params['General']['output_dir']) == True
    ), sys.exit("output_dir needs to be a valid path")

    # verbose should be a non-negative integer
    assert isinstance(
        params['General']['verbose'],
        int
    ), sys.exit("verbose takes integer values")

    assert (
        params['General']['verbose'] >= 0
    ), sys.exit("verbose takes non-negative values")

    # plotting should be a non-negative integer
    assert isinstance(
        params['General']['plotting'],
        int
    ), sys.exit("plotting takes integer values")

    assert (
        params['General']['plotting'] >= 0
    ), sys.exit("plotting takes non-negative values")

    # Lmax should be a positive integer
    assert isinstance(
        params['Data']['Lmax'],
        int
    ), sys.exit("Lmax takes integer values")

    assert (
        params['Data']['Lmax'] >= 1
    ), sys.exit("Lmax takes non-negative values")

    # N_obj should be a positive integer
    assert isinstance(
        params['Data']['N_obj'],
        int
    ), sys.exit("N_obj takes integer values")

    assert (
        params['Data']['N_obj'] >= 1
    ), sys.exit("N_obj takes non-negative integer values")

    # positions should be an integer between 1 and 5
    assert isinstance(
        params['Data']['positions'],
        int
    ), sys.exit("positions takes integer values")

    assert (
        params['Data']['positions'] >= 1
        and
        params['Data']['positions'] <= 5
    ), sys.exit("positions takes an integer value between 1 and 5")

    # positions_method should be one of ["uniform", "bunched"]
    assert (
        params['Data']['positions_method'] in ["uniform", "bunched"]
    ), sys.exit("positions_method takes values \"uniform\" or \"bunched\"")

    # positions_seed should be an int
    assert isinstance(
        params['Data']['positions_seed'],
        int
    ), sys.exit("positions_seed takes integer values")

    # bunch_size_polar should be a non-negative number
    assert isinstance(
        params['Data']['bunch_size_polar'],
        (int, float)
    ), sys.exit("bunch_size_polar takes numerical values")

    assert (
        params['Data']['bunch_size_polar'] >= 0.
    ), sys.exit("bunch_size_polar takes non-negative values")

    # bunch_size_azimuthal should be a non-negative number
    assert isinstance(
        params['Data']['bunch_size_azimuthal'],
        (int, float)
    ), sys.exit("bunch_size_azimuthal takes numerical values")

    assert (
        params['Data']['bunch_size_azimuthal'] >= 0.
    ), sys.exit("bunch_size_azimuthal takes non-negative values")

    # proper_motions should be an integer between 1 and 5
    assert isinstance(
        params['Data']['proper_motions'],
        int
    ), sys.exit("proper_motions takes integer values")

    assert (
        params['Data']['proper_motions'] >= 1
        and
        params['Data']['proper_motions'] <= 5
    ), sys.exit("proper_motions takes an integer value between 1 and 5")

    # proper_motions_method should be one of ["zero", "dipole", "multipole"]
    assert (
        params['Data']['proper_motions_method']
        in
        ["zero", "dipole", "multipole"]
    ), sys.exit("proper_motions_method takes values \"zero\", \"dipole\" or \"multipole\"")

    # proper_motions_seed should be an int
    assert isinstance(
        params['Data']['proper_motions_seed'],
        int
    ), sys.exit("proper_motions_seed takes integer values")

    # dipole should be a non-negative number (check only if proper_motions_method is "dipole")
    if params['Data']['proper_motions_method'] == "dipole":
        assert isinstance(
            params['Data']['dipole'],
            (int, float)
        ), sys.exit("dipole takes numerical values")

        assert (
            params['Data']['dipole'] >= 0.
        ), sys.exit("dipole takes non-negative values")

    # multipole should be a list of non-negative numbers with length equal to Lmax (check only if proper_motions_method is "multipole")
    if params['Data']['proper_motions_method'] == "multipole":
        assert isinstance(
            params['Data']['multipole'],
            list
        ), sys.exit("multipole takes a list of numbers")

        assert (
            len(params['Data']['multipole']) == params['Data']['Lmax']
        ), sys.exit("The size of multipole needs to match Lmax")

        for x in params['Data']['multipole']:
            assert isinstance(
                x,
                (int, float)
            ), sys.exit("multipole takes a list of numbers")

    # proper_motion_errors should be an integer between 1 and 5
    assert isinstance(
        params['Data']['proper_motion_errors'],
        int
    ), sys.exit("proper_motion_errors takes integer values")

    assert (
        params['Data']['proper_motion_errors'] >= 1
        and
        params['Data']['proper_motion_errors'] <= 5
    ), sys.exit("proper_motion_errors takes an integer value between 1 and 5")

    if params['Data']['proper_motion_errors'] == 1:
        # proper_motion_errors_method should be one of ["flat", "adaptive"]
        assert (
            params['Data']['proper_motion_errors_method']
            in
            ["flat", "adaptive"]
        ), sys.exit("proper_motion_errors_method takes values \"flat\", \"adaptive\"")

        # proper_motion_errors_std should be a positive number
        assert isinstance(
            params['Data']['proper_motion_errors_std'],
            (int, float)
        ), sys.exit("proper_motion_errors_std takes numerical values")

        assert (
            params['Data']['proper_motion_errors_std'] > 0.
        ), sys.exit("proper_motion_errors_std takes positive values")

        # proper_motion_errors_corr should be a number in the interval (-1, 1)
        assert isinstance(
            params['Data']['proper_motion_errors_corr'],
            (int, float)
        ), sys.exit("proper_motion_errors_corr takes numerical values")

        assert (
            params['Data']['proper_motion_errors_corr'] > -1.
            and
            params['Data']['proper_motion_errors_corr'] < 1.
        ), sys.exit("proper_motion_errors_corr takes values in the interval (-1, 1)")

    # proper_motion_noise should be a non-negative number
    if 'proper_motion_noise' in params['Data']:
        assert isinstance(
            params['Data']['proper_motion_noise'],
            (int, float)
        ), sys.exit("proper_motion_noise takes numerical values")

        assert (
            params['Data']['proper_motion_noise'] >= 0.
        ), sys.exit("proper_motion_noise takes non-negative values")

    # proper_motion_noise_seed should be an int
    if 'proper_motion_noise_seed' in params['Data']:
        assert isinstance(
            params['Data']['proper_motion_noise_seed'],
            int
        ), sys.exit("proper_motion_noise_seed takes integer values")

    # dimensionless_proper_motion_threshold should be positive
    if 'dimensionless_proper_motion_threshold' in params['Data']:
        assert isinstance(
            params['Data']['dimensionless_proper_motion_threshold'],
            (int, float)
        ), sys.exit("dimensionless_proper_motion_threshold takes numerical values")

        assert (
            params['Data']['dimensionless_proper_motion_threshold'] > 0.
        ), sys.exit("dimensionless_proper_motion_threshold takes positive values")

    # vsh_basis should be one of ["vsh", "orthogonal"]
    assert (
        params['Data']['basis'] in ["vsh", "orthogonal"]
    ), sys.exit("vsh_basis takes values \"vsh\" or \"orthogonal\"")

    # ll_method should be one of ["permissive", "quadratic"]
    assert (
        params['MCMC']['logL_method'] in ["quadratic", "permissive", "2Dpermissive", "goodandbad"]
    ), sys.exit("llmethod takes values \"quadratic\" or \"permissive\" or \"2Dpermissive\" or \"goodandbad\"")

    # nthreads should be a positive integer
    assert isinstance(
        params['MCMC']['nthreads'],
        int
    ), sys.exit("nthreads takes integer values")

    assert (
        params['MCMC']['nthreads'] >= 1
    ), sys.exit("nthreads takes positive values")

    # nlive should be a positive integer
    assert isinstance(
        params['MCMC']['nlive'],
        int
    ), sys.exit("nlive takes integer values")

    assert (
        params['MCMC']['nlive'] >= 1
    ), sys.exit("nlive takes positive values")

    # maxmcmc should be a positive integer
    assert (
        isinstance(params['MCMC']['maxmcmc'], int)
    ), sys.exit("maxmcmc takes integer values")

    assert (
        params['MCMC']['maxmcmc'] >= 1
    ), sys.exit("maxmcmc takes non-negative values")

    # prior_bounds should be a positive number
    assert isinstance(
        params['MCMC']['prior_bounds'], (int, float)
    ), sys.exit("prior_bounds takes numerical values")

    assert (
        params['MCMC']['prior_bounds'] > 0.
    ), sys.exit("prior_bounds takes positive values")

    # pol should be one of ["GR", "B"]
    assert (
        params['Post_processing']['pol'] in ["GR", "B"]
    ), sys.exit("pol takes values \"GR\" or \"B\"")

    # limit should be a number between 0 and 100
    assert isinstance(
        params['Post_processing']['limit'], (int, float)
    ), sys.exit("limit takes numerical values")

    assert (
        params['Post_processing']['limit'] >= 0.
        and
        params['Post_processing']['limit'] <= 100.
    ), sys.exit("limit takes values between 0 and 100")

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

def deg_to_rad(
        degree_vals
    ):
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
        norm_exponent = 1. / (2.*L*(L+2))

    norm_matrix = (1. / np.linalg.det(matrix)**norm_exponent) * matrix

    return norm_matrix

def chi_squared_limit(k, P):
    """
    Find the P-percent certainty limit of the chi-squared distribution

    INPUTS
    ------
    k: int
        number of dimensions of the distribution

    P: float
        certainty of the distrubtion, in percents

    RETURNS
    -------
    limit: float
        limit of the distribution
    """
    def CDF(x):
        return chi2.cdf(x, k) - P/100.

    limit = broyden1(CDF, k, f_tol=1e-10)

    return limit

def generalized_chi_squared_limit(k, A, P, N=1000000):
    """
    Find the P-percent certainty limit of the generalized chi-squared distribution

    INPUTS
    ------
    k: int
        number of dimensions of the distribution

    A: np.array shape=(k,k)

    P: float
        certainty of the distrubtion, in percents

    N: int
        number of random draws

    RETURNS
    -------
    limit: float
        limit of the distribution

    TO DO: rewrite this with a CDF instead of this brute force
    """
    z = np.random.normal(size=(N,k))

    samples = np.einsum("...i,...ij,...j->...", z, A, z)

    limit = np.percentile(samples, P)

    return limit

def export_data(
        ADf: AD.AstrometricDataframe,
        limit: float,
        output: str
    ):

    positions_file_name = os.path.join(output, 'positions.dat')
    np.savetxt(positions_file_name, ADf.positions)

    proper_motions_file_name = os.path.join(output, 'proper_motions.dat')
    np.savetxt(proper_motions_file_name, ADf.proper_motions)

    overlap_matrix_file_name = os.path.join(output, 'overlap_matrix.dat')
    np.savetxt(overlap_matrix_file_name, ADf.overlap_matrix)

    limit_file_name = os.path.join(output, 'limit.dat')
    np.savetxt(limit_file_name, np.array([limit]))
