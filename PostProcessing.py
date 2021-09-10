import os
import sys
import numpy as np
import matplotlib.pyplot as plt

import Utils as U

def C_l_GR(l: int) -> float:
    """
    Get the values of the correlation coefficients for a GR background

    INPUTS
    ------
    l: int
        the spectral mode
    """

    GR_coeffs_file_path = "./src/gr_coeffs.dat"

    if not os.path.isfile(GR_coeffs_file_path):
        sys.exit("The file gr_coeffs.dat cannot be found.")

    try:
        C_l = np.loadtxt(
                        GR_coeffs_file_path,
                        skiprows = 0,
                        usecols = None,
                        unpack=False,
                        ndmin=1,
                        encoding='bytes',
                        max_rows = None
                    )
    except:
        sys.exit("The file gr_coeffs.dat cannot be accessed.")

    assert l <= C_l.shape[0], sys.exit(f"The GR correlation coefficient for l = {l} is not available.")

    return C_l[l-1]

def C_l_B(l: int) -> float:
    """
    Get the values of the correlation coefficients for a Breathing mode background

    INPUTS
    ------
    l: int
        the spectral mode
    """
    if l == 1:
        return 8.77298
    else:
        return 0.

def post_process_results(
        posterior_file: str,
        which_basis: str,
        Lmax: int,
        L: np.ndarray,
        pol: str,
        limit: float
    ) -> float:
    """
    Post process CPNest results

    INPUTS
    ------
    post_process_results: str
        the path to the posterior.dat file produced by CPNest
    mod_basis: bool
        whether the modified basis of functions is used

    """
    with open(posterior_file) as f:
        coeff_names = f.readline().split()[1:-2]

    N_cols = 2 * Lmax * (Lmax+2)

    almQ_posterior_data = np.loadtxt(posterior_file)
    almQ_posterior_samples = almQ_posterior_data[:, 0:N_cols]

    if pol == "GR":
        assert Lmax>=2, sys.exit("Lmax for GR background needs to be larger than, or equal to 2")
        diag_of_M = [[0. if C_l_GR(l) == 0 else 1./C_l_GR(l)] * 2*(2*l+1) for l in range(1, Lmax+1)]
    elif pol == "B":
        diag_of_M = [[0. if C_l_B(l) == 0 else 1./C_l_B(l), 0.] * (2*l+1) for l in range(1, Lmax+1)]

    diag_of_M_flat = [coeff for coeffs in diag_of_M for coeff in coeffs]
    M = np.diag(diag_of_M_flat)

    Q = np.einsum('...i,ij,...j->...', almQ_posterior_samples, M, almQ_posterior_samples)

    A_prior = np.random.uniform(
        low=0.,
        high=np.sqrt(Q.max()),
        size=100000
    )

    P_A_given_D = np.array([np.sum(np.exp(- N_cols*np.log(A) - Q/(2.*(A**2.)))) for A in A_prior])

    P_sum = P_A_given_D.sum()

    P_A_given_D = P_A_given_D / P_sum

    # plt.hist(P_A_given_D)
    # plt.xlabel('A')
    # plt.ylabel('P(A|D)')
    # plt.title('Histogram of A (amplitude of the SGWB)')
    # plt.yscale('log')

    # outfile = "./hist_A.png"

    # plt.tight_layout()
    # plt.savefig(outfile)
    # plt.clf()

    A_limit = np.percentile(P_A_given_D, limit)

    return A_limit

def post_process_results_old(
        posterior_file: str,
        which_basis: str,
        Lmax: int,
        L: np.ndarray,
        pol: str,
        limit: float
    ) -> float:
    """
    Post process CPNest results

    INPUTS
    ------
    post_process_results: str
        the path to the posterior.dat file produced by CPNest
    mod_basis: bool
        whether the modified basis of functions is used

    """
    with open(posterior_file) as f:
        coeff_names = f.readline().split()[1:-2]

    N_cols = 2 * Lmax * (Lmax+2)

    almQ_posterior_data = np.loadtxt(posterior_file)
    almQ_posterior_samples = almQ_posterior_data[:, 0:N_cols]

    if pol == "GR":
        assert Lmax>=2
        diag_of_M = [[0. if C_l_GR(l) == 0 else 1./C_l_GR(l)] * 2*(2*l+1) for l in range(1, Lmax+1)]
    elif pol == "B":
        diag_of_M = [[0. if C_l_B(l) == 0 else 1./C_l_B(l), 0.] * (2*l+1) for l in range(1, Lmax+1)]

    diag_of_M_flat = [coeff for coeffs in diag_of_M for coeff in coeffs]
    M = np.diag(diag_of_M_flat)

    Q = np.einsum('...i,ij,...j->...', almQ_posterior_samples, M, almQ_posterior_samples)

    Q_limit = np.percentile(Q, limit)

    if which_basis == "vsh":
        chi_squared_limit = U.chi_squared_limit(len(coeff_names), limit)

        A_limit = np.sqrt(Q_limit/chi_squared_limit)

    elif which_basis == "orthogonal":
        X = np.einsum("li,lk,kj->ij", L, M, L)

        generalized_chi_squared_limit = U.generalized_chi_squared_limit(len(coeff_names), X, limit)

        A_limit = np.sqrt(Q_limit/generalized_chi_squared_limit)

    return A_limit
