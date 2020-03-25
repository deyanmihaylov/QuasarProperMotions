import numpy as np

import Utils as U

def C_l_GR(l: int) -> float:
    """
    Get the values of the correlation coefficients for a GR background

    INPUTS
    ------
    l: int
        the spectral mode

    TO DO: push this to a file
    """

    C_l = [0, 4.386490845, 0.4386490845, 0.08772981690, 0.02506566197,
           0.008952022133, 0.003730009222, 0.001740670970, 0.0008861597667,
           0.0004833598727, 0.0002788614650, 0.0001685426437, 0.0001059410903,
           0.00006886170871, 0.00004607658451, 0.00003162118544,
           0.00002219030558, 0.00001588358715, 0.00001157232778,
           8.566528356e-6, 6.433361216e-6, 4.894948752e-6,
           3.769110539e-6, 2.934107589e-6, 2.307161523e-6,
           1.831080574e-6, 1.465766469e-6, 1.182721909e-6,
           9.614384554e-7, 7.869838970e-7, 6.483674151e-7,
           5.374168414e-7, 4.479979048e-7, 3.754649107e-7,
           3.162699923e-7, 2.676822837e-7, 2.275841278e-7,
           1.943218322e-7, 1.665954245e-7, 1.433765500e-7]

    return C_l[l-1]

def C_l_B(l: int) -> float:
    """
    Get the values of the correlation coefficients for a Breatiing mode background

    INPUTS
    ------
    l: int
        the spectral mode
    """
    if l == 1:
        return 8.77298
    else:
        return 0.

def post_process_results(posterior_file: str,
                         which_basis: str,
                         Lmax: int,
                         L: np.ndarray,
                         pol: str,
                         limit: float):
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

    Ncols = 2*Lmax*(Lmax+2)

    almQ_posterior_samples = np.loadtxt(posterior_file)
    almQ_posterior_samples = almQ_posterior_samples[:, 0:Ncols]

    if pol == "GR":
        assert Lmax>=2
        diag_of_M = [[0. if C_l_GR(l) == 0 else 1./C_l_GR(l)] * 2*(2*l+1) for l in range(1, Lmax+1)]
    elif pol == "B":
        diag_of_M = [[0. if C_l_B(l) == 0 else 1./C_l_B(l), 0.] * (2*l+1) for l in range(1, Lmax+1)] # CJM: THIS IS WRONG

    diag_of_M_flat = [coeff for coeffs in diag_of_M for coeff in coeffs]
    M = np.diag(diag_of_M_flat)

    Q = np.einsum('...i,ij,...j->...', almQ_posterior_samples, M, almQ_posterior_samples)

    Q_limit = np.percentile(Q, limit)

    if which_basis == "vsh":
        chi_squared_limit = U.generalized_chi_squared_limit(len(coeff_names), M, limit)

        A_limit = np.sqrt(Q_limit/chi_squared_limit)

    elif which_basis == "orthogonal":
        X = np.einsum("li,lk,kj->ij", L, M, L)

        generalized_chi_squared_limit = U.generalized_chi_squared_limit(len(coeff_names), X, limit)

        A_limit = np.sqrt(Q_limit/generalized_chi_squared_limit)

    return A_limit
