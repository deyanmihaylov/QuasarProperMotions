import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from scipy.special import binom
from mpmath import hyp3f2

import Utils as U

def flatten(t: list) -> list:
    return [item for sublist in t for item in sublist]

def C_l_GR(l: int) -> float:
    """
    Implementing eq. (C2) from arXiv:1911.10356 (https://arxiv.org/abs/1911.10356)
    multiplied by an overall factor of 4 * np.pi
    
    Inputs:
        l: the mode number \ell
        
    Returns:
        C_l, the correlation coefficient
    """
    assert l >= 1
    
    C_l = (4 * np.pi**2 * 2**(l+1) / (3*l*(l+1)) *
           np.sum(
               [
                   binom(l, k) * binom((l+k-1)/2, l) *
                   (
                       (-1)**l * (2*k-1) +
                       13 + 4*k -
                       12 * (k+2) * hyp3f2(1,1,-k-1,2,2,2)
                   ) /
                   ((k+2)*(k+1))
                   for k in range(0, l+1)
               ]
           )
          )
    
    return np.float64(C_l)

def C_l_S_E(l: int) -> float:
    """
    """
    assert l >= 1
    
    if l == 1:
        return 8 * np.pi**2 / 9
    else:
        return 0

def C_l(
    l: int,
    P: str,
    Q: str,
) -> float:
    assert l >= 1, sys.exit("The spectral index l must be >= 1.")
    assert P in ["GR", "S"], sys.exit(f"Polarization {pol} is not implemented.")
    assert Q in ['E', 'B'], sys.exit(f"{Q} must be one of ['E', 'B'].")
    
    if P == "GR":
        if Q == 'E':
            return C_l_GR(l)
        elif Q == 'B':
            return C_l_GR(l)
    elif P == "S":
        if Q == 'E':
            return C_l_S_E(l)
        elif Q == 'B':
            return 0

def compute_A_posterior(
    pol: str,
    Lmax: int,
    almQ: np.array,
) -> np.array:
    N = 2 * Lmax * (Lmax+2)

    C_E = flatten([[C_l(l, pol, "E")] * (2*l+1) for l in range(1, Lmax+1)])
    C_B = flatten([[C_l(l, pol, "B")] * (2*l+1) for l in range(1, Lmax+1)])

    C = flatten([[C_E[i], C_B[i]] for i in range(0, Lmax*(Lmax+2))])

    diag_of_M = [0 if C[i] == 0 else 1/C[i] for i in range(0, len(C))]
    M = np.diag(diag_of_M)

    Q = np.einsum('...i,ij,...j->...', almQ, M, almQ)

    A_prior = np.linspace(0., np.sqrt(Q.max()), num=1000)
    A_prior = np.delete(A_prior, 0)

    P_A_given_D = np.array([np.sum(np.exp(- N*np.log(A) - Q/(2.*(A**2.)))) for A in A_prior])
    P_A_given_D = P_A_given_D / np.trapz(P_A_given_D, A_prior)

    return P_A_given_D

def post_process_results(
        posterior_file: str,
        Lmax: int,
    ) -> float:
    """
    Post process the results from the nested sampling

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

    # do the GR background first
    if Lmax >= 2:
        Cs_GR_E = flatten([[C_l(l, "GR", "E")] * (2*l+1) for l in range(1, Lmax+1)])
        Cs_GR_B = flatten([[C_l(l, "GR", "B")] * (2*l+1) for l in range(1, Lmax+1)])
        
        Cs_GR = flatten([[Cs_GR_E[i], Cs_GR_B[i]] for i in range(0, Lmax*(Lmax+2))])
        
        diag_of_M = [0 if Cs_GR[i] == 0 else 1/Cs_GR[i] for i in range(0, len(Cs_GR))]
        M = np.diag(diag_of_M)

        Q = np.einsum('...i,ij,...j->...', almQ_posterior_samples, M, almQ_posterior_samples)

        A_prior = np.linspace(0., np.sqrt(Q.max()), num=1000)
        A_prior = np.delete(A_prior, 0)

        P_A_given_D = np.array([np.sum(np.exp(- N_cols*np.log(A) - Q/(2.*(A**2.)))) for A in A_prior])
        P_A_given_D = P_A_given_D / np.trapz(P_A_given_D, A_prior)

        A_posterior = compute_A_posterior("GR", Lmax, almQ_posterior_samples)

        plt.figure(figsize=(8,6))
        plt.plot(A_prior, P_A_given_D)
        plt.vlines(0.005318, 1e-12, 1e3)
        plt.xlabel('A')
        plt.ylabel('P(A|D)')
        plt.xlim((0,np.sqrt(Q.max())))
        plt.ylim((1e-12,1e3))
        plt.title('Amplitude of the SGWB')
        plt.yscale('log')

        plt.show()

        print(P_A_given_D-A_posterior)
        exit()

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

# def post_process_results_old(
#         posterior_file: str,
#         which_basis: str,
#         Lmax: int,
#         L: np.ndarray,
#         pol: str,
#         limit: float
#     ) -> float:
#     """
#     Post process CPNest results

#     INPUTS
#     ------
#     post_process_results: str
#         the path to the posterior.dat file produced by CPNest
#     mod_basis: bool
#         whether the modified basis of functions is used

#     """
#     with open(posterior_file) as f:
#         coeff_names = f.readline().split()[1:-2]

#     N_cols = 2 * Lmax * (Lmax+2)

#     almQ_posterior_data = np.loadtxt(posterior_file)
#     almQ_posterior_samples = almQ_posterior_data[:, 0:N_cols]

#     if pol == "GR":
#         assert Lmax>=2
#         diag_of_M = [[0. if C_l_GR(l) == 0 else 1./C_l_GR(l)] * 2*(2*l+1) for l in range(1, Lmax+1)]
#     elif pol == "B":
#         diag_of_M = [[0. if C_l_B(l) == 0 else 1./C_l_B(l), 0.] * (2*l+1) for l in range(1, Lmax+1)]

#     diag_of_M_flat = [coeff for coeffs in diag_of_M for coeff in coeffs]
#     M = np.diag(diag_of_M_flat)

#     Q = np.einsum('...i,ij,...j->...', almQ_posterior_samples, M, almQ_posterior_samples)

#     Q_limit = np.percentile(Q, limit)

#     if which_basis == "vsh":
#         chi_squared_limit = U.chi_squared_limit(len(coeff_names), limit)

#         A_limit = np.sqrt(Q_limit/chi_squared_limit)

#     elif which_basis == "orthogonal":
#         X = np.einsum("li,lk,kj->ij", L, M, L)

#         generalized_chi_squared_limit = U.generalized_chi_squared_limit(len(coeff_names), X, limit)

#         A_limit = np.sqrt(Q_limit/generalized_chi_squared_limit)

#     return A_limit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post process results from the nested sampling.")
    parser.add_argument(
        "posterior_file",
        type=str,
        help="path to posterior file",
    )
    parser.add_argument(
        "Lmax",
        type=int,
        help="maximum value of L",
    )
    args = parser.parse_args()

    post_process_results(
        args.posterior_file,
        args.Lmax,
    )