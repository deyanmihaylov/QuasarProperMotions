import sys
import numpy as np
import bilby
from numba import jit, vectorize

import math

from scipy.stats import norm

import AstrometricData as AD
import Utils as U

from time import time as unix

from typing import Callable

@vectorize
def error_function(x):
    return math.erf(x)

def R_values(
    data: np.array,
    invcovs: np.array,
    model: np.array,
) -> np.array:
    """
    Compute R values from data, model, and the inverse of the
    covariant matrix
    """
    M = data - model
    R_values = np.sqrt(np.einsum(
        "...i, ...ij, ...j -> ...",
        M, invcovs, M,
        optimize=['einsum_path', (0, 1), (0, 1)],
    ))

    return R_values

@jit(nopython=True, nogil=True, cache=True)
def logL_quadratic(R: np.array) -> np.array:
    """
    The normal log-likelihood
    """
    return -0.5 * (R**2)

@jit(nopython=True, nogil=True, cache=True)
def logL_permissive(R: np.array) -> np.array:
    """
    The permissive log-likelihood
    As used in Darling et al. 2018 and coming from Sivia and Skilling,
    p.168
    """
    half_R_squared = 0.5 * (R**2)
    return np.log((1.-np.exp(-half_R_squared)) / half_R_squared)

@jit(nopython=True, nogil=True, cache=True)
def logL_2Dpermissive(R: np.array) -> np.array:
    """
    The modified permissive log-likelihood for 2D data
    A generalisation of the Sivia and Skilling likelihood (p.168) for
    2D data
    """    
    return np.log(
        (np.sqrt(np.pi/2) * error_function(R/np.sqrt(2))
        - R * np.exp(-R**2/2)) / (R**3)
    )

@jit(nopython=True, nogil=True, cache=True)
def logL_goodandbad(R: np.array, beta: float, gamma: float) -> np.array:
    """
    Following the notation of Sivia and Skilling, this is "the good
    and bad data model".

    Some fraction beta of the data is assumed to come from a
    normal distribution with errors larger by a factor of gamma.
    """
    # enforce conditions 0 < beta < 1 and 1 < gamma
    # my_beta = np.clip(beta, 0., 1.)
    # my_gamma = np.clip(gamma, 1., 10.)

    # result = logsumexp(
    #     [
    #         -0.5*(R/my_gamma)**2+np.log(my_beta/my_gamma**2),
    #         -0.5*R**2+np.log(1-my_beta)
    #     ],
    #     axis=0,
    # )
        
    return np.logaddexp(
        -0.5 * (R/gamma)**2 + np.log(beta/gamma**2),
        -0.5 * R**2 + np.log(1 - beta)
    )

def generate_model(almQ: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """
    Generate model of proper motions from an array of almQ
    coefficients and some spherical harmonics basis

    INPUTS
    ------
    almQ: np.ndarray
    	Array containing the vector spherical harmonics coefficients
    basis: np.ndarray
    	Array containing the vector spherical harmonics
    """
    model = np.einsum(
        "i, ijk -> jk",
        almQ, basis,
        optimize=['einsum_path', (0, 1)],
    )
    return model

# class model(cpnest.model.Model):
#     """
#     Model to fit to the proper motions
#     """

#     def __init__(
#         self,
#         ADf: AD.AstrometricDataframe,
#         params: dict,
#     ):
#         """
#         Initialise the model class

#         INPUTS
#         ------
#         prior_bound: float
#             the range of coefficients to search over [mas/yr]
#         logL_method: str
#             which likelihood function to use, one of
#             ["permissive", "quadratic", "2Dpermissive", "goodandbad"]
#         beta, gamma: float
#             If using logL_method is "goodandbad" then
#             must provide beta in range (0,1) and gamma > 1
#         """

#         self.tol = 1e-5

#         self.lmQ_ordered = ADf.lmQ_ordered
#         self.names = list(ADf.almQ_names.values())

#         self.names_ordered = [
#             ADf.almQ_names[lmQ] for lmQ in ADf.lmQ_ordered
#         ]

#         logL_method = params["logL_method"]
#         prior_bounds = params["prior_bounds"]

#         if logL_method == "permissive":
#             self.logL = logL_permissive
#         elif logL_method == "2Dpermissive":
#             self.logL = logL_2Dpermissive
#         elif logL_method == "quadratic":
#             self.logL = logL_quadratic
#         elif logL_method == "goodandbad":
#             self.logL = logL_goodandbad
#         else:
#             sys.exit("Oh dear. This doesn't look good.")
        
#         self.logL_method = logL_method

#         self.proper_motions = ADf.proper_motions
#         self.inv_proper_motion_error_matrix = ADf.inv_proper_motion_error_matrix

#         self.basis_ordered = np.array([
#             ADf.basis[lmQ] for lmQ in ADf.lmQ_ordered
#         ])

#         self.which_basis = ADf.which_basis
#         self.overlap_matrix_Cholesky = ADf.overlap_matrix_Cholesky

#         self.bounds = [[-prior_bounds, prior_bounds] for name in self.names]

#         if logL_method == "goodandbad":
#             self.names.extend(["log10_beta", "log10_gamma"])
#             self.bounds.extend([[-1.78, -1.20], [-0.08, 0.52]])

#             self.beta_prior = norm(np.log10(0.03165), 0.05)
#             self.gamma_prior = norm(np.log10(1.6596), 0.05)

#         U.logger("Searching over the following parameters:")
#         U.logger(", ".join(self.names_ordered))

#     def log_prior(
#         self,
#         params: dict,
#     ) -> float:
#         """
#         The log-prior function
#         """

#         if self.logL_method == "goodandbad":
#             log_prior = self.beta_prior.logpdf(params["log10_beta"])
#             log_prior += self.gamma_prior.logpdf(params["log10_gamma"])
#         else:
#             log_prior = 0

#         return log_prior

#     def log_likelihood(
#         self,
#         almQ: dict,
#     ) -> float:
#         """
#         The log-likelihood function
#         """
#         # START_TIME = unix()
#         almQ_ordered = np.array([almQ[name] for name in self.names_ordered])
#         model = generate_model(almQ_ordered, self.basis_ordered)

#         R = R_values(
#             self.proper_motions,
#             self.inv_proper_motion_error_matrix,
#             model,
#         )

#         R = np.maximum(R, self.tol)

#         if self.logL_method == "goodandbad":
#             beta = almQ["log10_beta"]
#             gamma = almQ["log10_gamma"]
#             log_likelihood = np.sum(self.logL(R, 10**beta, 10**gamma))
#         else:
#             log_likelihood = np.sum(self.logL(R))
        
#         # TOTAL_TIME = unix() - START_TIME
#         # print(TOTAL_TIME)
#         return log_likelihood

def select_log_likelihood_method(method: str) -> Callable:
    if method == "permissive":
        return logL_permissive
    elif method == "2Dpermissive":
        return logL_2Dpermissive
    elif method == "quadratic":
        return logL_quadratic
    elif method == "goodandbad":
        return logL_goodandbad
    else:
        sys.exit("Oh dear. This doesn't look good.")

# def sampling_priors()


class QuasarProperMotionLikelihood(bilby.Likelihood):
    def __init__(
        self,
        ADf: AD.AstrometricDataframe,
        params: dict,
    ):
        """
        Initialise the model class

        INPUTS
        ------
        prior_bound: float
            the range of coefficients to search over [mas/yr]
        logL_method: str
            which likelihood function to use, one of
            ["permissive", "quadratic", "2Dpermissive", "goodandbad"]
        beta, gamma: float
            If using logL_method is "goodandbad" then
            must provide beta in range (0,1) and gamma > 1
        """

        self.tol = 1e-5

        self.lmQ_ordered = ADf.lmQ_ordered
        self.names = list(ADf.almQ_names.values())

        self.names_ordered = [
            ADf.almQ_names[lmQ] for lmQ in ADf.lmQ_ordered
        ]

        init_params = {name: None for name in self.names_ordered}

        self.log_L_method_name = params["logL_method"]
        prior_bounds = params["prior_bounds"]

        self.logL = select_log_likelihood_method(self.log_L_method_name)

        # if logL_method == "permissive":
        #     self.logL = logL_permissive
        # elif logL_method == "2Dpermissive":
        #     self.logL = logL_2Dpermissive
        # elif logL_method == "quadratic":
        #     self.logL = logL_quadratic
        # elif logL_method == "goodandbad":
        #     self.logL = logL_goodandbad
        # else:
        #     sys.exit("Oh dear. This doesn't look good.")
        
        # self.logL_method = logL_method

        self.proper_motions = ADf.proper_motions
        self.inv_proper_motion_error_matrix = ADf.inv_proper_motion_error_matrix

        self.basis_ordered = np.array([
            ADf.basis[lmQ] for lmQ in ADf.lmQ_ordered
        ])

        self.which_basis = ADf.which_basis
        self.overlap_matrix_Cholesky = ADf.overlap_matrix_Cholesky

        self.bounds = [[-prior_bounds, prior_bounds] for name in self.names]

        if self.log_L_method_name == "goodandbad":
            self.names.extend(["log10_beta", "log10_gamma"])
            self.bounds.extend([[-1.78, -1.20], [-0.08, 0.52]])

            self.beta_prior = norm(np.log10(0.03165), 0.05)
            self.gamma_prior = norm(np.log10(1.6596), 0.05)

        U.logger("Searching over the following parameters:")
        U.logger(", ".join(self.names_ordered))
        
        super().__init__(parameters=init_params)

    def log_likelihood(self) -> float:
        """
        The log-likelihood function
        """
        almQ_ordered = np.array([
            self.parameters[name] for name in self.names_ordered
        ])
        model = generate_model(almQ_ordered, self.basis_ordered)

        R = R_values(
            self.proper_motions,
            self.inv_proper_motion_error_matrix,
            model,
        )
        R = np.maximum(R, self.tol)

        if self.log_L_method_name == "goodandbad":
            beta = almQ["log10_beta"]
            gamma = almQ["log10_gamma"]
            log_likelihood = np.sum(self.logL(R, 10**beta, 10**gamma))
        else:
            log_likelihood = np.sum(self.logL(R))
        
        return log_likelihood
