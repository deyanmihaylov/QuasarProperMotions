import sys
import numpy as np
import numba as nb
import bilby
import math

from typing import Callable
from numpy.typing import NDArray

import AstrometricData as AD
import Utils as U

@nb.vectorize
def error_function(x):
    return math.erf(x)

def compute_R(
    data: NDArray,
    invcovs: NDArray,
    model: NDArray,
) -> NDArray:
    """
    Compute R values from data, model, and the inverse of the
    covariant matrix
    """
    M = data - model
    R = np.sqrt(np.einsum(
        "...i, ...ij, ...j -> ...",
        M, invcovs, M,
        optimize=['einsum_path', (0, 1), (0, 1)],
    ))

    return R

@nb.jit(nopython=True, nogil=True, cache=True)
def logL_quadratic(R: NDArray) -> NDArray:
    """
    The normal log-likelihood
    """
    return -0.5 * (R**2)

@nb.jit(nopython=True, nogil=True, cache=True)
def logL_permissive(R: NDArray) -> NDArray:
    """
    The permissive log-likelihood
    As used in Darling et al. 2018 and coming from Sivia and Skilling,
    p.168
    """
    half_R_squared = 0.5 * (R**2)
    return np.log((1.-np.exp(-half_R_squared)) / half_R_squared)

@nb.jit(nopython=True, nogil=True, cache=True)
def logL_2Dpermissive(R: NDArray) -> NDArray:
    """
    The modified permissive log-likelihood for 2D data
    A generalisation of the Sivia and Skilling likelihood (p.168) for
    2D data
    """    
    return np.log(
        (np.sqrt(np.pi/2) * error_function(R/np.sqrt(2))
        - R * np.exp(-R**2/2)) / (R**3)
    )

@nb.jit(nopython=True, nogil=True, cache=True)
def logL_goodandbad(R: NDArray, beta: float, gamma: float) -> NDArray:
    """
    Following the notation of Sivia and Skilling, this is "the good
    and bad data model".

    Some fraction beta of the data is assumed to come from a
    normal distribution with errors larger by a factor of gamma.
    """
    # enforce conditions 0 < beta < 1 and 1 < gamma
    if beta < 0: beta = 0
    if beta > 1: beta = 1

    if gamma < 1: gamma = 1
        
    return np.logaddexp(
        -0.5 * (R/gamma)**2 + np.log(beta/gamma**2),
        -0.5 * R**2 + np.log(1 - beta)
    )

def generate_model(almQ: NDArray, basis: NDArray) -> NDArray:
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

def sample(
    ADf: AD.AstrometricDataframe,
    params: dict,
) -> None:
    bilby.core.utils.setup_logger(
        outdir=params['General']['output_dir'],
        label="bilby_output",
    )

    names_ordered = [
        data.almQ_names[lmQ] for lmQ in data.lmQ_ordered
    ]

    if params['MCMC']["logL_method"] == "goodandbad":
        names_ordered.extend(["log10_beta", "log10_gamma"])

    priors = {
        par: bilby.core.prior.Uniform(-0.2, 0.2, par) for par in names_ordered
    }

    if params['MCMC']["logL_method"] == "goodandbad":
        priors["log10_beta"] = bilby.core.prior.Uniform(
            -1.78, -1.20, "log10_beta",
        )
        priors["log10_gamma"] = bilby.core.prior.Uniform(
            -0.08, 0.52, "log10_gamma",
        )

    likelihood = sampler.QuasarProperMotionLikelihood(
        data,
        params = params['MCMC'],
    )

    result = bilby.run_sampler(
        outdir=params['General']['output_dir'],
        label="bilby_output",
        resume=False,
        plot=True,
        likelihood=likelihood,
        priors=priors,
        sampler="nessai",
        # injection_parameters={'x': 0.0, 'y': 0.0},
        analytic_priors=False,
        seed=1234,
        nlive=1024,
    )

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

        self.proper_motions = ADf.proper_motions
        self.inv_proper_motion_error_matrix = ADf.inv_proper_motion_error_matrix

        self.basis_ordered = np.array([
            ADf.basis[lmQ] for lmQ in ADf.lmQ_ordered
        ])

        self.which_basis = ADf.which_basis
        self.overlap_matrix_Cholesky = ADf.overlap_matrix_Cholesky

        self.bounds = [[-prior_bounds, prior_bounds] for name in self.names]

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

        R = compute_R(
            self.proper_motions,
            self.inv_proper_motion_error_matrix,
            model,
        )
        R = np.maximum(R, self.tol)

        if self.log_L_method_name == "goodandbad":
            beta = 10 ** self.parameters["log10_beta"]
            gamma = 10 ** self.parameters["log10_gamma"]
            log_likelihood = np.sum(self.logL(R, beta, gamma))
        else:
            log_likelihood = np.sum(self.logL(R))
        
        return log_likelihood
