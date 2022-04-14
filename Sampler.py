import sys
import numpy as np

from scipy.stats import norm

import cpnest
import cpnest.model

from scipy.special import erf, logsumexp

import AstrometricData as AD
import Model as M

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
    R_values = np.sqrt(np.einsum("...i,...ij,...j->...", M, invcovs, M))

    return R_values

def logL_quadratic(
    R: np.array,
) -> np.array:
    """
    The normal log-likelihood
    """
    return -0.5 * (R**2)

def logL_permissive(
    R: np.array,
) -> np.array:
    """
    The permissive log-likelihood
    As used in Darling et al. 2018 and coming from Sivia and Skilling,
    p.168
    """
    half_R_squared = 0.5 * (R**2)
    return np.log((1.-np.exp(-half_R_squared)) / half_R_squared)

def logL_2Dpermissive(
    R: np.array,
) -> np.array:
    """
    The modified permissive log-likelihood for 2D data
    A generalisation of the Sivia and Skilling likelihood (p.168) for
    2D data
    """
    return (
        np.log(
            (np.sqrt(np.pi/2) * erf(R/np.sqrt(2))
            - R * np.exp(-R**2/2)) / (R**3)
        )
    )

def logL_goodandbad(
    R: np.array,
    beta: float,
    gamma: float,
) -> np.array:
    """
    Following the notation of Sivia and Skilling, this is "the good
    and bad data model".

    Some fraction beta of the data is assumed to come from a
    normal distribution with errors larger by a factor of gamma.
    """

    # enforce conditions 0 < beta < 1 and 1 < gamma
    my_beta = np.clip(beta, 0, 1)
    my_gamma = np.clip(gamma, 1, 10)
        
    return logsumexp(
        [
            -0.5*(R/my_gamma)**2+np.log(my_beta/my_gamma**2),
            -0.5*R**2+np.log(1-my_beta)
        ],
        axis=0,
    )


class model(cpnest.model.Model):
    """
    Model to fit to the proper motions
    """

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

        self.tol = 1.0e-5

        self.names = list(ADf.almQ_names.values())

        logL_method = params['logL_method']
        prior_bounds = params['prior_bounds']

        if logL_method == "permissive":
            self.logL = logL_permissive
        elif logL_method == "2Dpermissive":
            self.logL = logL_2Dpermissive
        elif logL_method == "quadratic":
            self.logL = logL_quadratic
        elif logL_method == "goodandbad":
            self.logL = logL_goodandbad
        else:
            print("Oh dear. This doesn't look good.")
            sys.exit()
        
        self.logL_method = logL_method

        self.proper_motions = ADf.proper_motions
        self.inv_proper_motion_error_matrix = ADf.inv_proper_motion_error_matrix

        self.basis = {ADf.almQ_names[key]: ADf.basis[key] for key in ADf.basis.keys()}
        self.which_basis = ADf.which_basis
        self.overlap_matrix_Cholesky = ADf.overlap_matrix_Cholesky

        self.bounds = [[-prior_bounds, prior_bounds] for name in self.names]

        if logL_method == "goodandbad":
            self.names.extend(['log10_beta', 'log10_gamma'])
            self.bounds.extend([[-1.78, -1.20], [-0.08, 0.52]])

            self.beta_prior = norm(np.log10(0.03165), 0.05)
            self.gamma_prior = norm(np.log10(1.6596), 0.05)

        print("Searching over the following parameters:", ', '.join(self.names))

    def log_prior(
        self,
        params: dict
    ) -> float:
        """
        The log-prior function
        """

        if self.logL_method is "goodandbad":
            log_prior = self.beta_prior.logpdf(params['log10_beta'])
            log_prior += self.gamma_prior.logpdf(params['log10_gamma'])
        else:
            log_prior = 0.

        return log_prior

    def log_likelihood(
        self,
        almQ: dict,
    ) -> float:
        """
        The log-likelihood function
        """

        model = M.generate_model(almQ, self.basis)

        R = R_values(self.proper_motions, self.inv_proper_motion_error_matrix, model)

        R = np.maximum(R, self.tol)

        if self.logL_method == "goodandbad":
            log_likelihood = np.sum(self.logL(R, 10**almQ['beta'], 10**almQ['gamma']))
        else:
            log_likelihood = np.sum(self.logL(R))

        return log_likelihood
