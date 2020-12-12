import numpy as np
import cpnest
import cpnest.model

import AstrometricData as AD
import Model as M

def R_values(
        data,
        invcovs,
        model
    ):
    """
    Compute R values from data, model, and the inverse of the 
    covariant matrix
    """
    M = data - model
    R_values = np.sqrt(np.einsum("...i,...ij,...j->...", M, invcovs, M))

    return R_values

def logL_quadratic(R):
    """
    The normal log-likelihood
    """
    return -0.5 * (R**2)

def logL_permissive(R):
    """
    The permissive log-likelihood 
    As used in Darling et al. 2018 and coming from Sivia and Skilling,
    p.168
    """
    half_R_squared = 0.5 * (R**2)
    return np.log((1.-np.exp(-half_R_squared)) / half_R_squared)

from scipy.special import erf
def logL_2Dpermissive(R):
    """
    The modified permissive log-likelihood for 2D data
    A generalisation of the Sivia and Skilling likelihood (p.168) for 
    2D data
    """
    return np.log( (np.sqrt(np.pi/2)*erf(R/np.sqrt(2)) - R*np.exp(-R**2/2)) / (R**3) )

from scipy.special import logsumexp
def logL_goodandbad(R, beta, gamma):
    """
    Following the notation of Sivia and Skilling, this is "the good 
    and bad data model".
    
    Some fraction beta of the data is assumed to come from a
    normal distribution with errors larger by a factor of gamma.
    """

    # enforce conditions 0<beta<1 and 1<gamma
    my_beta = np.clip(beta,0,1)
    my_gamma = np.clip(gamma,1,10)
        
    return logsumexp([ -0.5*(R/my_gamma)**2+np.log(my_beta/my_gamma**2) , -0.5*R**2+np.log(1-my_beta) ], axis=0)


class model(cpnest.model.Model):
    """
    Model to fit to the proper motions
    """

    def __init__(
            self,
            ADf: AD.AstrometricDataframe,
            logL_method: str,
            prior_bounds: float,
            beta = 0.01,
            gamma = 2.0,
        ) -> None:
        """
        Initialise the model class

        INPUTS
        ------
        prior_bound: float
            the range of coefficients to search over [mas/yr]
        logL_method: str
            which likelihood function to use [either "permissive", "quadratic", "2Dpermissive", "goodandbad"]
        beta, gamma: float
            If using logL_method="goodandbad" then must provide beta in range (0,1) and gamma>1.
        """
        self.tol = 1.0e-5

        self.names = list(ADf.names.values())

        self.proper_motions = ADf.proper_motions
        self.inv_proper_motion_error_matrix = ADf.inv_proper_motion_error_matrix

        self.basis = {ADf.names[key]: ADf.basis[key] for key in ADf.names.keys()}
        self.which_basis = ADf.which_basis
        self.overlap_matrix_Cholesky = ADf.overlap_matrix_Cholesky

        self.logL_method = logL_method

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

        self.bounds = [[-prior_bounds, prior_bounds] for name in self.names]

        self.beta = beta
        self.gamma = gamma

        # TO DO:
        # print("Searching over the following parameters:\n", '\n'.join(self.names))

    def log_likelihood(
            self,
            almQ: dict
        ) -> float:
        """
        The log-likelihood function
        """

        model = M.generate_model(almQ, self.basis)

        R = R_values(self.proper_motions, self.inv_proper_motion_error_matrix, model)

        R = np.maximum(R, self.tol)

        if self.logL_method == "goodandbad":
            log_likelihood = np.sum(self.logL(R, self.beta, self.gamma))
        else:
            log_likelihood = np.sum(self.logL(R))

        return log_likelihood
