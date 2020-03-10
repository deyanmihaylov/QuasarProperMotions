import numpy as np
import cpnest
import cpnest.model

import AstrometricData as AD
import Model as M

def R_values(data, invcovs, model):
    """
    Compute R values from data, model, and the inverse of the covariant matrix
    """
    M = data - model
    R_values = np.sqrt(np.einsum('...i,...ij,...j->...', M, invcovs, M))
    
    return R_values

def logL_permissive(R):
    """
    The permissive log-likelihood (Darling et al. inspired)
    """
    half_R_squared = 0.5 * (R**2)
    return np.log((1.-np.exp(-half_R_squared)) / half_R_squared)

def logL_quadratic(R):
    """
    The normal log-likelihood 
    """
    return -0.5 * (R**2)


class model(cpnest.model.Model):
    """
    Model to fit to the proper motions
    """

    def __init__(self,
                 ADf: AD.AstrometricDataframe,
                 logL_method="permissive",
                 prior_bounds=1
                ):
        """
        Initialise the model class
      
        INPUTS
        ------
        prior_bound: float
            the range of coefficients to search over [mas/yr]
        logL_method: str
            which likelihood function to use [either "permissive" or "normal"]
        """
        self.tol = 1.0e-5

        self.names = list(ADf.names.values())

        self.proper_motions = ADf.proper_motions
        self.inv_proper_motion_error_matrix = ADf.inv_proper_motion_error_matrix

        self.basis = {ADf.names[key]: ADf.basis[key] for key in ADf.names.keys()}
        self.which_basis = ADf.which_basis
        self.overlap_matrix_Cholesky = ADf.overlap_matrix_Cholesky

        if logL_method == "permissive":
            self.logL = logL_permissive
        elif logL_method == "quadratic":
            self.logL = logL_quadratic
        
        self.bounds = [[-prior_bounds, prior_bounds] for name in self.names]
                       
        # TO DO: 
        # print("Searching over the following parameters:\n", '\n'.join(self.names))

    def log_likelihood(self, almQ):  
        """
        The log-likelihood function
        """
        model = M.generate_model(almQ, self.basis)
        
        R = R_values(self.proper_motions, self.inv_proper_motion_error_matrix, model)

        R = np.maximum(R, self.tol)

        log_likelihood = np.sum(self.logL(R))

        return log_likelihood
