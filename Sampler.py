import numpy as np

import cpnest
import cpnest.model

import Model 

import AstrometricData as AD



def R_values(pm, invcovs, model):
    """
    Compute R values from data, model, and the inverse of the covariant matrix
    """
    M = pm - model
    R_values = np.sqrt( np.einsum('...i,...ij,...j->...', M, invcovs, M))
    return R_values

  
  
def logL_permissive(R):
    """
    The permissive log-likelihood (Darling et al. inspired)
    """
    return np.log( 
                    ( 1. - np.exp(-0.5 * (R**2)) ) 
                    / ( 0.5 * (R**2) )
                )
  
  
  
def logL_quadratic(R):
    """
    The normal log-likelihood 
    """
    return -0.5 * (R**2)
  
  
  

class model(cpnest.model.Model):
    """
    Model to fit to the proper motions
    """

    def __init__(self, dataset, prior_bound=1, whichlikelihood="permissive"):
        """
        Initialise the model class
      
        INPUTS
        ------
        prior_bound: float
            the range of coefficients to search over [mas/yr]
        whichlikelihood: str
            which likelihood function to use [either "permissive" or "normal"]
        """
        assert whichlikelihood=="permissive" or whichlikelihood=="normal", "Unrecognised likelihood option"

        self.tol = 1.0e-5

        self.dataset = dataset
        self.whichlikelihood = whichlikelihood
        self.prior_bound = prior_bound
        
        self.names = [ name.replace("Y","a") for name in self.dataset.names]
        
        self.bounds = [[ -self.prior_bound , self.prior_bound ] for i in range(len(self.names))]
                        
        print("Searching over the following parameters:\n", '\n'.join(self.names))


    def log_likelihood(self, params):  
        """
        The log-likelihood function
        """
        model_pm = Model.generate_model(params, self.dataset.basis)

        Rvals = R_values(self.dataset.proper_motions, self.dataset.inv_proper_motion_error_matrix, model_pm)

        Rvals = np.maximum(Rvals, self.tol)

        if self.whichlikelihood == "permissive":
            log_likelihood = np.sum( logL_permissive( Rvals ) )

        else:
            log_likelihood = np.sum( logL_quadratic( Rvals ) )

        return log_likelihood

