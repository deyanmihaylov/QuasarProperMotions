import numpy as np

import cpnest
import cpnest.model

import Model 




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
    return np.log((1. - np.exp(-0.5 * (R**2))) / (0.5 * (R**2)))
  
  
  
def logL_quadratic(R):
    """
    The normal log-likelihood 
    """
    return np.sum(-0.5 * R**2)
  
  
  
class model(cpnest.model.Model):
    """
    model to fit to the proper motions
    """

    def __init__(self, dataset, prior_bound=1, whichlikelihood="permissive"):
      """
      initialise the model class
      
      INPUTS
      ------
      prior_bound: float
        the range of coefficients to search over [mas/yr]
      whichlikelihood: str
        which likelihood function to use [either "permissive" or "normal"]
      """
      
      assert whichlikelihood=="permissive" or whichlikelihood=="normal", "Unrecognised likelihood option"
      
        self.dataset = dataset
        self.whichlikelihood = whichlikelihood
        self.prior_bound_aQlm = prior_bound
        self.names = []
        self.bounds = []
        
        self.names += self.dataset.names
        
        self.bounds += [[ -self.prior_bound , self.prior_bound ] for i in range(len(self.names))]
                        
        print("Searching over the following parameters:", self.names)


    def log_likelihood(self, params):  
        
        model_pm = Model.generate_model(params, self.data.basis )
        Rvals = R_values(self.data.proper_motions, self.data.covariance_inv , model_pm)
        Rvals = np.maximum(Rvals, tol)
        if self.whichlikelihood == "permissive":
          log_likelihood = np.sum( logL_permissive( Rvals ) )
        else:
           log_likelihood = np.sum( logL_quadratic( Rvals ) )
        
        return log_likelihood
