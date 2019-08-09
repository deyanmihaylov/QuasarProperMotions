import numpy as np
import cpnest
from cpnest.model import Model


Lmax = 4


class VSHmodel(Model):
    """
    Vector Spherical Harmonic (VSH) fit to proper motions
    """
    
    def __init__(self):
    
        names = []
        bounds = []
        
        for Q in ['E', 'B']:
            for l in np.arange(1, Lmax+1):
                for m in np.arange(0, l+1):
                
                    if m==0:
                    
                        names += 'Re_a^'+Q+'_'+str(l)+str(m)
                        bounds += [-1,1]
                        
                    else:
                    
                        names += 'Re_a^'+Q+'_'+str(l)+str(m)
                        bounds += [-1,1]
                        names += 'Im_a^'+Q+'_'+str(l)+str(m)
                        bounds += [-1,1]
                        

    def log_likelihood(self, param):
        return 0.0
        

# run nested sampling
model = VSHmodel()
nest = cpnest.CPNest(model, output='output', nlive=2048, maxmcmc=512, nthreads=8, resume=True, verbose=3)
nest.run()

# post processing run
nest.get_nested_samples()
nest.get_posterior_samples()
