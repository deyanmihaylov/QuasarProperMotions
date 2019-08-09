import numpy as np
import cpnest
import cpnest.model


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
        


# set up model and log-likelihood
model = VSHmodel()


# run nested sampling 
outdir = "CPNestOutput/Lmax_"+str(Lmax)+"_test/"
if not os.path.isdir(outdir): os.system('mkdir '+outdir)

nest = cpnest.CPNest(model, output=outdir, nlive=4096, maxmcmc=1024, nthreads=16, resume=True, verbose=3)
nest.run()


# post processing
nest.get_nested_samples()
nest.get_posterior_samples()
