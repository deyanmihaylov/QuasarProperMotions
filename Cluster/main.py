import numpy as np
import cpnest
import cpnest.model

from data_load import *


Lmax = 4


# pm data to analyse:
#  (1) - mock dipole (? stars)
#  (2) - mock GW quad patter (? stars)
#  (3) - real type2 (2843 stars)
#  (4) - real type3 (489163 stars)
#  (5) - real type2+type3 (492006 stars)
dataset = 1
if dataset==1:
    pass
if dataset==2:
    pass
if dataset==3:
    data = import_Gaia_data("data/type2.csv")
if dataset==4:
    data = import_Gaia_data("data/type3.csv")
if dataset==5:
    data = import_Gaia_data("data/type2and3.csv")
else:
    raise ValueError('Unknown dataset {}'.format(dataset))

    
    


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
