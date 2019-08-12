import argparse

import numpy as np

import os

import cpnest
import cpnest.model

from data_load import *
from utils import *
benchmarking = False


parser = argparse.ArgumentParser()
parser.add_argument('--Lmax',    help='the maximum VSH index [default 4]', type=int, default=4)
parser.add_argument('--dataset', help='the dataset to use [default 1]', type=int, default=1)
parser.add_argument('--nthreads', help='the number of CPU threads to use [default 2]', type=int, default=2)
parser.add_argument('--nlive', help='the number cpnest live points [default 1024]', type=int, default=1024)
parser.add_argument('--maxmcmc', help='the mcmc length in cpnest [default 256]', type=int, default=256)
args = parser.parse_args()


log1over2 = np.log(0.5)
tol = 1.0e-3 # This is the minimum residual considered by the log_likelihood
Lmax = int(args.Lmax)


# pm data to analyse:
#  (1) - mock dipole (? stars)
#  (2) - mock GW quad patter (? stars)
#  (3) - real type2 (2843 stars)
#  (4) - real type3 (489163 stars)
#  (5) - real type2+type3 (492006 stars)
dataset = int(args.dataset)
if dataset==1:
    data = import_Gaia_data("../data/type2.csv")
    data.positions = deg_to_rad(data.positions)
    data = generate_scalar_bg (data)
elif dataset==2:
    data = import_Gaia_data("../data/type2.csv")
    data.positions = deg_to_rad(data.positions)
    data = generate_gr_bg (data)
elif dataset==3:
    data = import_Gaia_data("../data/type2.csv")
    data.positions = deg_to_rad(data.positions)
elif dataset==4:
    data = import_Gaia_data("../data/type3.csv")
    data.positions = deg_to_rad(data.positions)
elif dataset==5:
    data = import_Gaia_data("../data/type2and3.csv")
    data.positions = deg_to_rad(data.positions)
else:
    raise ValueError('Unknown dataset {}'.format(dataset))


    
print("Analysing dataset {0} with Lmax={1}".format(dataset, Lmax))



# Change proper motions from mas/yr to rad/s
#data.proper_motions = data.proper_motions * 1.5362818500441604e-16
#data.proper_motions_err = data.proper_motions_err * 1.5362818500441604e-16
        
    
    
    
    
if Lmax==1:
    from MappingTo_aQlm import CoefficientsFromParams_Lmax1 as mapping
else:
    from MappingTo_aQlm import CoefficientsFromParams_General as mapping
    
    
    

class VSHmodel(cpnest.model.Model):
    """
    Vector Spherical Harmonic (VSH) fit to proper motions
    """

    def __init__(self):

        self.prior_bound_aQlm = 2
        self.names = []
        self.bounds = []
        
        for Q in ['E', 'B']:
            for l in np.arange(1, Lmax+1):
                for m in np.arange(0, l+1):
                
                    if m==0:
                        
                        self.names += ['Re_a^'+Q+'_'+str(l)+str(m)]
                        self.bounds += [[-self.prior_bound_aQlm, self.prior_bound_aQlm]]
                        
                    else:
                    
                        self.names += ['Re_a^'+Q+'_'+str(l)+str(m)]
                        self.bounds += [[-self.prior_bound_aQlm, self.prior_bound_aQlm]]
                        self.names += ['Im_a^'+Q+'_'+str(l)+str(m)]
                        self.bounds += [[-self.prior_bound_aQlm, self.prior_bound_aQlm]]
        print(self.names)

    def log_likelihood(self, params):
        
        vsh_E_coeffs, vsh_B_coeffs = mapping(params)        
        model_pm = generate_model(vsh_E_coeffs, vsh_B_coeffs, data.positions)
        Rvals = R_values(data.proper_motions, data.proper_motions_err, data.proper_motions_err_corr , model_pm)
        Rvals = np.maximum(Rvals, tol)
        log_likelihood = np.sum( logLfunc( Rvals ) )
        
        return log_likelihood
    
    
        # Desired code - precomputed cov and invcov matrices
        # Rvals = R_values(data.proper_motions, data.inv_covs, model_pm)
        


# set up model and log-likelihood
model = VSHmodel()





if benchmarking:
    import time
    par = {}
    for l in np.arange(1,Lmax+1):
        for m in np.arange(0, l+1):
            if m==0:
                par['Re_a^E_'+str(l)+'0'] = 0
                par['Re_a^B_'+str(l)+'0'] = 0
            else:
                par['Re_a^E_'+str(l)+str(m)] = 0
                par['Im_a^E_'+str(l)+str(m)] = 0
                par['Re_a^B_'+str(l)+str(m)] = 0
                par['Im_a^B_'+str(l)+str(m)] = 0
    t_start = time.time()
    ll = model.log_likelihood(par)
    t_end = time.time()
    print(t_end-t_start, ll)
    exit(-1)




# run nested sampling 
outdir = "CPNestOutput/Lmax_"+str(Lmax)+"_dataset_"+str(dataset)+"/"
if not os.path.isdir(outdir): os.system('mkdir '+outdir)

nest = cpnest.CPNest(model, output=outdir, nlive=int(args.nlive), 
                     maxmcmc=int(args.maxmcmc), nthreads=int(args.nthreads), resume=True, verbose=3)
nest.run()


# post processing
nest.get_nested_samples()
nest.get_posterior_samples()
