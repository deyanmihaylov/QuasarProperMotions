import argparse

import numpy as np

import os

import cpnest
import cpnest.model

from data_load import *
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument('--Lmax',    help='the maximum VSH index [default 4]', type=int, default=4)
parser.add_argument('--dataset', help='the dataset to use [default 1]', type=int, default=1)
parser.add_argument('--nthreads', help='the number of CPU threads to use [default 2]', type=int, default=2)
parser.add_argument('--nlive', help='the number cpnest live points [default 1024]', type=int, default=1024)
parser.add_argument('--maxmcmc', help='the mcmc length in cpnest [default 256]', type=int, default=256)
args = parser.parse_args()


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


    
print("Analysing dataset {} with Lmax={}".format(Lmax, dataset))



# Change proper motions from mas/yr to rad/s
#data.proper_motions = data.proper_motions * 1.5362818500441604e-16
#data.proper_motions_err = data.proper_motions_err * 1.5362818500441604e-16
        
    

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
        
        par = dict(params)
        for Q in ['E', 'B']:
            for l in np.arange(1, Lmax+1):
                for m in np.arange(-l, 0):
                    aQlm = par['Re_a^'+Q+'_'+str(l)+str(-m)]+(1j)*par['Im_a^'+Q+'_'+str(l)+str(-m)]
                    par['Re_a^'+Q+'_'+str(l)+str(m)] = np.real( ((-1)**(-m)) * np.conj(aQlm) )
                    par['Im_a^'+Q+'_'+str(l)+str(m)] = np.imag( ((-1)**(-m)) * np.conj(aQlm) )
        
        
        vsh_E_coeffs = [ [ 
                            par['Re_a^E_'+str(l)+'0']+0*(1j)   
                            if m==0 else   
                            par['Re_a^E_'+str(l)+str(m)]+(1j)*par['Im_a^E_'+str(l)+str(m)]
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]
        vsh_B_coeffs = [ [ 
                            par['Re_a^B_'+str(l)+'0']+0*(1j)   
                            if m==0 else   
                            par['Re_a^B_'+str(l)+str(m)]+(1j)*par['Im_a^B_'+str(l)+str(m)]
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]
        
        model_pm = generate_model(vsh_E_coeffs, vsh_B_coeffs, data.positions)
        Rvals = R_values(data.proper_motions, data.proper_motions_err, data.proper_motions_err_corr , model_pm)
        condition = Rvals > tol
        modify_Rvals = np.extract(condition, Rvals)
        log_likelihood = np.log( ( 1. - np.exp ( - Rvals ** 2 / 2.) ) / ( Rvals ** 2 ) ).sum()
        
        return log_likelihood
        


# set up model and log-likelihood
model = VSHmodel()


# run nested sampling 
outdir = "CPNestOutput/Lmax_"+str(Lmax)+"_dataset_"+str(dataset)+"/"
if not os.path.isdir(outdir): os.system('mkdir '+outdir)

nest = cpnest.CPNest(model, output=outdir, nlive=int(args.nlive), 
                     maxmcmc=int(args.maxmcmc), nthreads=int(args.nthreads), resume=True, verbose=3)
nest.run()


# post processing
nest.get_nested_samples()
nest.get_posterior_samples()
