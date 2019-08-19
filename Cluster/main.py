import argparse
import time
import os

import corner

import csv

import numpy as np

import cpnest
import cpnest.model

from data_load import *
from utils import *
plotting = False
benchmarking = False

parser = argparse.ArgumentParser()
parser.add_argument('--Lmax',    help='the maximum VSH index [default 4]', type=int, default=4)
parser.add_argument('--dataset', help='the dataset to use [default 1]', type=int, default=1)
parser.add_argument('--nthreads', help='the number of CPU threads to use [default 2]', type=int, default=2)
parser.add_argument('--nlive', help='the number cpnest live points [default 1024]', type=int, default=1024)
parser.add_argument('--maxmcmc', help='the mcmc length in cpnest [default 256]', type=int, default=256)
args = parser.parse_args()


log1over2 = np.log(0.5)
tol = 1.0e-5 # This is the minimum residual considered by the log_likelihood
Lmax = int(args.Lmax)


# pm data to analyse:
#  (1) - mock dipole (2843 stars)
#  (2) - mock GW quad patter (2843 stars)
#  (3) - real type2 (2843 stars)
#  (4) - real type3 (489163 stars)
#  (5) - real type2+type3 (492006 stars)
dataset = int(args.dataset)
if dataset==1:
    data = import_Gaia_data("../data/type2.csv")
    VSH_bank = generate_VSH_bank (data , Lmax)
    data = generate_scalar_bg (data , Lmax , VSH_bank, err_scale=10) # Seperate inject from import
elif dataset==2:
    data = import_Gaia_data("../data/type2.csv")
    VSH_bank = generate_VSH_bank (data , Lmax)
    data = generate_gr_bg (data , Lmax , VSH_bank)
elif dataset==3:
    data = import_Gaia_data("../data/type2.csv")
    VSH_bank = generate_VSH_bank (data , Lmax)
elif dataset==4:
    data = import_Gaia_data("../data/type3.csv")
    VSH_bank = generate_VSH_bank (data , Lmax)
elif dataset==5:
    data = import_Gaia_data("../data/type2and3.csv")
    VSH_bank = generate_VSH_bank (data , Lmax)
else:
    raise ValueError('Unknown dataset {}'.format(dataset))

if plotting: 
    data.plot(self, "fig_dataset{}.png".format(dataset), proper_motions=True, proper_motion_scale=1)
    
print("Analysing dataset {0} with Lmax={1} using nthreads={2}".format(dataset, Lmax, args.nthreads))

    

class VSHmodel(cpnest.model.Model):
    """
    Vector Spherical Harmonic (VSH) fit to proper motions
    """

    def __init__(self):

        self.prior_bound_aQlm = 0.06
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
        
        model_pm = generate_model(params, VSH_bank , Lmax)
        Rvals = R_values(data.proper_motions, data.covariance_inv , model_pm)
        Rvals = np.maximum(Rvals, tol)
        log_likelihood = np.sum( logLfunc( Rvals ) )
        
        return log_likelihood
    


# set up model and log-likelihood
model = VSHmodel()


# This can be removed
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
outdir = "CPNestOutput_NarrowPrior/Lmax_"+str(Lmax)+"_dataset_"+str(dataset)+"/"
if not os.path.isdir(outdir): os.system('mkdir '+outdir)

nest = cpnest.CPNest(model, output=outdir, nlive=int(args.nlive), 
                     maxmcmc=int(args.maxmcmc), nthreads=int(args.nthreads), 
                     resume=True, verbose=3, n_periodic_checkpoint=1000)
nest.run()


# post processing
nest.get_nested_samples()
nest.get_posterior_samples()

# custom corner plot
if os.path.isfile(outdir+'injectedC.csv'):
    
    header_file = outdir+'header.txt'
    with open(header_file,'r') as f:
        names = f.readline.split()[0:-1]
        
    truths = {}
        
    truths_file = outdir+'injectedC.csv'
    
    with open(truths_file , mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            truths [row[0]] = row[1]
            
    truths_arr = np.array([truths[name] for name in names])
    
    data_to_plot = np.loadtxt(outdir+'posterior.dat')[:,0:-2]
    
    corner.corner(data_to_plot, truths=truths_arr, labels=names)
    
    plt.savefig(outdir+'corner_truths.png')
