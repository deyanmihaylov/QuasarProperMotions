import argparse
import time
import datetime
import os

import corner

import csv

import numpy as np

import cpnest
import cpnest.model

import matplotlib
matplotlib.use('Agg')

from data_load import *
from injection import *
from utils import *
plotting = False
benchmarking = False

parser = argparse.ArgumentParser()
parser.add_argument('--Lmax',    help='the maximum VSH index [default 4]', type=int, default=4)
parser.add_argument('--dataset', help='the dataset to use [default 1]', type=int, default=1)
parser.add_argument('--injection', help='the injection mode to use [default 0]', type=int, default=0)
parser.add_argument('--nthreads', help='the number of CPU threads to use [default 2]', type=int, default=2)
parser.add_argument('--nlive', help='the number cpnest live points [default 1024]', type=int, default=1024)
parser.add_argument('--maxmcmc', help='the mcmc length in cpnest [default 256]', type=int, default=256)
args = parser.parse_args()

log1over2 = np.log(0.5)

tol = 1.0e-5 # This is the minimum residual considered by the log_likelihood

import config as c

c.Lmax = int ( args.Lmax )
dataset = int ( args.dataset )
injection = int ( args.injection )
nthreads = int ( args.nthreads )
nlive = int ( args.nlive )
maxmcmc = int ( args.maxmcmc )
        
# Create a directory for the results

current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

dir_name = "Run_" + str(c.Lmax) + "_" + str(dataset) + "_" + str(injection) + "_" + current_datetime
dir_path = "Results/" + dir_name + "/"

if not os.path.isdir(dir_path):
    os.system('mkdir ' + dir_path)

# Record the run
    
with open("Results/catalogue.txt", "a") as catalogue:
    catalogue.write ( dir_name + '\n' )
    catalogue.write ( "time = " + str ( time.time() ) + '\n' )
    catalogue.write ( "Lmax = " + str ( c.Lmax ) + '\n' )
    catalogue.write ( "dataset = " + str ( dataset ) + '\n' )
    catalogue.write ( "injection = " + str ( injection ) + '\n' )
    catalogue.write ( "nthreads = " + str ( nthreads ) + '\n' )
    catalogue.write ( "nlive = " + str ( nlive ) + '\n' )
    catalogue.write ( "maxmcmc = " + str ( maxmcmc ) + '\n' )
    catalogue.write ( '\n\n\n' )

# Select a dataset to analyze:
#  (1) - type2 (2843 stars)
#  (2) - type3 (489163 stars)
#  (3) - type2+type3 (492006 stars)

if dataset == 1:
    data = import_Gaia_data ( "../data/type2.csv" )
elif dataset == 2:
    data = import_Gaia_data ( "../data/type3.csv" )
elif dataset == 3:
    data = import_Gaia_data ( "../data/type2and3.csv" )
else:
    raise ValueError ( 'Unknown dataset ' + str ( dataset ) )

# PM data to analyze:
#  (0) - no injection (use PM from data)
#  (2) - mock dipole
#  (3) - mock GR pattern

if injection == 0:
    pass
elif injection == 1:
    data , injected_par = generate_scalar_bg ( data , err_scale=10 )
elif injection == 2:
    data , injected_par = generate_gr_bg ( data )
else:
    raise ValueError ( 'Unknown injection ' + str ( dataset ) )

if injection == 1 or injection == 2:
    par_file = csv.writer ( open( dir_path + "/injected_par.txt" , "w"))
    
    for key, val in injected_par.items():
        par_file.writerow ( [ key , val ] )

# pm data to analyse:
#  (1) - mock dipole (2843 stars)
#  (2) - mock GW quad patter (2843 stars)
#  (3) - real type2 (2843 stars)
#  (4) - real type3 (489163 stars)
#  (5) - real type2+type3 (492006 stars)
# if dataset==1:
#     data = import_Gaia_data("../data/type2.csv")
#     VSH_bank = generate_VSH_bank (data , Lmax)
#     data = generate_scalar_bg (data , Lmax , VSH_bank, err_scale=10) # Seperate inject from import
# elif dataset==2:
#     data = import_Gaia_data("../data/type2.csv")
#     VSH_bank = generate_VSH_bank (data , Lmax)
#     data = generate_gr_bg (data , Lmax , VSH_bank)
# elif dataset==3:
#     data = import_Gaia_data("../data/type2.csv")
#     VSH_bank = generate_VSH_bank (data , Lmax)
# elif dataset==4:
#     data = import_Gaia_data("../data/type3.csv")
#     VSH_bank = generate_VSH_bank (data , Lmax)
# elif dataset==5:
#     data = import_Gaia_data("../data/type2and3.csv")
#     VSH_bank = generate_VSH_bank (data , Lmax)
# else:
#     raise ValueError('Unknown dataset {}'.format(dataset))

if plotting: 
    data.plot(self, "fig_dataset{}.png".format(dataset), proper_motions=True, proper_motion_scale=1)
    
print ("Analysing dataset " + str(dataset) + " and injection " + str(injection) + " with Lmax = " + str(c.Lmax) + " using nthreads = " + str(nthreads))

class VSHmodel(cpnest.model.Model):
    """
    Vector Spherical Harmonic (VSH) fit to proper motions
    """

    def __init__(self):

        self.prior_bound_aQlm = 1.
        self.names = []
        self.bounds = []
        
        for Q in ['E', 'B']:
            for l in np.arange(1, c.Lmax+1):
                for m in np.arange(0, l+1):
                
                    if m==0:
                        
                        self.names += [ 'Re[a^' + Q + '_' + str(l) + str(m) + ']' ]
                        self.bounds += [[ -self.prior_bound_aQlm , self.prior_bound_aQlm ]]
                        
                    else:
                    
                        self.names += [ 'Re[a^' + Q + '_' + str(l) + str(m) + ']' ]
                        self.bounds += [[ -self.prior_bound_aQlm , self.prior_bound_aQlm ]]
                        self.names += [ 'Im[a^' + Q + '_' + str(l) + str(m) + ']' ]
                        self.bounds += [[ -self.prior_bound_aQlm , self.prior_bound_aQlm ]]
                        
        print(self.names)


    def log_likelihood(self, params):  
        
        model_pm = generate_model(params, data.VSH )
        Rvals = R_values(data.proper_motions, data.covariance_inv , model_pm)
        Rvals = np.maximum(Rvals, tol)
        log_likelihood = np.sum( logLfunc( Rvals ) )
        
        return log_likelihood

# set up model and log-likelihood
model = VSHmodel()


# This can be removed
# if benchmarking:
#     import time
#     par = {}
#     for l in np.arange(1,Lmax+1):
#         for m in np.arange(0, l+1):
#             if m==0:
#                 par['Re_a^E_'+str(l)+'0'] = 0
#                 par['Re_a^B_'+str(l)+'0'] = 0
#             else:
#                 par['Re_a^E_'+str(l)+str(m)] = 0
#                 par['Im_a^E_'+str(l)+str(m)] = 0
#                 par['Re_a^B_'+str(l)+str(m)] = 0
#                 par['Im_a^B_'+str(l)+str(m)] = 0
#     t_start = time.time()
#     ll = model.log_likelihood(par)
#     t_end = time.time()
#     print(t_end-t_start, ll)
#     exit(-1)


# run nested sampling 

nest = cpnest.CPNest ( model ,
                       output=dir_path ,
                       nlive=nlive , 
                       maxmcmc=maxmcmc ,
                       nthreads=nthreads , 
                       resume=True ,
                       verbose=3)
nest.run()

# post processing
nest.get_nested_samples()
nest.get_posterior_samples()

# custom corner plot
if os.path.isfile(dir_path+'injectedC.csv'):
    
    header_file = dir_path+'header.txt'
    with open(header_file,'r') as f:
        names = f.readline().split()[0:-1]
        
    truths = {}
        
    truths_file = dir_path+'injectedC.csv'
    
    with open(truths_file , mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            truths [row[0]] = row[1]
            
    truths_arr = np.array([truths[name] for name in names])
    
    data_to_plot = np.loadtxt(dir_path+'posterior.dat')[:,0:-2]
    
    corner.corner(data_to_plot, truths=truths_arr, labels=names)
    
    plt.savefig(dir_path+'corner_truths.png')
