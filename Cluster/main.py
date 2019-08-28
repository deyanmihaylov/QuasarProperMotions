import argparse
import csv
import time
import datetime
import os
import corner
import pandas

import numpy as np

import cpnest
import cpnest.model

import matplotlib
matplotlib.use('Agg')

from data_load import *
from injection import *
from utils import *

plotting = False
pm_histogram = False
VSH_matrix = True
VSH_corr = False
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

# Create folder Results if it doesn't exist

if not os.path.isdir ( "Results" ):
    os.system ( "mkdir Results" )


# Create catalogue CSV file if it doesn't exist

if not os.path.isfile ( "Results/catalogue.csv" ):
    catalogue_header = [ "directory" , "submitted" , "start_time" , "end_time" , "duration" , "Lmax" , "dataset" , "injection" , "nthreads" , "nlive" , "maxmcmc" ]

    with open ( "Results/catalogue.csv" , 'w' , newline='') as catalogue_csv:
        catalogue_csv_writer = csv.writer ( catalogue_csv )
        catalogue_csv_writer.writerow ( catalogue_header )

    catalogue_csv.close()


# Create a directory for the results

with open ( "Results/catalogue.csv", "r+" ) as catalogue_csv:
    catalogue_csv_reader = csv.reader ( catalogue_csv )

    n_records = len ( list ( catalogue_csv_reader ) )

catalogue_csv.close()

dir_name = "Run_" + str(n_records) + "_" + str(c.Lmax) + str(dataset) + str(injection)
dir_path = "Results/" + dir_name + "/"

if not os.path.isdir(dir_path):
    os.system('mkdir ' + dir_path)


# Record the run

start_time = time.time()
    
with open ( "Results/catalogue.csv", "a" , newline='' ) as catalogue_csv:
    run_record = [ dir_name , datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') , start_time , 0 , 0 , c.Lmax , dataset , injection , nthreads , nlive , maxmcmc ]

    catalogue_csv_writer = csv.writer ( catalogue_csv )
    catalogue_csv_writer.writerow ( run_record )

catalogue_csv.close()


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
    par_file_open = open( dir_path + "/injected_par.txt" , "w" )
    par_file = csv.writer ( par_file_open )
    
    for key, val in injected_par.items():
        par_file.writerow ( [ key , val ] )

    par_file_open.close()

if plotting: 
    data.plot ( self , "fig_dataset{}.png".format(dataset), proper_motions=True, proper_motion_scale=1)

if pm_histogram:
    data.pm_hist ( dir_path + "pm_histogram.png" )

if VSH_matrix:
    data.vsh_matrix_plot ( dir_path + "vsh_matrix.png" )

if VSH_corr:
    data.vsh_corr_plot ( dir_path + "vsh_corr.png" )

exit()

# Analyze the dataset

def R_values ( pm , invcovs , model ):
    # Compute R values from data, model, and the inverse of the covariant matrix

    M = pm - model
    
    R_values = numpy.sqrt ( numpy.einsum ( '...i,...ij,...j->...' , M , invcovs , M ) )
        
    return R_values

def logLfunc(R):
    return numpy.log ( ( 1 - numpy.exp ( -0.5 * ( R ** 2 ) ) ) / ( 0.5 * ( R ** 2 ) ) )
    
print ("Analysing dataset " + str(dataset) + " and injection " + str(injection) + " with Lmax = " + str(c.Lmax) + " using nthreads = " + str(nthreads))

class VSHmodel(cpnest.model.Model):
    """
    Vector Spherical Harmonic (VSH) fit to proper motions
    """

    def __init__(self):

        self.prior_bound_aQlm = 0.1
        self.names = []
        self.bounds = []
        
        for Q in [ 'E' , 'B' ]:
            for l in np.arange(1, c.Lmax+1):
                for m in np.arange(0, l+1):
                    if m == 0:
                        self.names += [ 'Re[a^' + Q + '_' + str(l) + str(m) + ']' ]

                        # if l == 1 and Q == 'E':
                        #     self.bounds += [[ 0.75 , 1.25 ]]
                        # else:
                        #     self.bounds += [[ -self.prior_bound_aQlm , self.prior_bound_aQlm ]]
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
if os.path.isfile ( dir_path + 'injected_par.txt' ):
    header_file = dir_path + 'header.txt'
    with open(header_file,'r') as f:
        names = f.readline().split()[0:-1]
        
    truths = {}
        
    truths_file = dir_path + 'injected_par.txt'
    
    with open(truths_file , mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)

        for row in csv_reader:
            truths [row[0]] = row[1]
            
    truths_arr = np.array ( [ float( truths[name] ) for name in names] )
    
    data_to_plot = np.loadtxt(dir_path+'posterior.dat')[:,0:-2]
    
    corner.corner(data_to_plot, truths=truths_arr, labels=names)
    
    plt.savefig(dir_path+'corner_truths.png')


# Record end time

end_time = time.time()

catalogue = pandas.read_csv ( "Results/catalogue.csv" )

catalogue.loc [ catalogue [ "directory" ] == dir_name , "end_time" ] = end_time
catalogue.loc [ catalogue [ "directory" ] == dir_name , "duration" ] = end_time - start_time

catalogue.to_csv ( "Results/catalogue.csv" , index=False )
