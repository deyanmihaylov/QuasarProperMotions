import argparse

import AstrometricData as AD
import Sampler
from PostProcessing import post_process_results

import csv
import time
import datetime
import os
import pandas as pd

import numpy as np

import cpnest


parser = argparse.ArgumentParser()
parser.add_argument('--Lmax',    help='the maximum VSH index [default 4]', type=int, default=4)

parser.add_argument('--dataset', help="Select a dataset to analyze:\n(1) - mock dataset\n(2) - type2 (2843 stars)\n(3) - type3 (489163 stars)\n(4) - type2+3 (492006 stars) [default 2]", type=int, default=2)

parser.add_argument('--injection', help="PM data to inject:\n(0) - no injection (use PM from data)\n(2) - mock dipole\n(3) - mock GR pattern (not implemented yet [default 0]", type=int, default=0)

parser.add_argument('--nthreads', help='The number of CPU threads to use [default 2]', type=int, default=2)
parser.add_argument('--nlive', help='The number cpnest live points [default 1024]', type=int, default=1024)
parser.add_argument('--maxmcmc', help='The mcmc length in cpnest [default 128]', type=int, default=128)
parser.add_argument('--llmethod', help='The log likelihood method to use [default permissive]', type=str, default="permissive")
parser.add_argument('--plotting', help="Plot data", type=bool, default=True)
parser.add_argument('--mod_basis', help="Use modified basis", type=bool, default=False)
args = parser.parse_args()


# Create a directory for the results
n_records = len(open("Results/catalogue.csv").readlines())
dir_name = "Run_" + str(n_records) + "_" + str(args.Lmax) + str(args.dataset) + str(args.injection)
dir_path = "Results/" + dir_name + "/"
if not os.path.isdir(dir_path):
    os.system('mkdir ' + dir_path)



# Record the run
start_time = time.time()    
with open ( "Results/catalogue.csv", "a" , newline='' ) as catalogue_csv:
    run_record = [dir_name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), start_time, 0, 0, args.Lmax, args.dataset, args.injection, args.nthreads, args.nlive, args.maxmcmc, args.mod_basis, args.llmethod]
    catalogue_csv_writer = csv.writer ( catalogue_csv )
    catalogue_csv_writer.writerow ( run_record )
catalogue_csv.close()




# Load/Setup the astrometric data
data = AD.AstrometricDataframe()

if args.dataset == 1:
    data.gen_mock_data(500, eps=0.25, noise=0.1)
elif args.dataset == 2:
    data.load_Gaia_data("data/type2.csv")
elif args.dataset == 3:
    data.load_Gaia_data( "data/type3.csv" )
elif args.dataset == 4:
    data.load_Gaia_data( "data/type2and3.csv" )
else:
    raise ValueError("Unknown dataset " + str(args.dataset))

if args.injection == 0:
    pass
elif args.injection == 1:
    data.inject_proper_motions(dipole=0.1, dir_path=dir_path)
elif args.injection == 2:
    ValueError("Not implemented yet.")
else:
    raise ValueError("Unknown injection " + str(dataset))

if args.plotting:
    data.plot(dir_path + "/qso_positions.png")
    data.pm_hist(dir_path + "/qso_pm_hist.png")
    data.plot_overlap_matrix(dir_path + "/qso_vsh_overlaps.png")

if args.mod_basis:
    data.change_basis()
    if args.plotting:
        data.plot_overlap_matrix(dir_path + "/qso_mod_overlaps.png")




# Nested sampling
mymodel = Sampler.model(data, whichlikelihood=args.llmethod)
nest = cpnest.CPNest ( mymodel ,
                       output=dir_path ,
                       nlive=args.nlive , 
                       maxmcmc=args.maxmcmc ,
                       nthreads=args.nthreads , 
                       resume=True ,
                       verbose=3)
nest.run()
nest.get_nested_samples()
nest.get_posterior_samples()

# post_process_results()



# Record end time
end_time = time.time()
catalogue = pd.read_csv ( "Results/catalogue.csv" )
catalogue.loc[ catalogue [ "directory" ] == dir_name, "end_time" ] = end_time
catalogue.loc[ catalogue [ "directory" ] == dir_name, "duration" ] = end_time - start_time
catalogue.to_csv("Results/catalogue.csv", index=False)
