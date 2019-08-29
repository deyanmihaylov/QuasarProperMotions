import argparse

import AstrometricData as AD

# import csv
# import time
# import datetime
# import os
# import corner
# import pandas

# import numpy as np

# import cpnest
# import cpnest.model

# import matplotlib
# matplotlib.use('Agg')

# from data_load import *
# from injection import *
# from utils import *

# plotting = False
# pm_histogram = False
# VSH_matrix = True
# VSH_corr = False
# benchmarking = False














parser = argparse.ArgumentParser()
parser.add_argument('--Lmax',    help='the maximum VSH index [default 4]', type=int, default=4)
parser.add_argument('--dataset', help='the dataset to use [default 1]', type=int, default=1)
parser.add_argument('--injection', help='the injection mode to use [default 0]', type=int, default=0)
parser.add_argument('--nthreads', help='the number of CPU threads to use [default 2]', type=int, default=2)
parser.add_argument('--nlive', help='the number cpnest live points [default 1024]', type=int, default=1024)
parser.add_argument('--maxmcmc', help='the mcmc length in cpnest [default 256]', type=int, default=256)
parser.add_argument('--llmethod', help='the log likelihood method to use [default permissive]', type=str, default='permissive')
args = parser.parse_args()