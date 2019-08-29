import pandas
import numpy
import csv

import CoordinateTransformations as CT
import VectorSphericalHarmonics as VSH

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AstrometricDataframe:
    def __init__(self):

        self.n_objects = 0
        
        self.positions = numpy.array ([])
        self.positions_Cartesian = numpy.array ([])

        self.proper_motions = numpy.array ([])

        self.inv_proper_motion_error_matrix = numpy.array ([])


        def load_Gaia_data(self):
            """
            Load the postions, proper motions and proper motion errors from file
            """
            pass

        def gen_mock_data(self):
            """
            Simulate the postions, proper motions and proper motion errors
            """
            pass

        def generate_VSH_bank(self):
            """
            Precompute VSH functions at QSO locations 
            """
            pass





        def plot(self, outfile, proper_motions=False, projection='mollweide', proper_motion_scale=1):
            """
            method to plot positions (and optionally pms) of QSOs in dataframe
            """
            pass

        def pm_hist(self, outfile):
            """
            Plot a histogram of the proper motions of the quasars 
            """
            pass

        
                
