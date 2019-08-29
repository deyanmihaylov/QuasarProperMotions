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
        self.positions = numpy.array ([])
                self.positions_Cartesian = numpy.array ([])

                self.n_objects = 0

                self.proper_motions = numpy.array ([])

                self.covariance_inv = numpy.array ([])
                
                
