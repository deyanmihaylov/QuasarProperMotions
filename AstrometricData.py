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
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=projection)

            ra = numpy.array([ x-2*numpy.pi if x>numpy.pi else x for x in self.positions[:,0]])
            dec = self.positions[:,1]

            # plot the positions                                                                                                      
            ax.plot(ra, dec, 'o', color='r', markersize=1, alpha=0.8)

            # plot the proper motions                                                                                                 
            if proper_motions:
                Nstars = len(self.positions)
                for i in range(Nstars):
                    Ra = [ ra[i] - proper_motion_scale*self.proper_motions[i,0], ra[i] + proper_motion_scale*self.proper_motions[i,0] ]
                    Dec = [ dec[i] - proper_motion_scale*self.proper_motions[i,1], dec[i] + proper_motion_scale*self.proper_motions[i,1] ]
                    ax.plot(Ra, Dec, '-', color='r', alpha=0.6)

            # plot grid lines                                                                                                         
            plt.grid(True)
            
            plt.savefig(outfile)
            

        def pm_hist(self, outfile):
            """
            Plot a histogram of the proper motions of the quasars 
            """
            proper_motions_Cartesian = numpy.linalg.norm(geographic_to_Cartesian_vector(self.positions, self.proper_motions), axis = 1)
		plt.hist(proper_motions_Cartesian)
            
            plt.xlabel('Proper motion [mas/yr]')
            plt.ylabel('Number of quasars')
            plt.title('Histogram of quasar proper motions')
            plt.yscale('log')
            plt.savefig(outfile)
            
