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

        def gen_mock_data(self, NumObjects, eps=0.2, noise=0.1, signal=0.5):
            """
            Simulate the postions, proper motions and proper motion errors
            
            INPUTS
            ------
            NumObjects: int
                Desired number of objects in mock catalog
            eps: float
                The QSO positions will be drawn from a distribution which favours the equator over the poles.
                eps controls this distribution; large eps (e.g. 100) is uniform, small eps (e.g. 0.1) is very non-uniform
            noise: float
                The size of the proper motion error [mas/yr]
            signal: float
                Size of the dipole (a^E_10) to be injected [mas/yr]
            """
            
            # Positions
            theta = np.arccos(truncnorm.rvs(-1./eps, 1./eps, 0., eps, size=data.n_objects))
            phi = np.random.uniform(0, 2*np.pi, size=data.n_objects)
            ra, dec = phi, 0.5*np.pi-theta
            self.positions = np.array(list(zip(ra, dec)))
            self.positions_Cartesian = CT.geographic_to_Cartesian_point(self.positions)
            
            # Proper Motion Errors
            errors = np.zeros((self.n_objects, 2))
            errors[:,0] = noise * np.reciprocal(np.cos(self.positions[:,1])) # RA error
            errors[:,1] = noise * np.ones(len(self.proper_motions_err))      # DEC error
            cov = np.einsum('...i,...j->...ij', errors, errors)              # Diagonal cov matrix
            self.inv_proper_motion_error_matrix = np.linalg.inv(cov)
            
            # Proper Motions - noise component
            pmra_noise = np.array([ np.random.normal(0, noise/np.cos(self.positions[i][1])) for i in range(self.n_objects)])
            pmdec_noise = np.random.normal(0, noise, size=self.n_objects)
            self.proper_motions += np.array(list(zip(pmra_noise, pmdec_noise)))
            
            # Proper Motions - signal component
            par = {}
            Lmax_temp = 1
            for l in range(1, Lmax_temp+1):
                for m in range(0, l+1):
                    if m==0:
                        par[ 'Re[a^E_' + str(l) +'0]' ] = 0.
                        par[ 'Re[a^B_' + str(l) +'0]' ] = 0.
                else:
                        par[ 'Re[a^E_' + str(l) + str(m) + ']' ] = 0.
                        par[ 'Im[a^E_' + str(l) + str(m) + ']' ] = 0.
                        par[ 'Re[a^B_' + str(l) + str(m) + ']' ] = 0.
                        par[ 'Im[a^B_' + str(l) + str(m) + ']' ] = 0.
            par[ 'Re[a^E_10]' ] = dipole
            self.proper_motions += Model(par, self.VSH_bank, Lmax)
    
        def generate_VSH_bank(self):
            """
            Precompute VSH functions at QSO locations 
            """
            





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
            
