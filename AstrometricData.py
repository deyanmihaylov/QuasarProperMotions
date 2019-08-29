import pandas as pd
import numpy as np
import csv

import CoordinateTransformations as CT
import VectorSphericalHarmonics as VSH
from Model import generate_model

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy.stats import truncnorm
from scipy.linalg import cholesky



def covariant_matrix(errors, corr):
    """
    Function for computing the covariant matrix from errors and correlations  
    """
    covariant_matrix = np.einsum('...i,...j->...ij', errors, errors )
    covariant_matrix[...,0,1] = covariant_matrix[...,1,0] = np.multiply(covariant_matrix[...,1,0], corr.flatten())
    return covariant_matrix



class AstrometricDataframe:
    def __init__(self, Lmax=2):
        """
	Initialise Class: use VSH harmonics up to and including Lmax
        """
        self.n_objects = 0
        self.Lmax = Lmax
	
        assert Lmax<10, "WARNING, the way the names are currently coded will break for double digit l"
        
        self.positions = np.array ([])
        self.positions_Cartesian = np.array ([])

        self.proper_motions = np.array ([])

        self.inv_proper_motion_error_matrix = np.array ([])

        self.VSH_bank = dict()


    def deg_to_rad(self, degree_vals):
        """
        Does what it says on the tin
        """
        return np.deg2rad(degree_vals)


    def load_Gaia_data(self , path):
        """
        Load the postions, proper motions and proper motion errors from file
        """
                
        dataset = pd.read_csv(path,
            	                      sep=',',
                                      delimiter=None,
                                      header='infer',
                                      names=None,
                                      index_col=None,
                                      usecols=None,
                                      squeeze=False,
                                      prefix=None,
                                      mangle_dupe_cols=True,
                                      dtype=None,
                                      engine='python',
                                      converters=None,
                                      true_values=None,
                                      false_values=None,
                                      skipinitialspace=False,
                                      skiprows=None,
                                      skipfooter=0,
                                      nrows=None,
                                      na_values=None,
                                      keep_default_na=True,
                                      na_filter=True,
                                      verbose=False,
                                      skip_blank_lines=True,
                                      parse_dates=False,
                                      infer_datetime_format=False,
                                      keep_date_col=False,
                                      date_parser=None,
                                      dayfirst=False,
                                      iterator=False,
                                      chunksize=None,
                                      compression=None,
                                      thousands=None,
                                      decimal=b'.',
                                      lineterminator=None,
                                      quotechar='"',
                                      quoting=0,
                                      doublequote=True,
                                      escapechar=None,
                                      comment=None,
                                      encoding=None,
                                      dialect=None,
                                      error_bad_lines=True,
                                      warn_bad_lines=True,
                                      delim_whitespace=False,
                                      low_memory=True,
                                      memory_map=False,
                                      float_precision=None)
	
        dropna_columns = ['ra',
			      'dec',
			      'pmra',
			      'pmdec',
			      'pmra_error',
			      'pmdec_error',
			      'pmra_pmdec_corr']

        dataset.dropna ( axis=0,
                             how='any',
                             thresh=None,
                             subset=dropna_columns,
                             inplace=True)
                                
        self.positions = dataset[[ 'ra' , 'dec' ]].values
        self.positions = self.deg_to_rad ( self.positions )

        self.positions_Cartesian = CT.geographic_to_Cartesian_point ( self.positions )

        self.n_objects = dataset.shape[0]
                                
        self.proper_motions = dataset[[ 'pmra' , 'pmdec' ]].values
                
        proper_motions_err = dataset[[ 'pmra_error' , 'pmdec_error' ]].values
        proper_motions_err[:,0] = proper_motions_err[:,0] / np.cos ( self.positions[:,1] )

        proper_motions_err_corr = dataset[[ 'pmra_pmdec_corr' ]].values
                
        covariance = covariant_matrix ( proper_motions_err , proper_motions_err_corr )
                
        self.inv_proper_motion_error_matrix = np.linalg.inv ( covariance )
                
        self.VSH_bank = self.generate_VSH_bank()
	
	
    def non_uniform_random_positions(self, eps=0.2):
        """
	Generate random positions from a distorted distribution
	
	INPUTS
	------
	eps: float
            controls this distribution; large eps (e.g. 100) is uniform, small eps (e.g. 0.1) is very non-uniform
	"""
        theta = np.arccos(truncnorm.rvs(-1./eps, 1./eps, 0., eps, size=self.n_objects))
        phi = np.random.uniform(0, 2*np.pi, size=self.n_objects)
        ra, dec = phi, 0.5*np.pi-theta
        self.positions = np.array(list(zip(ra, dec)))
        self.positions_Cartesian = CT.geographic_to_Cartesian_point(self.positions)


    def gen_mock_data(self, NumObjects, eps=0.2, noise=0.1, dipole=0.5):
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
        dipole: float
                Size of the dipole (a^E_10) to be injected [mas/yr]
        """
            
        self.n_objects = NumObjects

        # Positions
        self.non_uniform_random_positions(eps=eps)
	
        # Compute the VSH bank
        self.VSH_bank = self.generate_VSH_bank()
            
        # Proper Motion Errors
        errors = np.zeros((self.n_objects, 2))
        errors[:,0] = noise * np.reciprocal(np.cos(self.positions[:,1])) # RA error
        errors[:,1] = noise * np.ones(self.n_objects)                    # DEC error
        corr = np.zeros(self.n_objects)
        cov = covariant_matrix(errors, corr)
        self.inv_proper_motion_error_matrix = np.linalg.inv(cov)
            
        # Proper Motions - noise component
        self.proper_motions = np.zeros((self.n_objects, 2))
        pmra_noise = np.array([ np.random.normal(0, noise/np.cos(self.positions[i][1])) for i in range(self.n_objects)])
        pmdec_noise = np.random.normal(0, noise, size=self.n_objects)
        self.proper_motions += np.array(list(zip(pmra_noise, pmdec_noise)))
            
        # Proper Motions - signal component
        par = {}
        for l in range(1, self.Lmax+1):
            for m in range(-l, l+1):
                par[ 'a^E_' + str(l) + ',' + str(m) ] = 0.
                par[ 'a^B_' + str(l) + ',' + str(m) ] = 0.
        par[ 'a^E_1,0' ] = dipole
        self.proper_motions += generate_model(par, self.VSH_bank)
    

    def generate_VSH_bank(self):
        """
        Precompute VSH functions at QSO locations 
        """

        VSH_bank = {}

        for l in range(1, self.Lmax + 1):
            for m in range(-l, l+1):
                VSH_bank['Y^E_' + str(l) + ',' + str(m)] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, np.real(VSH.RealVectorSphericalHarmonicE (l, m, self.positions_Cartesian)))
                VSH_bank['Y^B_' + str(l) + ',' + str(m)] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, np.real(VSH.RealVectorSphericalHarmonicB (l, m, self.positions_Cartesian)))
                        
        return VSH_bank


    def compute_overlap_matrix(self):
        """
        Calculate the overlap matrix between VSH
        """
        self.names = []

        for l in range(1, self.Lmax+1):
            for m in range(-l, l+1):
                for Q in ["E", "B"]:
                    self.names.append("Y^"+Q+"_"+str(l)+","+str(m))

        self.overlap_matrix = np.zeros((len(self.names), len(self.names)))

        prefactor = 4 * np.pi / self.n_objects

        for i, name_x in enumerate(self.names):
            Q_x = name_x[2]
            l_x = int(name_x[4])
            m_x = int(name_x.split(',')[1])
		    
            for j, name_y in enumerate(self.names):
                Q_y = name_y[2]
                l_y = int(name_y[4])
                m_y = int(name_y.split(',')[1])
		        
                X = VSH.RealVectorSphericalHarmonicE(l_x, m_x, self.positions_Cartesian) if Q_x=='E' else VSH.RealVectorSphericalHarmonicB(l_x, m_x, self.positions_Cartesian)
		        
                Y = VSH.RealVectorSphericalHarmonicE(l_y, m_y, self.positions_Cartesian) if Q_y=='E' else VSH.RealVectorSphericalHarmonicB(l_y, m_y, self.positions_Cartesian)
		 
                assert np.max(abs(np.imag(X))) == 0
                assert np.max(abs(np.imag(Y))) == 0
       
                self.overlap_matrix[i,j] = prefactor * np.einsum ( "...j,...j->..." , X , Y ).sum()

    def plot_overlap_matrix(self, Matrix):
        """
        Plot an overlap matrix
        """
        plt.imshow(Matrix)

        plt.xticks(np.arange(len(self.names)), self.names, rotation=90)
        plt.yticks(np.arange(len(self.names)), self.names)

        plt.colorbar()

        plt.tight_layout()
        plt.savefig(output)
        plt.clf()
        

    def plot(self, outfile, proper_motions=False, projection='mollweide', proper_motion_scale=1):
        """
        method to plot positions (and optionally pms) of QSOs in dataframe
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection=projection)

        ra = np.array([ x-2*np.pi if x>np.pi else x for x in self.positions[:,0]])
        dec = self.positions[:,1]

        # plot the positions
        ax.plot(ra, dec, 'o', color='r', markersize=1, alpha=0.8)

        # plot the proper motions 
        if proper_motions:
            Nstars = len(self.positions)
            for i in range(Nstars):
                Ra = [ ra[i] - 0*proper_motion_scale*self.proper_motions[i,0], ra[i] + proper_motion_scale*self.proper_motions[i,0] ]
                Dec = [ dec[i] - 0*proper_motion_scale*self.proper_motions[i,1], dec[i] + proper_motion_scale*self.proper_motions[i,1] ]
                ax.plot(Ra, Dec, '-', color='r', alpha=0.6)

        # plot grid lines
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(outfile)
        plt.clf()
            

    def pm_hist(self, outfile):
        """
        Plot a histogram of the proper motions of the quasars 
        """
        proper_motions_Cartesian = np.linalg.norm(CT.geographic_to_Cartesian_vector(self.positions, self.proper_motions), axis = 1)
        plt.hist(proper_motions_Cartesian)
            
        plt.xlabel('Proper motion [mas/yr]')
        plt.ylabel('Number of quasars')
        plt.title('Histogram of quasar proper motions')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(outfile)
        plt.clf()
            
