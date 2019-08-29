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

        self.basis = dict()
        self.which_basis = ""


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
                
        self.generate_VSH_bank()

        self.compute_overlap_matrix()
	
	
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


    def gen_mock_data(self, NumObjects, eps=0.2, noise=0.1):
        """
        Generate mock postions, and proper motion errors for the QSO
	The proper motions are just noise at this stage (the injection is handled elsewhere)
            
        INPUTS
        ------
        NumObjects: int
                Desired number of objects in mock catalog
        eps: float
                The QSO positions will be drawn from a distribution which favours the equator over the poles.
                eps controls this distribution; large eps (e.g. 100) is uniform, small eps (e.g. 0.1) is very non-uniform
        noise: float
                The size of the proper motion error [mas/yr]
        """
            
        self.n_objects = NumObjects

        # Positions
        self.non_uniform_random_positions(eps=eps)
	
        # Compute the VSH bank
        self.generate_VSH_bank()
        self.compute_overlap_matrix()
            
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
	
	
    def inject_proper_motions(self, dipole=0.0, quadrupole=0.0, dir_path=None):
        """
        Inject some proper motions into the data
	
        INPUTS
        ------
        dipole: float
		the strength of the a^E_1,0 component
        quadrupole: float
		the strength of the GW background [not implemented]
        """
        par = {}
        for l in range(1, self.Lmax+1):
            for m in range(-l, l+1):
                par[ 'a^E_' + str(l) + ',' + str(m) ] = 0.
                par[ 'a^B_' + str(l) + ',' + str(m) ] = 0.
        par[ 'a^E_1,0' ] = dipole
        self.proper_motions += generate_model(par, self.basis)

        if dir_path is not None:
            par_file_open = open(dir_path + "/injected_par.txt" , "w")
            par_file = csv.writer(par_file_open)
        
            for key, val in par.items():
                par_file.writerow([key, val])

            par_file_open.close()
	
	# TO DO: implement GR quadrupole injection
	

    def generate_VSH_bank(self):
        """
        Precompute VSH functions at QSO locations 
        """

        VSH_bank = {}

        for l in range(1, self.Lmax + 1):
            for m in range(-l, l+1):
                VSH_bank['Y^E_' + str(l) + ',' + str(m)] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, np.real(VSH.RealVectorSphericalHarmonicE (l, m, self.positions_Cartesian)))
                VSH_bank['Y^B_' + str(l) + ',' + str(m)] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, np.real(VSH.RealVectorSphericalHarmonicB (l, m, self.positions_Cartesian)))
            
        self.which_basis = "VSH basis"

        self.basis = VSH_bank

    def compute_overlap_matrix(self):
        """
        Calculate the overlap matrix (and its Cholesky decomposition) between VSH basis functions
        """

        self.names = []

        for l in range(1, self.Lmax+1):
            for m in range(-l, l+1):
                for Q in ["E", "B"]:
                    self.names.append("Y^"+Q+"_"+str(l)+","+str(m))

        basis_Cart = {name: CT.geographic_to_Cartesian_vector(self.positions, self.basis[name]) for name in self.names}

        self.overlap_matrix = np.zeros((len(self.names), len(self.names)))

        prefactor = 4 * np.pi / self.n_objects

        for i, name_x in enumerate(self.names):
            for j, name_y in enumerate(self.names):
                self.overlap_matrix[i,j] = prefactor * np.einsum ( "...j,...j->...", basis_Cart[name_x], basis_Cart[name_y]).sum()

        # compute Cholesky decompo of overlap matrix
        self.Cholesky_overlap_matrix = cholesky(self.overlap_matrix)

    def change_basis(self):
        """
        Method to change from VSH basis to orthogonal basis
        """
	
        new_basis = dict( { name:np.zeros((self.n_objects, 2)) for name in self.names } )
    
        invL = np.linalg.inv(self.Cholesky_overlap_matrix)

        for i in range(self.n_objects):
            v = np.array([ self.basis[name][i] for name in self.names])
            u = np.einsum('ij,ik->jk', invL, v)
            for j, name in enumerate(self.names):
                new_basis[name][i] = u[j]

        self.basis = new_basis
        self.which_basis = "modified orthogonal basis"

        self.compute_overlap_matrix()


    def plot_overlap_matrix(self, output):
        """
        Plot an overlap matrix
        """
        plt.imshow(self.overlap_matrix)

        names = self.names
        if self.which_basis == "modified orthogonal basis":
            names = [ name.replace("Y","T") for name in names]

        plt.xticks(np.arange(len(names)), names, rotation=90)
        plt.yticks(np.arange(len(names)), names)

        plt.colorbar()

        plt.tight_layout()
        plt.savefig(output)
        plt.clf()

    def eccentricity(self, axes):
        return np.sqrt(1. - (np.min(axes, axis=1)/np.max(axes, axis=1))**2)

    def ecc_hist(self, outfile):
        self.inv_proper_motion_error_matrix[...,0,0] = self.inv_proper_motion_error_matrix[...,0,0] / (np.cos(self.positions[:,1]) ** 2)
        self.inv_proper_motion_error_matrix[...,1,0] = self.inv_proper_motion_error_matrix[...,1,0] / np.cos(self.positions[:,1])
        self.inv_proper_motion_error_matrix[...,0,1] = self.inv_proper_motion_error_matrix[...,0,1] / np.cos(self.positions[:,1])

        sq_eigenvalues, eigenvectors = np.linalg.eig(self.inv_proper_motion_error_matrix)

        eigenvalues = np.sqrt(sq_eigenvalues)

        eccentricities = self.eccentricity(eigenvalues)

        plt.hist(eccentricities)
            
        plt.xlabel('PM errors eccentricity')
        plt.ylabel('Number of quasars')
        plt.title('Histogram of PM errors eccentricity')
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig(outfile)
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

    def plot_astrometric_data(self, dir_path):
        self.plot(dir_path + "/qso_positions.png")
        self.pm_hist(dir_path + "/qso_pm_hist.png")
        self.ecc_hist(dir_path + "/qso_err_ecc_hist.png")
        self.plot_overlap_matrix(dir_path + "/qso_vsh_overlaps.png")
        
