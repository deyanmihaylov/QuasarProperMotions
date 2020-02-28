import numpy as np
import sys
import itertools
from scipy.stats import truncnorm
import pandas as pd

import CoordinateTransformations as CT
import VectorSphericalHarmonics as VSH
import Utils as U
import Model as M

# import csv





# 
# from scipy.linalg import cholesky

def load_astrometric_data(df,
                          Lmax=2,
                          positions=1,
                          injection=1,
                          pm_errors=1,
                          N_obj = None,
                          bunch_size = 0.,
                          dipole = 0.,
                          pm_noise = 0.
                         ):
    df.Lmax = Lmax

    df.generate_names()

    dataset_dict = {2: {"cat": "Gaia", "file_name": "data/type2.csv"},
                    3: {"cat": "Gaia", "file_name": "data/type3.csv"},
                    4: {"cat": "Gaia", "file_name": "data/type2and3.csv"},
                    5: {"cat": "TD", "file_name": "data/TD6.dat"}
                   }

    which_dataset = set([positions, injection, pm_errors]).intersection(set(dataset_dict.keys()))

    if len(which_dataset) > 1:
        sys.exit("Conflicting datasets cannot be combined.")
    elif len(which_dataset) == 1:
        chosen_dataset = next(iter(which_dataset))
        
        if dataset_dict[chosen_dataset]['cat'] == "Gaia":
            dataset = import_Gaia_dataset(dataset_dict[chosen_dataset]['file_name'])
        elif dataset_dict[chosen_dataset]['cat'] == "TD":
            dataset = import_TD_dataset(dataset_dict[chosen_dataset]['file_name'])
    else:
        dataset = None

    if dataset is None:
        df.N_obj = N_obj
    else:
        df.N_obj = dataset.shape[0]

    if positions == 1:
        df.generate_positions(method="random", bunch_size=bunch_size)
    elif positions in [2, 3, 4]:
        df.load_Gaia_positions(dataset)
    elif positions == 5:
        df.load_TD_positions(dataset)

    df.generate_VSHs()

    if injection == 1:
        df.generate_proper_motions(method="zero")
    elif injection in [2, 3, 4]:
        df.load_Gaia_proper_motions(dataset)
    elif injection == 5:
        df.load_TD_proper_motions(dataset)
    elif injection == 6:
        df.generate_proper_motions(method="dipole", dipole=dipole)
    elif injection == 7:
        df.generate_proper_motions(method="multipole")
    exit()
    if pm_errors == 1:
        df.generate_proper_motion_errors(method="zero")
    elif pm_errors in [2, 3, 4]:
        df.load_Gaia_proper_motion_errors(dataset)
    elif pm_errors == 5:
        df.load_TD_proper_motion_errors(dataset)
    elif pm_errors == 6:
        df.generate_proper_motion_errors(method="noise")
    # print(df.proper_motions)
    



    self.generate_VSH_bank()

    self.compute_overlap_matrix()

    print(df.names)

def import_Gaia_dataset(path):
    """
    Import Gaia dataset
    """
    dataset = pd.read_csv(path,
                          sep=',',
                          header='infer',
                          squeeze=False,
                          mangle_dupe_cols=True,
                          engine='python',
                          skipinitialspace=False,
                          skipfooter=0,
                          keep_default_na=True,
                          na_filter=True,
                          verbose=False,
                          skip_blank_lines=True,
                          parse_dates=False,
                          infer_datetime_format=False,
                          keep_date_col=False,
                          dayfirst=False,
                          iterator=False,
                          decimal=b'.',
                          doublequote=True,
                          error_bad_lines=True,
                          warn_bad_lines=True,
                          delim_whitespace=False,
                          low_memory=True,
                          memory_map=False,
                         )

    dropna_columns = ['ra',
            'dec',
            'pmra',
            'pmdec',
            'pmra_error',
            'pmdec_error',
            'pmra_pmdec_corr']

    dataset.dropna(axis=0,
                   how='any',
                   thresh=None,
                   subset=dropna_columns,
                   inplace=True
                  )

    return dataset

def import_TD_dataset(path):
    """
    Import TD dataset
    """

    col_names = ['Name', 'RAh', 'RAm', 'RAs', 'e_RAs', 'DEd', 'DEm',
                 'DEs', 'e_DEs', 'pmRA', 'e_pmRA', 'o_pmRA', 'chi2a', 'pmDE', 'e_pmDE',
                 'o_pmDE', 'chi2d', 'Length', 'MJD', 'Flag', 'z', 'f_z', 'r_z']

    dataset = pd.read_fwf(path,
                          colspecs='infer',
                          names=col_names,
                          widths=None,
                          comment = '#',
                          infer_nrows=500
                         )

    dropna_columns = ['RAh', 'RAm', 'RAs', 'e_RAs', 'DEd', 'DEm', 'DEs', 'e_DEs',
                      'pmRA', 'e_pmRA', 'o_pmRA', 'chi2a', 'pmDE', 'e_pmDE', 'o_pmDE', 'chi2d']

    dataset.dropna(axis=0,
                   how='any',
                   thresh=None,
                   subset=dropna_columns,
                   inplace=True
                  )
    
    return dataset


class AstrometricDataframe:
    def __init__(self):
        """
	    Initialise Class
        """
        self.N_obj = 0
        self.Lmax = None
	        
        self.positions = np.array([])
        self.positions_Cartesian = np.array([])

        self.proper_motions = np.array([])

        self.inv_proper_motion_error_matrix = np.array([])

        self.basis = dict()
        self.which_basis = None
	
        self.names = []

    def generate_names(self):
        names = {}

        for l in range(1, self.Lmax + 1):
            names[l] = dict()

            for m in range(-l, l+1):
                names[l][m] = dict()

                names[l][m]['E'] = f"Y^E_{l},{m}"
                names[l][m]['B'] = f"Y^B_{l},{m}"

        self.names = names.copy()

    def generate_VSHs(self):
        """
        Precompute VSH functions at QSO locations 
        """
        VSHs = {}

        for l in range(1, self.Lmax + 1):
            VSHs[l] = dict()

            for m in range(-l, l+1):
                VSHs[l][m] = dict()

                VSHs[l][m]['E'] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, VSH.RealVectorSphericalHarmonicE(l, m, self.positions_Cartesian))
                VSHs[l][m]['B'] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, VSH.RealVectorSphericalHarmonicB(l, m, self.positions_Cartesian))
            
        self.which_basis = "vsh"

        self.basis = VSHs.copy()

    def generate_positions(self, method=None, bunch_size=0.):
        """
        Generate random positions from a distorted distribution
    
        INPUTS
        ------
        bunch_size: float
            controls this distribution; large eps (e.g. 100) is uniform, small eps (e.g. 0.1) is very non-uniform
        """
        if method == "random":
            if bunch_size == 0.:
                theta = np.random.uniform(0, np.pi, size=self.N_obj)
            else:
                theta = 0.5*np.pi + truncnorm.rvs(-0.5*np.pi/bunch_size, 0.5*np.pi/bunch_size, scale=bunch_size, size=self.N_obj)

            phi = np.random.uniform(0, 2*np.pi, size=self.N_obj)

            ra, dec = phi, 0.5*np.pi-theta

            self.positions = np.array(list(zip(ra, dec)))
            self.positions_Cartesian = CT.geographic_to_Cartesian_point(self.positions)

    def generate_proper_motions(self, method=None, dipole=0.):
        if method == "zero":
            self.proper_motions = np.zeros((self.N_obj, 2))
        elif method == "dipole":
            almQ = dict()
            
            for l in range(1, self.Lmax+1):
                almQ[l] = dict()

                for m in range(-l, l+1):
                    almQ[l][m] = dict()

                    almQ[l][m]['E'] = 0.
                    almQ[l][m]['B'] = 0.
            
            almQ[1][0]['E'] = dipole

            self.proper_motions = M.generate_model(almQ, self.basis)
            
            # self.proper_motions += M.generate_model(almQ, self.basis)

                # if dir_path is not None:
                #     par_file_open = open(dir_path + "/injected_par.txt" , "w")
                #     par_file = csv.writer(par_file_open)
                
                #     for key, val in par.items():
                #         par_file.writerow([key, val])

                #     par_file_open.close()
            
            # TO DO: implement GR quadrupole injection
        elif method == "multipole":
            pass # implement miltipole injection

    def generate_proper_motion_errors(self, method=None, pm_noise=0.):
        if method == "zero":
            proper_motions_errors = np.zeros((self.N_obj, 2))
            proper_motions_err_corr = np.ones(self.N_obj)

            covariance = U.covariant_matrix(proper_motions_errors, proper_motions_err_corr)
                
            self.inv_proper_motion_error_matrix = np.linalg.inv(covariance)
        elif method == "noise":
            if pm_noise == 0.:
                self.generate_proper_motion_errors(method="zero")
            else:
                pm_errors = np.zeros((N_obj, 2))
                pm_errors[:, 0] = np.random.normal(0, pm_noise) * np.reciprocal(np.cos(self.positions[:,1]))
                pm_errors[:, 1] = np.random.normal(0, pm_noise) * np.ones(self.N_obj)
                pm_corr = np.zeros(self.N_obj)

                covariance = U.covariant_matrix(errors, corr)
                
                self.inv_proper_motion_error_matrix = np.linalg.inv(covariance)

    def load_Gaia_positions(self, dataset):
        """
        Load the positions from Gaia file
        """
        self.positions = dataset[['ra', 'dec']].values
        self.positions = U.deg_to_rad(self.positions)

        self.positions_Cartesian = CT.geographic_to_Cartesian_point(self.positions)

    def load_Gaia_proper_motions(self, dataset):
        """
        Load the proper motions from Gaia file
        """
        self.proper_motions = dataset[['pmra', 'pmdec']].values

    def load_Gaia_proper_motion_errors(self, dataset):
        """
        Load the proper motion errors from Gaia file
        """
        proper_motions_errors = dataset[['pmra_error', 'pmdec_error']].values
        proper_motions_errors[:,0] = proper_motions_errors[:,0] / np.cos(self.positions[:,1])

        proper_motions_err_corr = dataset[['pmra_pmdec_corr']].values
                
        covariance = U.covariant_matrix(proper_motions_errors, proper_motions_err_corr)
                
        self.inv_proper_motion_error_matrix = np.linalg.inv(covariance)

    def load_TD_positions(self, dataset):
        """
        Load the positions from Truebenbach-Darling file
        """
        hours = 360. / 24.
        mins = hours / 60.
        secs = mins / 60.

        deg = 1.
        arcmin = deg / 60. 
        arcsec = arcmin / 60.

        RAh = dataset['RAh'].values
        RAm = dataset['RAm'].values
        RAs = dataset['RAs'].values

        DEd = dataset['DEd'].values
        DEm = dataset['DEm'].values
        DEs = dataset['DEs'].values

        ra = RAh*hours + RAm*mins + RAs*secs
        dec = DEd*deg + DEm*arcmin + DEs*arcsec

        self.positions = np.transpose([ra, dec])
        self.positions = U.deg_to_rad(self.positions)

        self.positions_Cartesian = CT.geographic_to_Cartesian_point(self.positions)

    def load_TD_proper_motions(self, dataset):
        """
        Load the proper motions from Truebenbach-Darling file
        """

        self.proper_motions = dataset[['pmRA', 'pmDE']].values

    def load_TD_proper_motion_errors(self, dataset):
        """
        Load the proper motion errors from Truebenbach-Darling file
        TO DO: Use chi2 statistics for correlation
        """
        proper_motions_errors = dataset[['e_pmRA', 'e_pmDE']].values
        proper_motions_errors[:,0] = proper_motions_errors[:,0] / np.cos(self.positions[:,1])

        proper_motions_err_corr = np.zeros(self.N_obj)

        covariance = U.covariant_matrix(proper_motions_errors, proper_motions_err_corr)
                
        self.inv_proper_motion_error_matrix = np.linalg.inv(covariance)

    def load_Truebenbach_Darling_data(self, path):
        """
        Load the postions, proper motions and proper motion errors from file  
        """

        hours = 360. / 24.
        mins = hours / 60.
        secs = mins / 60.

        deg = 1.
        arcmin = deg / 60. 
        arcsec = arcmin / 60.

        ra = []
        dec = []
        pm_ra = []
        pm_dec = []
        pm_ra_err = []
        pm_dec_err = []

        with open(path) as fp:
            content = fp.readlines()

        for line in content:
            if line[0] is "#":
                pass
            else:
                name, RAh, RAm, RAs, e_RAs, DEd, DEm, DEs, e_DEs, pmRA, e_pmRA, o_pmRA, chi2a, pmDE, e_pmDE, o_pmDE, chi2d = line.split()[0:17]
                
                ra.append(int(RAh)*hours+int(RAm)*mins+float(RAs)*secs)
                dec.append(int(DEd)*deg+int(DEm)*arcmin+float(DEs)*arcsec)

                pm_ra.append(float(pmRA))
                pm_dec.append(float(pmDE))

                pm_ra_err.append(float(e_pmRA))
                pm_dec_err.append(float(e_pmDE))


        self.n_objects = len(ra)

        self.positions = np.array([ [ra[i], dec[i]] for i in range(self.n_objects)])
        self.positions = self.deg_to_rad ( self.positions )

        self.positions_Cartesian = CT.geographic_to_Cartesian_point ( self.positions )

        self.proper_motions = np.array([ [pm_ra[i], pm_dec[i]] for i in range(self.n_objects)])

        proper_motions_err = np.array([ [pm_ra_err[i], pm_dec_err[i]] for i in range(self.n_objects)])
        proper_motions_err[:,0] = proper_motions_err[:,0] / np.cos ( self.positions[:,1] )

        proper_motions_err_corr = np.zeros(self.n_objects)

        covariance = covariant_matrix ( proper_motions_err , proper_motions_err_corr )

        self.inv_proper_motion_error_matrix = np.linalg.inv ( covariance )

        self.generate_VSH_bank()

        self.compute_overlap_matrix()



 #    def gen_mock_data(self, NumObjects, eps=0.2, noise=0.1):
 #        """
 #        Generate mock postions, and proper motion errors for the QSO
	# The proper motions are just noise at this stage (the injection is handled elsewhere)
            
 #        INPUTS
 #        ------
 #        NumObjects: int
 #                Desired number of objects in mock catalog
 #        eps: float
 #                The QSO positions will be drawn from a distribution which favours the equator over the poles.
 #                eps controls this distribution; large eps (e.g. 100) is uniform, small eps (e.g. 0.1) is very non-uniform
 #        noise: float
 #                The size of the proper motion error [mas/yr]
 #        """
            
 #        self.n_objects = NumObjects

 #        # Positions
 #        self.non_uniform_random_positions(eps=eps)
	
 #        # Compute the VSH bank
 #        self.generate_VSH_bank()
 #        self.compute_overlap_matrix()
            
 #        # Proper Motion Errors
 #        errors = np.zeros((self.n_objects, 2))
 #        errors[:,0] = noise * np.reciprocal(np.cos(self.positions[:,1])) # RA error
 #        errors[:,1] = noise * np.ones(self.n_objects)                    # DEC error
 #        corr = np.zeros(self.n_objects)
 #        cov = covariant_matrix(errors, corr)
 #        self.inv_proper_motion_error_matrix = np.linalg.inv(cov)
            
 #        # Proper Motions - noise component
 #        self.proper_motions = np.zeros((self.n_objects, 2))
 #        pmra_noise = np.array([ np.random.normal(0, noise/np.cos(self.positions[i][1])) for i in range(self.n_objects)])
 #        pmdec_noise = np.random.normal(0, noise, size=self.n_objects)
 #        self.proper_motions += np.array(list(zip(pmra_noise, pmdec_noise)))
	
	
 #    def inject_proper_motions(self, dipole=0.0, quadrupole=0.0, dir_path=None):
 #        """
 #        Inject some proper motions into the data
	
 #        INPUTS
 #        ------
 #        dipole: float
	# 	the strength of the a^E_1,0 component
 #        quadrupole: float
	# 	the strength of the GW background [not implemented]
 #        """
 #        par = {}
 #        for l in range(1, self.Lmax+1):
 #            for m in range(-l, l+1):
 #                par[ 'a^E_' + str(l) + ',' + str(m) ] = 0.
 #                par[ 'a^B_' + str(l) + ',' + str(m) ] = 0.
 #        par[ 'a^E_1,0' ] = dipole
 #        self.proper_motions += generate_model(par, self.basis)

 #        if dir_path is not None:
 #            par_file_open = open(dir_path + "/injected_par.txt" , "w")
 #            par_file = csv.writer(par_file_open)
        
 #            for key, val in par.items():
 #                par_file.writerow([key, val])

 #            par_file_open.close()
	
	# # TO DO: implement GR quadrupole injection
	

 #    def normalize_matrix(self, matrix):
 #        """
 #        Normalize the overlap matrix so that the diagonals are of order 1e0.

 #        matrix: numpy.ndarray
 #            the matrix to be normalized
 #        """
 #        return matrix / ( np.linalg.det(matrix) ** (1./(2*self.Lmax*(self.Lmax+2))) )

 #    def compute_overlap_matrix(self, weighted_overlaps=True):
 #        """
 #        Calculate the overlap matrix (and its Cholesky decomposition) between VSH basis functions

 #        weighted_overlaps: bool
 #            whether or not to use the error weighted overlap sums
 #        """
 #        prefactor = 4 * np.pi / self.n_objects

 #        self.overlap_matrix = np.zeros((len(self.names), len(self.names)))

 #        metric = np.zeros((self.n_objects,2,2))
 #        metric[:,0,0] = np.cos(self.positions[:,1].copy())**2.
 #        metric[:,1,1] = 1.

 #        for i, name_x in enumerate(self.names):
 #            for j, name_y in enumerate(self.names):
 #                if weighted_overlaps is True:
 #                    self.overlap_matrix[i,j] = prefactor * np.einsum ( "...i,...ij,...j->...", self.basis[name_x], self.inv_proper_motion_error_matrix, self.basis[name_y]).sum()
 #                else:
 #                    self.overlap_matrix[i,j] = prefactor * np.einsum ( "...i,...ij,...j->...", self.basis[name_x], metric, self.basis[name_y]).sum()

 #        self.overlap_matrix = self.normalize_matrix(self.overlap_matrix)
        
 #        # compute Cholesky decompo of overlap matrix
 #        self.Cholesky_overlap_matrix = cholesky(self.overlap_matrix)

 #    def change_basis(self):
 #        """
 #        Method to change from VSH basis to orthogonal basis
 #        """
	
 #        new_basis = dict( { name:np.zeros((self.n_objects, 2)) for name in self.names } )
    
 #        invL = np.linalg.inv(self.Cholesky_overlap_matrix)

 #        for i in range(self.n_objects):
 #            v = np.array([ self.basis[name][i] for name in self.names])
 #            u = np.einsum('ij,ik->jk', invL, v)
 #            for j, name in enumerate(self.names):
 #                new_basis[name][i] = u[j]

 #        self.basis = new_basis
 #        self.which_basis = "modified orthogonal basis"

 #        self.compute_overlap_matrix()


 #    def plot_overlap_matrix(self, output):
 #        """
 #        Plot an overlap matrix
 #        """
 #        plt.imshow(self.overlap_matrix)

 #        names = self.names
 #        if self.which_basis == "modified orthogonal basis":
 #            names = [ name.replace("Y","T") for name in names]

 #        plt.xticks(np.arange(len(names)), names, rotation=90)
 #        plt.yticks(np.arange(len(names)), names)

 #        plt.colorbar()

 #        plt.tight_layout()
 #        plt.savefig(output)
 #        plt.clf()

 #    def eccentricity(self, axes):
 #        return np.sqrt(1. - (np.min(axes, axis=1)/np.max(axes, axis=1))**2)

 #    def ecc_hist(self, outfile):
 #        inv_matrix = self.inv_proper_motion_error_matrix.copy()
        
 #        inv_matrix[...,0,0] = self.inv_proper_motion_error_matrix[...,0,0] / (np.cos(self.positions[:,1]) ** 2)
 #        inv_matrix[...,1,0] = self.inv_proper_motion_error_matrix[...,1,0] / np.cos(self.positions[:,1])
 #        inv_matrix[...,0,1] = self.inv_proper_motion_error_matrix[...,0,1] / np.cos(self.positions[:,1])

 #        sq_eigenvalues, eigenvectors = np.linalg.eig(inv_matrix)

 #        eigenvalues = np.sqrt(sq_eigenvalues)

 #        eccentricities = self.eccentricity(eigenvalues)

 #        plt.hist(eccentricities)
            
 #        plt.xlabel('PM errors eccentricity')
 #        plt.ylabel('Number of quasars')
 #        plt.title('Histogram of PM errors eccentricity')
 #        plt.yscale('log')

 #        plt.tight_layout()
 #        plt.savefig(outfile)
 #        plt.clf()
        
 #    def plot(self, outfile, proper_motions=False, projection='mollweide', proper_motion_scale=1):
 #        """
 #        method to plot positions (and optionally pms) of QSOs in dataframe
 #        """
 #        fig = plt.figure()
 #        ax = fig.add_subplot(111, projection=projection)

 #        ra = np.array([ x-2*np.pi if x>np.pi else x for x in self.positions[:,0]])
 #        dec = self.positions[:,1]

 #        # plot the positions
 #        ax.plot(ra, dec, 'o', color='r', markersize=1, alpha=0.8)

 #        # plot the proper motions 
 #        if proper_motions:
 #            Nstars = len(self.positions)
 #            for i in range(Nstars):
 #                Ra = [ ra[i] - 0*proper_motion_scale*self.proper_motions[i,0], ra[i] + proper_motion_scale*self.proper_motions[i,0] ]
 #                Dec = [ dec[i] - 0*proper_motion_scale*self.proper_motions[i,1], dec[i] + proper_motion_scale*self.proper_motions[i,1] ]
 #                ax.plot(Ra, Dec, '-', color='r', alpha=0.6)

 #        # plot grid lines
 #        plt.grid(True)
        
 #        plt.tight_layout()
 #        plt.savefig(outfile)
 #        plt.clf()
            

 #    def pm_hist(self, outfile):
 #        """
 #        Plot a histogram of the proper motions of the quasars 
 #        """
 #        proper_motions_Cartesian = np.linalg.norm(CT.geographic_to_Cartesian_vector(self.positions, self.proper_motions), axis = 1)
 #        plt.hist(proper_motions_Cartesian)
            
 #        plt.xlabel('Proper motion [mas/yr]')
 #        plt.ylabel('Number of quasars')
 #        plt.title('Histogram of quasar proper motions')
 #        plt.yscale('log')

 #        plt.tight_layout()
 #        plt.savefig(outfile)
 #        plt.clf()

 #    def plot_astrometric_data(self, dir_path):
 #        self.plot(dir_path + "/qso_positions.png")
 #        self.pm_hist(dir_path + "/qso_pm_hist.png")
 #        self.ecc_hist(dir_path + "/qso_err_ecc_hist.png")
 #        self.plot_overlap_matrix(dir_path + "/qso_vsh_overlaps.png")
        
