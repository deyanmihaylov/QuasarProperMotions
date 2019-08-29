import pandas as pd
import numpy as np
import csv

import CoordinateTransformations as CT
import VectorSphericalHarmonics as VSH

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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
        """

        self.n_objects = 0

        self.Lmax = Lmax
        
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

        VSH_bank = {}

        for l in range(1, self.Lmax + 1):
            for m in range(-l, l+1):
                VSH_bank['Y^E_' + str(l) + ',' + str(m)] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, np.real(VSH.RealVectorSphericalHarmonicE (l, m, self.positions_Cartesian)))
                VSH_bank['Y^B_' + str(l) + ',' + str(m)] = CT.Cartesian_to_geographic_vector(self.positions_Cartesian, np.real(VSH.RealVectorSphericalHarmonicB (l, m, self.positions_Cartesian)))
                        
        return VSH_bank





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
                Ra = [ ra[i] - proper_motion_scale*self.proper_motions[i,0], ra[i] + proper_motion_scale*self.proper_motions[i,0] ]
                Dec = [ dec[i] - proper_motion_scale*self.proper_motions[i,1], dec[i] + proper_motion_scale*self.proper_motions[i,1] ]
                ax.plot(Ra, Dec, '-', color='r', alpha=0.6)

        # plot grid lines
        plt.grid(True)
            
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
        plt.savefig(outfile)
        plt.clf()
            
