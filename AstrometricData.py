import pandas as pd
import numpy as np
import csv

import CoordinateTransformations as CT
import VectorSphericalHarmonics as VSH

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class AstrometricDataframe:
    def __init__(self):

        self.n_objects = 0
        
        self.positions = np.array ([])
        self.positions_Cartesian = np.array ([])

        self.proper_motions = np.array ([])

        self.inv_proper_motion_error_matrix = np.array ([])

        self.VSH_bank = dict()


        def load_Gaia_data(self , path):
            """
            Load the postions, proper motions and proper motion errors from file
            """
            
            def deg_to_rad ( degree_vals ):
                return np.deg2rad ( degree_vals )
                

                
            dataset = pandas.read_csv(path_to_Gaia_data,
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
			self.positions = deg_to_rad ( new_dataframe.positions )

			self.positions_Cartesian = CT.geographic_to_Cartesian_point ( self.positions )

			self.n_objects = dataset.shape[0]
                                
			proper_motions = dataset[[ 'pmra' , 'pmdec' ]].values
                
			proper_motions_err = dataset[[ 'pmra_error' , 'pmdec_error' ]].values
			proper_motions_err[:,0] = proper_motions_err[:,0] / np.cos ( self.positions[:,1] )

			proper_motions_err_corr = dataset[[ 'pmra_pmdec_corr' ]].values
                
            covariance = covariant_matrix ( proper_motions_err , proper_motions_err_corr )
                
            self.inv_proper_motion_error_matrix = np.linalg.inv ( covariance )
                
            self.VSH_bank = generate_VSH_bank ( self )
                
            return 0

        def gen_mock_data(self):
            """
            Simulate the postions, proper motions and proper motion errors
            """
            pass

        def generate_VSH_bank(self):
            """
            Precompute VSH functions at QSO locations 
            """
            
            # VSH_bank = {}

            # for l in range ( 1 , c.Lmax + 1 ):
            #     VSH_bank['Re[Y^E_' + str(l) + '0]'] = Cartesian_to_geographic_vector (data.positions_Cartesian , np.real ( VectorSphericalHarmonicE ( l , 0 , data.positions_Cartesian ) ) )

            #     VSH_bank['Re[Y^B_' + str(l) + '0]'] = Cartesian_to_geographic_vector (data.positions_Cartesian , np.real ( VectorSphericalHarmonicB ( l , 0 , data.positions_Cartesian ) ) )

            #     for m in range ( 1 , l + 1 ):
            #         VSH_bank['Re[Y^E_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , np.real ( VectorSphericalHarmonicE ( l , m , data.positions_Cartesian ) ) )

            #         VSH_bank['Im[Y^E_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , np.imag ( VectorSphericalHarmonicE ( l , m , data.positions_Cartesian ) ) )

            #         VSH_bank['Re[Y^B_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , np.real ( VectorSphericalHarmonicB ( l , m , data.positions_Cartesian ) ) )

            #         VSH_bank['Im[Y^B_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , np.imag ( VectorSphericalHarmonicB ( l , m , data.positions_Cartesian ) ) )
                        
            # return VSH_bank





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

        
                
