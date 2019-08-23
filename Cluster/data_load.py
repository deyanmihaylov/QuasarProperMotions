import pandas
import numpy
import csv

from CoordinateTransformations import *
from utils import *

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as c

class AstrometricDataframe:
    def __init__(self): 
        self.positions = numpy.array ([])

        self.positions_Cartesian = numpy.array ([])
        
        self.positions_err = numpy.array ([])

        self.proper_motions = numpy.array ([])

        self.proper_motions_err = numpy.array ([])
        
        self.proper_motions_err_corr = numpy.array ([])
        
        self.covariance = numpy.array ([])
        
        self.covariance_inv = numpy.array ([])
        
        self.positions_Cartesian = numpy.array ([])
        
        
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
                Ra = [ ra[i] - proper_motion_scale*self.proper_motions[i,0],
                       ra[i] + proper_motion_scale*self.proper_motions[i,0] ]
                Dec = [ dec[i] - proper_motion_scale*self.proper_motions[i,1],
                        dec[i] + proper_motion_scale*self.proper_motions[i,1] ]
                ax.plot(Ra, Dec, '-', color='r', alpha=0.6)
                    
        # plot grid lines 
        plt.grid ( True )
    
        plt.savefig ( outfile )

    def pm_hist ( self , outfile ):
        # Plot a histogram of the proper motions of the quasars

        proper_motions_Cartesian = numpy.linalg.norm ( geographic_to_Cartesian_vector ( self.positions , self.proper_motions ) , axis = 1 )

        n_bins = int ( numpy.ceil ( proper_motions_Cartesian.max() ) )

        fig = plt.figure()
        plt.hist ( proper_motions_Cartesian ,
                   bins=n_bins ,
                   range=( 0.0 , float(n_bins) ) ,
                   density=None ,
                   weights=None ,
                   cumulative=False ,
                   bottom=None ,
                   histtype='bar' ,
                   align='mid' ,
                   orientation='vertical' ,
                   rwidth=0.85 ,
                   log=False ,
                   color=None ,
                   label=None ,
                   stacked=False ,
                   normed=None ,
                   data=None )

        plt.xlabel ( 'Proper motion [mas/yr]' )
        plt.ylabel ( 'Number of quasars' )
        plt.xticks ( ticks = range ( 0 , n_bins+1 ) , labels=None )
        plt.title ( 'Histogram of quasar proper motion' )
        plt.yscale ( 'log' )
        plt.grid ( False )
        plt.savefig ( outfile )
        
        
def import_Gaia_data (path_to_Gaia_data):
    def deg_to_rad ( degree_vals ):
        return numpy.deg2rad ( degree_vals )
    
    def generate_VSH_bank ( data ):
        VSH_bank = {}

        for l in range ( 1 , c.Lmax + 1 ):
            VSH_bank['Re[Y^E_' + str(l) + '0]'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicE ( l , 0 , data.positions_Cartesian ) ) )

            VSH_bank['Re[Y^B_' + str(l) + '0]'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicB ( l , 0 , data.positions_Cartesian ) ) )

            for m in range ( 1 , l + 1 ):
                VSH_bank['Re[Y^E_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicE ( l , m , data.positions_Cartesian ) ) )

                VSH_bank['Im[Y^E_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.imag ( VectorSphericalHarmonicE ( l , m , data.positions_Cartesian ) ) )

                VSH_bank['Re[Y^B_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicB ( l , m , data.positions_Cartesian ) ) )

                VSH_bank['Im[Y^B_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.imag ( VectorSphericalHarmonicB ( l , m , data.positions_Cartesian ) ) )
            
        return VSH_bank
    
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
                      'ra_error',
                      'dec_error',
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
    
    new_dataframe = AstrometricDataframe()
    
    new_dataframe.positions = dataset[[ 'ra' , 'dec' ]].values
    new_dataframe.positions = deg_to_rad ( new_dataframe.positions )
    
    new_dataframe.positions_err = dataset[[ 'ra_error' , 'dec_error' ]].values
    new_dataframe.positions_err[:,0] = new_dataframe.positions_err[:,0] / numpy.cos ( new_dataframe.positions[:,1] )
    
    new_dataframe.proper_motions = dataset[[ 'pmra' , 'pmdec' ]].values
    
    new_dataframe.proper_motions_err = dataset[[ 'pmra_error' , 'pmdec_error' ]].values
    
    new_dataframe.proper_motions_err_corr = dataset[[ 'pmra_pmdec_corr' ]].values
    
    new_dataframe.covariance = covariant_matrix ( new_dataframe.proper_motions_err , new_dataframe.proper_motions_err_corr )
    
    new_dataframe.covariance_inv = numpy.linalg.inv ( new_dataframe.covariance )
    
    new_dataframe.positions_Cartesian = geographic_to_Cartesian_point ( new_dataframe.positions )
    
    new_dataframe.VSH = generate_VSH_bank ( new_dataframe )
    
    return new_dataframe

def covariant_matrix ( errors , corr ):
    # Compute the covariant matrix from errors and correlation

    covariant_matrix = numpy.einsum ( '...i,...j->...ij' , errors , errors )
    
    covariant_matrix[...,0,1] = covariant_matrix[...,1,0] = numpy.multiply ( covariant_matrix[...,1,0] , corr.flatten() )
    return covariant_matrix
