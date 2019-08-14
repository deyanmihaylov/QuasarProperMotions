import pandas
import numpy

from CoordinateTransformations import *
from utils import *

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
    

def import_Gaia_data (path_to_Gaia_data):
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

    dataset.dropna(axis=0,
                   how='any',
                   thresh=None,
                   subset=dropna_columns,
                   inplace=True)
    
    new_dataframe = AstrometricDataframe()
    
    new_dataframe.positions = dataset.as_matrix ( columns = [ 'ra' , 'dec' ] )
    
    new_dataframe.positions_err = dataset.as_matrix ( columns = [ 'ra_error' , 'dec_error' ] )
    
    new_dataframe.proper_motions = dataset.as_matrix ( columns = [ 'pmra' , 'pmdec' ] )
    
    new_dataframe.proper_motions_err = dataset.as_matrix ( columns = [ 'pmra_error' , 'pmdec_error' ] )
    
    new_dataframe.proper_motions_err_corr = dataset.as_matrix ( columns = [ 'pmra_pmdec_corr' ] )
    
    new_dataframe.covariance = covariant_matrix ( new_dataframe.proper_motions_err , new_dataframe.proper_motions_err_corr )
    
    new_dataframe.covariance_inv = numpy.linalg.inv ( new_dataframe.covariance )
    
    new_dataframe.positions_Cartesian = geographic_to_Cartesian ( new_dataframe.positions )
    
    return new_dataframe

def generate_VSH_bank (data , Lmax):
    VSH_bank = {}

    for l in range ( 1 , Lmax + 1 ):
        VSH_bank['Re[Y^E_' + str(l) + '0]'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicE ( l , 0 , data.positions_Cartesian ) ) )

        VSH_bank['Re[Y^B_' + str(l) + '0]'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicB ( l , 0 , data.positions_Cartesian ) ) )

        for m in range ( 1 , l + 1 ):
            VSH_bank['Re[Y^E_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicE ( l , m , data.positions_Cartesian ) ) )

            VSH_bank['Im[Y^E_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.imag ( VectorSphericalHarmonicE ( l , m , data.positions_Cartesian ) ) )

            VSH_bank['Re[Y^B_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.real ( VectorSphericalHarmonicB ( l , m , data.positions_Cartesian ) ) )

            VSH_bank['Im[Y^B_' + str(l) + str(m) + ']'] = Cartesian_to_geographic_vector (data.positions_Cartesian , numpy.imag ( VectorSphericalHarmonicB ( l , m , data.positions_Cartesian ) ) )
            
    return VSH_bank

def generate_scalar_bg ( data , Lmax , VSH_bank):
    scale = 1.0
    err_scale = 1.0
    
    par = {}
    
    for l in range(1,Lmax+1):
        for m in range(0, l+1):
            if m==0:
                if l==1:
                    par['Re_a^E_'+str(l)+'0'] = 1.
                else:
                    par['Re_a^E_'+str(l)+'0'] = 0.
                
                par['Re_a^B_'+str(l)+'0'] = 0.
            else:
                par['Re_a^E_'+str(l)+str(m)] = 0.
                par['Im_a^E_'+str(l)+str(m)] = 0.
                par['Re_a^B_'+str(l)+str(m)] = 0.
                par['Im_a^B_'+str(l)+str(m)] = 0.
    
    model_pm = generate_model ( par , VSH_bank , Lmax)
    
    data.proper_motions = model_pm

    data.proper_motions_err = err_scale * numpy.ones(data.proper_motions_err.shape, dtype=None, order='C')
    data.proper_motions_err_corr = numpy.zeros(data.proper_motions_err_corr.shape, dtype=None, order='C')
    
    return data

def generate_gr_bg (data):
    scale = 1.0

    variance = numpy.array([ 0.0 , 0.3490658503988659 , 0.03490658503988659 , 0.006981317007977318 , 0.0019946620022792336 ])

    vsh_E_coeffs = [ [ scale * (numpy.random.normal(0.0 , numpy.sqrt(variance[l-1])) + (1j) * numpy.random.normal(0.0 , numpy.sqrt(variance[l-1]))) for m in range (-l,l+1)] for l in range (1,len(variance)+1)]

    for l , l_coeffs in enumerate(vsh_E_coeffs):
        L = l+1
        for m in range (-L, L+1):
            if m < 0:
                l_coeffs[m+L] = ((-1)**(-m)) * numpy.conj (l_coeffs[-m+L])
            elif m == 0:
                l_coeffs[L] = numpy.real(l_coeffs[L]) + (1j) * 0.0
                
    vsh_B_coeffs = [ [ scale * (numpy.random.normal(0.0 , numpy.sqrt(variance[l-1])) + (1j) * numpy.random.normal(0.0 , numpy.sqrt(variance[l-1]))) for m in range (-l,l+1)] for l in range (1,len(variance)+1)]

    for l , l_coeffs in enumerate(vsh_B_coeffs):
        L = l+1
        for m in range (-L,L+1):
            if m < 0:
                l_coeffs[m+L] = ((-1)**(-m)) * numpy.conj (l_coeffs[-m+L])
            elif m == 0:
                l_coeffs[L] = numpy.real(l_coeffs[L]) + (1j) * 0.0

    model_pm = generate_model ( vsh_E_coeffs , vsh_B_coeffs , data.positions )
    
    data.proper_motions = model_pm

    data.proper_motions_err = scale * numpy.ones(data.proper_motions_err.shape, dtype=None, order='C')
    data.proper_motions_err_corr = numpy.zeros(data.proper_motions_err_corr.shape, dtype=None, order='C')
    
    return data
    
