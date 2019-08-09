import pandas
import numpy

from utils import *

class AstrometricDataframe:
    def __init__(self): 
        self.positions = numpy.array ([])
        self.positions_coord_system = ""

        self.positions_err = numpy.array ([])

        self.proper_motions = numpy.array ([])

        self.proper_motions_err = numpy.array ([])
        
        self.proper_motions_err_corr = numpy.array ([])
    

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
    new_dataframe.positions_coord_system = "Geographic"
    
    new_dataframe.positions_err = dataset.as_matrix ( columns = [ 'ra_error' , 'dec_error' ] )
    
    new_dataframe.proper_motions = dataset.as_matrix ( columns = [ 'pmra' , 'pmdec' ] )
    
    new_dataframe.proper_motions_err = dataset.as_matrix ( columns = [ 'pmra_error' , 'pmdec_error' ] )
    
    new_dataframe.proper_motions_err_corr = dataset.as_matrix ( columns = [ 'pmra_pmdec_corr' ] )
    
    return new_dataframe

def generate_scalar_bg (data):
    scale = 1.0
    err_scale = 20.0
    
    vsh_E_coeffs = [[0j, 1.0 * scale + 0j, 0j], [0j, 0j, 0j, 0j, 0j]]
    vsh_B_coeffs = [[0j, 0j, 0j], [0j, 0j, 0j, 0j, 0j]]
    
    model_pm = generate_model ( vsh_E_coeffs , vsh_B_coeffs , data.positions )
    
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
    
