import pandas
import numpy

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
    
