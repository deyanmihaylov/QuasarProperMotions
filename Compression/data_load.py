import pandas
import os
import numpy as np



from CoordinateConversions import X, Y



class AstrometricDataframe:
    
   def __init__(self): 
        
      # Geographic coord positions: degrees
      self.positions = np.array ([])
        
      # Cartesian positions: unit three vectors
      self.positions_Cartesian = np.array ([])

      # Geographic coord proper motions and errors: mas/year
      self.proper_motions = np.array ([])
      self.proper_motions_err = np.array ([])
      self.proper_motions_err_corr = np.array ([])
      self.proper_motions_invcov = np.array ([])
    




def import_Gaia_data (path_to_Gaia_data):
   """
   Load Gaia data from file into AstrometricDataframe

   INPUTS
   ------
   path_to_Gaia_data: str
       path to the .csv data file
   
   RETURNS
   -------
   new_dataframe: AstrometricDataframe
   """

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

   N = len(dataset)
   print("Loading data from file. Number of objects =", N)
    
   # positions and errors geographic
   new_dataframe.positions = dataset[ ['ra', 'dec'] ].to_numpy()
   new_dataframe.positions_err = dataset[ ['ra_error', 'dec_error'] ].to_numpy()
    
   # proper motions and errors geographic
   new_dataframe.proper_motions = dataset[ ['pmra', 'pmdec'] ].to_numpy()
   new_dataframe.proper_motions_err = dataset[ ['pmra_error', 'pmdec_error'] ].to_numpy()
   new_dataframe.proper_motions_err_corr = dataset[ 'pmra_pmdec_corr' ].to_numpy()
        
   raerr = dataset[ 'pmra_error' ].to_numpy()
   decerr = dataset[ 'pmra_error' ].to_numpy()
   corr = dataset[ 'pmra_pmdec_corr' ].to_numpy()
   new_dataframe.proper_motions_invcov = np.array([
            np.linalg.inv([[ raerr[i]**2, raerr[i]*decerr[i]*corr[i] ] , [ raerr[i]*decerr[i]*corr[i], decerr[i]**2 ]]) 
            for i in range(N)])

   # positions Cartesian
   ra = dataset[ 'ra' ].to_numpy()
   dec = dataset[ 'dec' ].to_numpy()
   new_dataframe.positions_Cartesian = np.array([ 
                                                        geographic_to_Cartesian ( np.array([ra[i], dec[i]]) )
                                            for i in range(N)])
    
   return new_dataframe






def CompressDataFrame(dataframe, compression_level=1):
    """
    Compress the dataframe object onto a grid

    INPUTS
    ------
    dataframe: AstrometricDataframe
        Data to be compressed
    compression_level: int
        Which grid to use. Must be one of 1, 2, ... , 10. Lower numbers are coarser, losier grids.

    RETURNS
    -------
    new_dataframe: AstrometricDataframe
        Compressed version of dataframe
    """


    new_dataframe = AstrometricDataframe()


    # grids 
    virtual_QSO_file = "../../AstroGW/grids/grid"+str(compression_level)+"/star_positions.dat"
    assert os.path.isfile(virtual_QSO_file)
    QSOs = np.loadtxt(virtual_QSO_file, delimiter='\t')

    N = len(QSOs)
    print("Compressing data onto grid {} with {} virtual QSO".format(compression_level, N))


    # Cartesian positions
    new_dataframe.positions_Cartesian = QSOs

    # Geographic coord positions and errors
    new_dataframe.positions = 1 # Geo to Cart(QSOs)

    # Geographic coord proper motions and errors
    new_dataframe.proper_motions = 1 # ???
    new_dataframe.proper_motions_err = 1 # ???
    new_dataframe.proper_motions_err_corr = 1 # ???
    new_dataframe.proper_motions_invcov = 1 # ???


    return new_dataframe
