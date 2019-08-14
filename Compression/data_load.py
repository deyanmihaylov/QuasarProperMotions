import pandas
import os

import numpy as np

import matplotlib.pyplot as plt


from CoordinateConversions import Cartesian_to_geographic, geographic_to_Cartesian



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


   def plot(self, proper_motions=False, outfile=None, projection='mollweide', proper_motion_scale=1):
      """
      method to plot the positions (and optionally the pms) of all QSOs in dataframe
      """

      fig = plt.figure()
      ax = fig.add_subplot(111, projection=projection)

      ra = self.positions[:,0]
      ra = np.array([ x-360 if x>180 else x for x in ra]) * (np.pi/180.)
      dec = self.positions[:,1] * (np.pi/180.)

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
      plt.grid(True)
      
      if outfile is not None:
         plt.savefig(outfile)







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


def rand_unit_three_vector():
   p = np.random.normal(size=3)
   return p/np.linalg.norm(p)

def mock_dipole(Nstars, sigma=1):
   """
   Create a mock astrometric catalog with a simple dipole pattern and no noise
   
   INPUTS
   ------
   Nstars: int
       The number of stars in the data set
   sigma: float
       The error on the proper motions [mas/year]

   RETURNS
   -------
   dataframe: AstrometricDataframe
   """

   dataframe = AstrometricDataframe()

   # Cartesian positions: unit three vectors                                                                                                                                                            
   dataframe.positions_Cartesian = np.array ([ rand_unit_three_vector() for i in range(Nstars)])

   # Geographic coord positions: degrees  
   dataframe.positions = Cartesian_to_geographic(dataframe.positions_Cartesian)

   # Geographic coord proper motions and errors: mas/year                                                                                                                                               
   dataframe.proper_motions = np.array([ [0, np.cos(dataframe.positions[i,1]*(np.pi/180.))] for i in range(Nstars)])
   #dataframe.proper_motions_err = NOT NEEDED
   #dataframe.proper_motions_err_corr = NOT NEEDED
   dataframe.proper_motions_invcov = np.array ([ np.diag([sigma**2, sigma**2]) for i in range(Nstars)])

   return dataframe
   
def mock_dipole(Nstars, sigma=1):
   """
   Create a mock astrometric catalog with a GW pattern and no noise

   INPUTS 
   ------
   Nstars: int
       The number of stars in the data set
       
   sigma: float
       The error on the proper motions [mas/year]                                                                                                                                                                                                           
   RETURNS
   -------
   dataframe: AstrometricDataframe
   """

   dataframe = AstrometricDataframe()

   # Cartesian positions: unit three vectors
   dataframe.positions_Cartesian = np.array ([ rand_unit_three_vector() for i in range(Nstars)])

   # Geographic coord positions: degrees
   dataframe.positions = Cartesian_to_geographic(dataframe.positions_Cartesian)

   # Geographic coord proper motions and errors: mas/year 
   scale = 1.
   variance = numpy.array([ 0.0 , 0.3490658503988659 , 0.03490658503988659 , 0.006981317007977318 , 0.0019946620022792336 ])

   vsh_E_coeffs = [ [ 
         scale * (numpy.random.normal(0.0 , numpy.sqrt(variance[l-1])) + (1j) * numpy.random.normal(0.0 , numpy.sqrt(variance[l-1]))) 
         for m in range (-l,l+1)] for l in range (1,len(variance)+1)]

   for l , l_coeffs in enumerate(vsh_E_coeffs):
      L = l+1
      for m in range (-L, L+1):
         if m < 0:
            l_coeffs[m+L] = ((-1)**(-m)) * numpy.conj (l_coeffs[-m+L])
         elif m == 0:
            l_coeffs[L] = numpy.real(l_coeffs[L]) + (1j) * 0.0
                
   vsh_B_coeffs = [ [ 
         scale * (numpy.random.normal(0.0 , numpy.sqrt(variance[l-1])) + (1j) * numpy.random.normal(0.0 , numpy.sqrt(variance[l-1]))) 
         for m in range (-l,l+1)] for l in range (1,len(variance)+1)]

   for l , l_coeffs in enumerate(vsh_B_coeffs):
      L = l+1
      for m in range (-L,L+1):
         if m < 0:
            l_coeffs[m+L] = ((-1)**(-m)) * numpy.conj (l_coeffs[-m+L])
         elif m == 0:
            l_coeffs[L] = numpy.real(l_coeffs[L]) + (1j) * 0.0

   dataframe.proper_motions = generate_model ( vsh_E_coeffs , vsh_B_coeffs , data.positions )

   dataframe.proper_motions_invcov = np.array ([ np.diag([sigma**2, sigma**2]) for i in range(Nstars)])

   return dataframe

   















def WhichVoronoiCell(virtual_Cartesian, QSO_Cartesian):
   """
   Determines which voronoi cell each star belongs to.

   INPUTS
   ------
   virtual_Cartesian: array shape (3,N)
       The Cartesian coord of the virtual QSOs
   QSO_Cartesian: array shape (3,N)
       The Cartesian coord of the objects to compress

   RETURNS
   -------
   ans: list len QSO_Cartesian
       for each object an int describing to which virtual QSO is belongs
   """

   N = len(QSO_Cartesian)
   ans = [1 for i in range(N)]

   for i in range(N):
      x = [np.dot(QSO_Cartesian[i], p) for p in virtual_Cartesian]
      ans[i] = np.argmax(x, axis=0)

   return ans



def CompressDataFrame(dataframe, compression_level=None):
    """
    Compress the dataframe object onto a grid

    INPUTS
    ------
    dataframe: AstrometricDataframe
        Data to be compressed
    compression_level: int
        Which grid to use. Must be one of 1, 2, ... , 10. 
        Lower numbers are coarser, losier grids. 
        If None then no compression is used

    RETURNS
    -------
    new_dataframe: AstrometricDataframe
        Compressed version of dataframe
    """
    
    if compression_level is None:
       return dataframe


    N_real_stars = len(dataframe.positions)

    new_dataframe = AstrometricDataframe()


    # load the grid 
    virtual_QSO_file = "../../AstroGW/grids/grid"+str(compression_level)+"/star_positions.dat"
    assert os.path.isfile(virtual_QSO_file)
    QSOs = np.loadtxt(virtual_QSO_file, delimiter='\t')

    N_virtual_stars = len(QSOs)
    print("Compressing data onto grid {} with {} virtual QSO".format(compression_level, N_virtual_stars))


    # Cartesian positions
    new_dataframe.positions_Cartesian = QSOs

    # Geographic coord positions
    new_dataframe.positions = Cartesian_to_geographic(QSOs)

    vor_cells = WhichVoronoiCell(QSOs, dataframe.positions_Cartesian)

    # Geographic coord proper motions and errors  
    new_dataframe.proper_motions_invcov = np.array([ np.zeros((2,2)) for i in range(N_virtual_stars)])
    for i in range(N_real_stars):
       index = vor_cells[i]
       new_dataframe.proper_motions_invcov[index] += dataframe.proper_motions_invcov[i]

    new_dataframe.proper_motions = np.array([ np.zeros(2) for i in range(N_virtual_stars)])
    for i in range(N_real_stars):
       index = vor_cells[i]
       new_dataframe.proper_motions[index] += np.dot( dataframe.proper_motions_invcov[i], dataframe.proper_motions[i] )
    for j in range(N_virtual_stars):
       new_dataframe.proper_motions[j] = np.dot( np.linalg.inv(new_dataframe.proper_motions_invcov[j]), new_dataframe.proper_motions[j] )

    #new_dataframe.proper_motions_err = DO NOT NEED
    #new_dataframe.proper_motions_err_corr = DO NOT NEED

    return new_dataframe
