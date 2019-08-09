import numpy
from VectorSphericalHarmonicsVectorized import VectorSphericalHarmonicE, VectorSphericalHarmonicB




def deg_to_rad(degree_vals):
    return numpy.deg2rad (degree_vals)
    
    
    
    
    
def random_aQlm_coeffs ( lmax , lower_bound , upper_bound ):
    negative_coeffs = [ [ random.uniform ( lower_bound , upper_bound ) + 1j * random.uniform ( lower_bound , upper_bound ) for m in range ( -l , 0 ) ] for l in range ( 1 , lmax + 1 ) ]
    
    coeffs = [ [ negative_coeffs[ l-1 ][ m+l ] if m < 0
                 else random.uniform ( lower_bound , upper_bound ) + 1j * 0.0 if m == 0
                 else (-1) ** m * numpy.conj ( negative_coeffs[ l-1 ][ m-l ] )
                 for m in range ( -l , l+1 ) ] for l in range ( 1 , lmax + 1 ) ]
    
    return coeffs

def random_vsh_coeffs ( lmax , lower_bound , upper_bound):
    vsh_E_coeffs = random_aQlm_coeffs ( lmax , lower_bound , upper_bound )
    vsh_B_coeffs = random_aQlm_coeffs ( lmax , lower_bound , upper_bound )
        
    return vsh_E_coeffs , vsh_B_coeffs
    
    
    
    
    
    
def geographic_to_Cartesian (points):
    if len ( points.shape ) == 1:
        nrows = 1
    else:
        nrows = points.shape[0]
        
    new_points = numpy.zeros ( ( len(points) , 3 ))
    
    theta = numpy.pi / 2 - points[... , 1]
    phi = points[... , 0]
    
    new_points[...,0] = numpy.sin ( theta ) * numpy.cos ( phi )
    new_points[...,1] = numpy.sin ( theta ) * numpy.sin ( phi )
    new_points[...,2] = numpy.cos ( theta )
    
    if len ( points.shape ) == 1:
        return new_points[0]
    else:
        return new_points

def tangent_Cartesian_to_geographic (points , dpoints):
    if points.ndim == 1:
        tangent_vector = numpy.zeros ( ( 2 ) , dtype = float)
    else:
        tangent_vector = numpy.zeros ( ( len(points) , 2 ) , dtype = float)
    
    x = points[... , 0]
    y = points[... , 1]
    z = points[... , 2]
    
    dx = dpoints[... , 0]
    dy = dpoints[... , 1]
    dz = dpoints[... , 2]
    
    tangent_vector[... , 0] = dz / ( numpy.sqrt ( 1 - z ** 2 ) )
    tangent_vector[... , 1] = ( x * dy - y * dx ) / ( x ** 2 + y ** 2 )
    
    return tangent_vector

def generate_model ( vsh_E_coeffs , vsh_B_coeffs , positions ):
    lmax = min ( len( vsh_E_coeffs ) , len( vsh_B_coeffs ) )

    positions_Cartesian = geographic_to_Cartesian ( positions )
    
    v_E = numpy.sum ( [ numpy.sum ( [ vsh_E_coeffs[ l-1 ][ m+l ] * VectorSphericalHarmonicE ( l , m , positions_Cartesian ) for m in range ( -l , l+1 ) ] , axis = 0 ) for l in range ( 1 , lmax + 1 ) ] , axis = 0 )
    
    v_B = numpy.sum ( [ numpy.sum ( [ vsh_B_coeffs[ l-1 ][ m+l ] * VectorSphericalHarmonicB ( l , m , positions_Cartesian ) for m in range ( -l , l+1 ) ] , axis = 0 ) for l in range ( 1 , lmax + 1 ) ] , axis = 0 )
    
    numpy.testing.assert_almost_equal(numpy.imag(v_E).sum(), 0.)
    numpy.testing.assert_almost_equal(numpy.imag(v_B).sum(), 0.)
    
    v_Q = numpy.real ( numpy.add ( numpy.array ( v_E ) , numpy.array ( v_B ) ) )
        
    return tangent_Cartesian_to_geographic ( positions_Cartesian , v_Q )
    
    
    
    
    
    
def covariant_matrix ( errors , corr ):
    covariant_matrix = numpy.einsum ( '...i,...j->...ij' , errors , errors )
    
    covariant_matrix[...,0,1] = covariant_matrix[...,1,0] = numpy.multiply ( covariant_matrix[...,1,0] , corr.flatten() )
    return covariant_matrix

def R_values ( pm , pm_err , pm_err_corr , model ):
    covariant_matrices = covariant_matrix ( pm_err , pm_err_corr )
    
    M = pm - model
    
    R_values = numpy.einsum ( '...i,...ij,...j->...' , M , numpy.linalg.inv ( covariant_matrices ) , M ) 
        
    return R_values

def compute_log_likelihood ( R_values ):

    condition = R_values > 1.0e-3
    print(R_values)
    
    modify_R_values = numpy.extract(condition, R_values)
    print(modify_R_values)

    log_likelihood = numpy.log ( ( 1. - numpy.exp ( - modify_R_values ** 2 / 2.) ) / ( modify_R_values ** 2 ) ).sum()
    
    return log_likelihood
    
    
    
    
    
    
    
