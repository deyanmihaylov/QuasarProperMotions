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

def generate_model ( coeffs , VSH_bank , Lmax):
    v_Q = numpy.sum ( [ numpy.sum ( [ 
                        coeffs['Re_a^' + Q + '_' + str(l) + '0'] * VSH_bank['Re[Y^' + Q + '_' + str(l) + '0]'] 
                        + 2 * numpy.sum ( [ 
                        coeffs['Re_a^' + Q + '_'+str(l)+str(m)] * VSH_bank['Re[Y^' + Q + '_' + str(l) + str(m) + ']'] 
                        - coeffs['Im_a^'+ Q + '_'+str(l)+str(m)] * VSH_bank['Im[Y^' + Q + '_' + str(l) + str(m) + ']'] 
                        for m in range ( 1 , l + 1 ) ] , axis=0 )
                    for l in range ( 1 , Lmax + 1 ) ] , axis=0 )
                for Q in [ 'E' , 'B' ] ] , axis=0 )
        
    return v_Q




# Faster model
def generate_model_fast(par, positions_Cartesian, VSHs_geographic):
    
    lmax = len( vsh_E_coeffs )
    
    # Use precomputed Cartesian positions - Line (A) becomes unecessary
    # Use precomputed VSHs in geo coords - No call to VectorSphericalHarmonicQ() or tangent_Cartesian_to_geographic()
    # use par dict instead of full a^Qlm and deal with re im parts explicitly - Lines (B) and (C) become unecessary
    
    return 1
    
    
    
    
    
    
def covariant_matrix ( errors , corr ):
    covariant_matrix = numpy.einsum ( '...i,...j->...ij' , errors , errors )
    
    covariant_matrix[...,0,1] = covariant_matrix[...,1,0] = numpy.multiply ( covariant_matrix[...,1,0] , corr.flatten() )
    return covariant_matrix

def R_values ( pm , invcovs , model ):
    
    M = pm - model
    
    R_values = numpy.einsum ( '...i,...ij,...j->...' , M , invcovs , M ) 
        
    return R_values

def compute_log_likelihood ( R_values ):

    condition = R_values > 1.0e-3
    print(R_values)
    
    modify_R_values = numpy.extract(condition, R_values)
    print(modify_R_values)

    log_likelihood = numpy.log ( ( 1. - numpy.exp ( - modify_R_values ** 2 / 2.) ) / ( modify_R_values ** 2 ) ).sum()
    
    return log_likelihood
    


def logLfunc(R):
    return numpy.log( ( 1 - numpy.exp(-0.5*R) ) / (0.5*R) )
    
    
    
    
    
    
