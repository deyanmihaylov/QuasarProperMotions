import numpy

import config as c
from VectorSphericalHarmonicsVectorized import VectorSphericalHarmonicE, VectorSphericalHarmonicB

from CoordinateTransformations import Cartesian_to_geographic_vector, geographic_to_Cartesian_point

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_coeffs(coeffs, outfile, Lmax, projection='mollweide', proper_motion_scale=1):
    """
    method to plot smooth vector field on sphere

    INPUTS
    ------
    coeffs: dict
        The dictionary object that contains the named coefficients
    outfile: str
        file destination
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=projection)

    eps = 0.01
    DEC = numpy.arange(-80, 80+eps, 20) * (numpy.pi/180.)
    RA = numpy.arange(-170, 170+eps, 20) * (numpy.pi/180.)

    for ra in RA:
        for dec in DEC:

            ax.plot([ra], [dec], 'ro', markersize=1, alpha=0.3)

            n = geographic_to_Cartesian_point ( numpy.array([ra, dec]) )

            pm = numpy.zeros(3)
            for l in numpy.arange(1, Lmax+1):
                for m in numpy.arange(0, l+1):
                    if m==0:
                        pm += coeffs['Re_a^E_'+str(l)+'0'] * numpy.real(VectorSphericalHarmonicE(l, 0, n))
                        pm += coeffs['Re_a^B_'+str(l)+'0'] * numpy.real(VectorSphericalHarmonicB(l, 0, n))
                    else:
                        pm += 2*coeffs['Re_a^E_'+str(l)+str(m)] * numpy.real(VectorSphericalHarmonicE(l, m, n))
                        pm -= 2*coeffs['Im_a^E_'+str(l)+str(m)] * numpy.imag(VectorSphericalHarmonicE(l, m, n))
                        pm += 2*coeffs['Re_a^B_'+str(l)+str(m)] * numpy.real(VectorSphericalHarmonicB(l, m, n))
                        pm -= 2*coeffs['Im_a^B_'+str(l)+str(m)] * numpy.imag(VectorSphericalHarmonicB(l, m, n))

            pm_geo = Cartesian_to_geographic_vector(n, proper_motion_scale*pm)

            ra_vals = [ra-pm_geo[0], ra+pm_geo[0]]
            dec_vals = [dec-pm_geo[1], dec+pm_geo[1]]

            ax.plot(ra_vals, dec_vals, 'r-', alpha=0.8)

                    
    # plot grid lines 
    plt.grid(True)
    
    plt.savefig(outfile)

    
    
    
    
    
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
    
    tangent_vector[... , 1] = dz / ( numpy.sqrt ( 1 - z ** 2 ) )
    tangent_vector[... , 0] = ( x * dy - y * dx ) / ( x ** 2 + y ** 2 )
    
    return tangent_vector

def generate_model ( coeffs , VSH_bank ):
    # Generate model of PMs from a^Q_lm coefficients and VSH
    
    v_Q = numpy.sum ( [ numpy.sum ( [ 
                        coeffs['Re[a^' + Q + '_' + str(l) + '0]'] * VSH_bank['Re[Y^' + Q + '_' + str(l) + '0]'] 
                        + 2 * numpy.sum ( [ 
                        coeffs['Re[a^' + Q + '_'+str(l)+str(m) + ']'] * VSH_bank['Re[Y^' + Q + '_' + str(l) + str(m) + ']'] 
                        - coeffs['Im[a^'+ Q + '_'+str(l)+str(m) + ']'] * VSH_bank['Im[Y^' + Q + '_' + str(l) + str(m) + ']'] 
                        for m in range ( 1 , l + 1 ) ] , axis=0 )
                    for l in range ( 1 , c.Lmax + 1 ) ] , axis=0 )
                for Q in [ 'E' , 'B' ] ] , axis=0 )
    
    return v_Q

def covariant_matrix ( errors , corr ):
    # Compute the covariant matrix from errors and correlation

    covariant_matrix = numpy.einsum ( '...i,...j->...ij' , errors , errors )
    
    covariant_matrix[...,0,1] = covariant_matrix[...,1,0] = numpy.multiply ( covariant_matrix[...,1,0] , corr.flatten() )
    return covariant_matrix

def R_values ( pm , invcovs , model ):
    # Compute R values from data, model, and the inverse of the covariant matrix

    M = pm - model
    
    R_values = numpy.sqrt ( numpy.einsum ( '...i,...ij,...j->...' , M , invcovs , M ) )
        
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
    
    
    
    
    
    
