######################################
### The Vector Spherical Harmonics ###
######################################

# These functions are normalised such that
# < (Y^l_m)^E | (Y^l_m)^E > = < (Y^l_m)^B | (Y^l_m)^B > = 1
#
# They agree with the following Mathematica functions
#
#  YE[l_, m_, th_, ph_] := Block[{eth, eph, pol, az, DYDTH, DYDPH},
#  eth = {Cos[th] Cos[ph], Cos[th] Sin[ph], -Sin[th]};
#  eph = {-Sin[ph], Cos[ph], 0};
#  DYDTH = D[SphericalHarmonicY[l, m, pol, ph], pol] /. pol -> th;
#  DYDPH = D[SphericalHarmonicY[l, m, th, az], az] /. az -> ph;
#  Return[(DYDTH eth + DYDPH eph/Sin[th])/Sqrt[l (l + 1)]];]
#
#  YB[l_, m_, th_, ph_] := Block[{eth, eph, pol, az, DYDTH, DYDPH, grad, n},
#  eth = {Cos[th] Cos[ph], Cos[th] Sin[ph], -Sin[th]};
#  eph = {-Sin[ph], Cos[ph], 0};
#  DYDTH = D[SphericalHarmonicY[l, m, pol, ph], pol] /. pol -> th;
#  DYDPH = D[SphericalHarmonicY[l, m, th, az], az] /. az -> ph;
#  n = {Sin[th] Cos[ph], Sin[th] Sin[ph], Cos[th]};
#  grad = (DYDTH eth + DYDPH eph/Sin[th]);
#  Return[Cross[n,grad]/Sqrt[l (l + 1)]];]



import numpy as np
from scipy.special import lpmv
from math import factorial



# The Normalised Associated Legendre Polynomials, P^m_l(x), where x = cos(theta)
def NormalisedAssociatedLegendrePolynomial ( l , m , x ):

    legendre = lpmv ( m , l , x )
    norm = ( np.sqrt ( ( 2 * l + 1) / ( 4. * np.pi ) ) 
            * np.sqrt ( factorial ( l - m ) / factorial ( l + m ) ) ) 
    
    return norm * legendre



# The Scalar Spherical Harmonics, Y^l_m(n)
# The Variable n Is A Unit Three-Vector (i.e. a point on the sphere)
def ScalarSphericalHarmonicY ( l , m , n ):
    
    # The Spherical Polar Angles Of The Position Vector n
    theta = np.arccos ( n[2] / np.sqrt ( np.dot ( n , n ) ) )
    phi = np.arctan2 ( n[1] , n[0] )
    
    #  Useful to Define x = cos ( theta )
    x = np.cos ( theta )
    
    return NormalisedAssociatedLegendrePolynomial ( l , m , x ) * np.exp ( (1j) * m * phi )


# The Gradient Vector Spherical Harmonics, ((Y^l_m)^E(n))_i
# The Variable n Is A Unit Three-Vector (i.e. a point on the sphere)
def VectorSphericalHarmonicE ( l , m , n ):
    if n.ndim == 1:
        n = np.array ( [ n ] )

    # The Spherical Polar Angles Of The Position Vector n
    theta = np.arccos ( np.divide( n[...,2] , np.sqrt ( np.einsum ( "...i,...i->..." , n , n ) ) ) )
    phi = np.arctan2 ( n[...,1] , n[...,0] )
    
    #  Useful to Define x = cos ( theta )
    x = np.cos ( theta )
    
    # The Coordinate Basis Vectors Associated With The Spherical Polar Angles
    e_theta = np.array ( [ x * np.cos ( phi ) , x * np.sin ( phi ) , -np.sqrt ( 1. - x * x ) ] )
    e_phi = np.array ( [ -np.sin ( phi ) , np.cos ( phi ) , np.zeros ( len ( phi ) ) ] )
    
    # The Derivative Of The Spherical Harmonic Function Y^l_m WRT To Theta
    if m == 0:
        DY_Dtheta = -( -np.sqrt ( l * ( l + 1 ) ) * NormalisedAssociatedLegendrePolynomial ( l , 1 , x ) 
                     * np.exp ( (1j) * m * phi ) )
    elif m == l:
        DY_Dtheta = ( -np.sqrt ( l / 2 ) * NormalisedAssociatedLegendrePolynomial ( l , l-1 , x ) 
                     * np.exp ( (1j) * m * phi ) )
    elif m == -l:
        DY_Dtheta = np.power ( -1. , m) * ( -np.sqrt ( l / 2 ) * NormalisedAssociatedLegendrePolynomial ( l , l-1 , x ) 
                     * np.exp ( (1j) * m * phi ) )
    else:
        c1 = 0.5 * np.sqrt ( ( l + m ) * ( l - m + 1 ) )
        c2 = 0.5 * np.sqrt ( ( l + m + 1 ) * ( l - m ) )
        DY_Dtheta = -( c1 * NormalisedAssociatedLegendrePolynomial ( l , m-1 , x ) - 
                     c2 * NormalisedAssociatedLegendrePolynomial ( l , m+1 , x ) ) * np.exp ( (1j) * m * phi )
    
    # The Derivative Of The Spherical Harmonic Function Y^l_m WRT To Phi Divided By Sin(Theta)
    c1 = np.sqrt ( ( 2*l+1 ) * ( l-m+1 ) * ( l-m+2 ) / ( 2*l+3 ) )
    c2 = np.sqrt ( ( 2*l+1 ) * ( l+m+1 ) * ( l+m+2 ) / ( 2*l+3 ) )
    DY_Dphi_OVER_sinTheta = -( ( 1. / 2. ) * ( c1 * NormalisedAssociatedLegendrePolynomial ( l+1 , m-1 , x )
                                                    + c2 * NormalisedAssociatedLegendrePolynomial ( l+1 , m+1 , x ) )
                                                    * (1j) * np.exp ( (1j) * m * phi ) )
    
    v_E = ( np.einsum ( "i,ji->ij" , DY_Dtheta , e_theta ) + np.einsum ( "i,ji->ij" , DY_Dphi_OVER_sinTheta , e_phi ) ) / np.sqrt ( l * ( l + 1 ) )
    
    if v_E.shape[0] == 1:
        v_E = v_E.flatten()
    
    return v_E



# The Curl Vector Spherical Harmonics, ((Y^l_m)^B(n))_i
# The Variable n Is A Unit Three-Vector (i.e. a point on the sphere)
def VectorSphericalHarmonicB ( l , m , n ):
    if n.ndim == 1:
        n = np.array ( [ n ] )
    
    # The Spherical Polar Angles Of The Position Vector n
    theta = np.arccos ( np.divide( n[...,2] , np.sqrt ( np.einsum ( "...i,...i->..." , n , n ) ) ) )
    phi = np.arctan2 ( n[...,1] , n[...,0] )
    
    #  Useful to Define x = cos ( theta )
    x = np.cos ( theta )
    
    # The Coordinate Basis Vectors Associated With The Spherical Polar Angles
    e_theta = np.array ( [ x * np.cos ( phi ) , x * np.sin ( phi ) , -np.sqrt ( 1. - x * x ) ] )
    e_phi = np.array ( [ -np.sin ( phi ) , np.cos ( phi ) , np.zeros ( len ( phi ) ) ] )
    
    # The Derivative Of The Spherical Harmonic Function Y^l_m WRT To Theta
    if m == 0:
        DY_Dtheta = -( -np.sqrt ( l * ( l + 1 ) ) * NormalisedAssociatedLegendrePolynomial ( l , 1 , x ) 
                     * np.exp ( (1j) * m * phi ) )
    elif m == l:
        DY_Dtheta = ( -np.sqrt ( l / 2 ) * NormalisedAssociatedLegendrePolynomial ( l , l-1 , x ) 
                     * np.exp ( (1j) * m * phi ) )
    elif m == -l:
        DY_Dtheta = np.power ( -1. , m) * ( -np.sqrt ( l / 2 ) * NormalisedAssociatedLegendrePolynomial ( l , l-1 , x ) 
                     * np.exp ( (1j) * m * phi ) )
    else:
        c1 = 0.5 * np.sqrt ( ( l + m ) * ( l - m + 1 ) )
        c2 = 0.5 * np.sqrt ( ( l + m + 1 ) * ( l - m ) )
        DY_Dtheta = -( c1 * NormalisedAssociatedLegendrePolynomial ( l , m-1 , x ) - 
                     c2 * NormalisedAssociatedLegendrePolynomial ( l , m+1 , x ) ) * np.exp ( (1j) * m * phi )
    
    # The Derivative Of The Spherical Harmonic Function Y^l_m WRT To Phi Divided By Sin(Theta)
    c1 = np.sqrt ( ( 2*l+1 ) * ( l-m+1 ) * ( l-m+2 ) / ( 2*l+3 ) )
    c2 = np.sqrt ( ( 2*l+1 ) * ( l+m+1 ) * ( l+m+2 ) / ( 2*l+3 ) )
    DY_Dphi_OVER_sinTheta = -( ( 1. / 2. ) * ( c1 * NormalisedAssociatedLegendrePolynomial ( l+1 , m-1 , x )
                                                    + c2 * NormalisedAssociatedLegendrePolynomial ( l+1 , m+1 , x ) )
                                                    * (1j) * np.exp ( (1j) * m * phi ) )
    
    v_B = -( np.einsum ( "i,ji->ij" , DY_Dphi_OVER_sinTheta , e_theta ) - np.einsum ( "i,ji->ij" , DY_Dtheta , e_phi ) ) / np.sqrt ( l * ( l + 1 ) )
    
    if v_B.shape[0] == 1:
        v_B = v_B.flatten()
    
    return v_B

def RealVectorSphericalHarmonicE ( l , m , n ):
    if m < 0:
        return np.sqrt(2) * ( (-1) ** np.abs(m) ) * np.imag ( VectorSphericalHarmonicE ( l , np.abs(m) , n ) )
    elif m == 0:
        return VectorSphericalHarmonicE ( l , 0 , n )
    else:
        return np.sqrt(2) * ( (-1) ** np.abs(m) ) * np.real ( VectorSphericalHarmonicE ( l , np.abs(m) , n ) )

def RealVectorSphericalHarmonicB ( l , m , n ):
    if m < 0:
        return np.sqrt(2) * ( (-1) ** np.abs(m) ) * np.imag ( VectorSphericalHarmonicB ( l , np.abs(m) , n ) )
    elif m == 0:
        return VectorSphericalHarmonicB ( l , 0 , n )
    else:
        return np.sqrt(2) * ( (-1) ** np.abs(m) ) * np.real ( VectorSphericalHarmonicB ( l , np.abs(m) , n ) )
