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

def NormalisedAssociatedLegendrePolynomial(l, m, x):
    """
    The Normalised Associated Legendre Polynomials
    P^m_l(x), where x = cos(theta).
    
    INPUTS
    ------
    l: int
        Harmonic polar index.
    m: int
        Harmonic azimuthal index.
    x: float
        cos(theta)
        
    RETURNS
    -------
    ans: float
        P^m_l(x)
    """
    legendre = lpmv(m, l, x)
    norm = ( np.sqrt( (2*l+1) / (4.*np.pi) ) 
            * np.sqrt( factorial(l-m)/factorial(l+m) ) ) 
    return norm * legendre

def ScalarSphericalHarmonicY(l, m, n):
    """
    The Scalar Spherical Harmonics, Y^l_m(n)
    
    INPUTS
    ------
    l: int
        Harmonic polar index.
    m: int
        Harmonic azimuthal index.
    n: numpy array shape (3,)
        The Cartesian coordinates of a point on the unit sphere.
        
    RETURNS
    -------
    ans: float
    """
    theta = np.arccos( n[2] / np.sqrt(np.dot(n, n)) )
    phi = np.arctan2(n[1], n[0])
    
    x = np.cos(theta)
    
    return NormalisedAssociatedLegendrePolynomial(l, m, x) * np.exp((1j)*m*phi)

def VectorSphericalHarmonicE(l, m, n):
    """
    The Gradient Vector Spherical Harmonics, (YE)^l_m(n).
    
    INPUTS
    ------
    l: int
        Harmonic polar index.
    m: int
        Harmonic azimuthal index.
    n: numpy array 
        The Cartesian coordinates of a point on the unit sphere.
        Either a single point [shape=(3,)] or several [shape=(3,Npoints)].
        
    RETURNS
    -------
    ans: numpy array shape (3,) or (3,Npoints)
    """
    if n.ndim == 1:
        n = np.array ( [ n ] )

    theta = np.arccos ( np.divide( n[...,2] , np.sqrt ( np.einsum ( "...i,...i->..." , n , n ) ) ) )
    phi = np.arctan2 ( n[...,1] , n[...,0] )
    
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

def VectorSphericalHarmonicB(l, m, n):
    """
    The Curl Vector Spherical Harmonics, (YB)^l_m(n).
    
    INPUTS
    ------
    l: int
        Harmonic polar index.
    m: int
        Harmonic azimuthal index.
    n: numpy array 
        The Cartesian coordinates of a point on the unit sphere.
        Either a single point [shape=(3,)] or several [shape=(3,Npoints)].
        
    RETURNS
    -------
    ans: numpy array shape (3,) or (3,Npoints)
    """
    if n.ndim == 1:
        n = np.array ( [ n ] )
    
    theta = np.arccos ( np.divide( n[...,2] , np.sqrt ( np.einsum ( "...i,...i->..." , n , n ) ) ) )
    phi = np.arctan2 ( n[...,1] , n[...,0] )
    
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

def RealVectorSphericalHarmonicE(l, m, n):
    """
    The Real Gradient Vector Spherical Harmonics, (YE)^l_m(n).
    
    INPUTS
    ------
    l: int
        Harmonic polar index.
    m: int
        Harmonic azimuthal index.
    n: numpy array 
        The Cartesian coordinates of a point on the unit sphere.
        Either a single point [shape=(3,)] or several [shape=(3,Npoints)].
        
    RETURNS
    -------
    ans: numpy array shape (3,) or (3,Npoints)
    """
    if m < 0:
        return np.sqrt(2) * ( (-1)**m ) * np.imag( VectorSphericalHarmonicE(l, np.abs(m), n ) )
    elif m == 0:
        return np.real(VectorSphericalHarmonicE(l, 0, n))
    else:
        return np.sqrt(2) * ( (-1)**m ) * np.real( VectorSphericalHarmonicE(l, m, n ) )

def RealVectorSphericalHarmonicB(l, m, n):
    """
    The Real Curl Vector Spherical Harmonics, (YB)^l_m(n).
    
    INPUTS
    ------
    l: int
        Harmonic polar index.
    m: int
        Harmonic azimuthal index.
    n: numpy array 
        The Cartesian coordinates of a point on the unit sphere.
        Either a single point [shape=(3,)] or several [shape=(3,Npoints)].
        
    RETURNS
    -------
    ans: numpy array shape (3,) or (3,Npoints)
    """
    if m < 0:
        return np.sqrt(2) * ( (-1)**m ) * np.imag( VectorSphericalHarmonicB(l, np.abs(m), n) )
    elif m == 0:
        return np.real(VectorSphericalHarmonicB ( l , 0 , n ))
    else:
        return np.sqrt(2) * ( (-1)**m ) * np.real( VectorSphericalHarmonicB(l, m, n) )
