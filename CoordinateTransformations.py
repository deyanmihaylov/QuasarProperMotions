import numpy as np
from scipy.optimize import root

def point_transform(input_points: np.ndarray,
                    input_coordinates: str,
                    output_coordinates: str):

    if input_coordinates == "geographic":
        if output_coordinates == "Cartesian":
            output_points = geographic_to_Cartesian_point(input_points)
        elif output_coordinates == "Mollweide":
            output_points = geographic_to_Mollweide_point(input_points)
        else:
            output_points = input_points.copy()
    elif input_coordinates == "Cartesian":
        if output_coordinates == "geographic":
            output_points = Cartesian_to_geographic_point(input_points)
        else:
            output_points = input_points.copy()
    elif input_coordinates == "Mollweide":
        if output_coordinates == "geographic":
            output_points = Mollweide_to_geographic_point(input_points)
        else:
            output_points = input_points.copy()
    else:
        output_points = input_points.copy()

    return output_points

def geographic_to_Cartesian_point(points):
    """
    Transform points on the unit sphere from geographic 
    coords (ra,dec) to Cartesian coords (x,y,z)
    
    INPUTS
    ------
    points: numpy array
        The coords ra and dec in radians.
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    
    RETURNS
    -------
    new_points: numpy array
        The coords (x,y,z) with x^2+y^2+z^=1.
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints,3)].
    """  
    new_points = np.zeros((len(points), 3))
    
    theta = np.pi/2 - points[... , 1]
    phi = points[... , 0]
    
    new_points[...,0] = np.sin(theta) * np.cos(phi)
    new_points[...,1] = np.sin(theta) * np.sin(phi)
    new_points[...,2] = np.cos(theta)
    
    if len(points.shape) == 1:
        return new_points[0]
    else:
        return new_points
    
def Cartesian_to_geographic_point(points_Cartesian):
    """
    Transform points on the unit sphere from Cartesian 
    coords (x,y,z) to geographic coords (ra,dec)
    
    INPUTS
    ------
    points: numpy array
        The Cartesian coords (x,y,z).
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints,3)].
    
    RETURNS
    -------
    new_points: numpy array
        The coords (ra,dec) in radians.
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,3)].
    """
    shape_geographic = list(points_Cartesian.shape)
    shape_geographic[-1] = 2

    points_geographic = np.zeros(shape=shape_geographic)
    
    theta = np.arccos(points_Cartesian[..., 2] / np.linalg.norm(points_Cartesian, axis=-1))
    phi = np.arctan2(points_Cartesian[..., 1], points_Cartesian[..., 0])
    phi[phi<0] += 2*np.pi
    
    points_geographic[...,0] = phi
    points_geographic[...,1] = np.pi/2 - theta
    
    return points_geographic

def geographic_to_Mollweide_point(points_geographic):
    """
    Transform points on the unit sphere from geographic coordinates (ra,dec)
    to Mollweide projection coordiantes (x,y).
    
    INPUTS
    ------
    points_geographic: numpy array
        The geographic coords (ra,dec).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    
    RETURNS
    -------
    points_Mollweide: numpy array
        The Mollweide projection coords (x,y).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    """
    final_shape_Mollweide = list(points_geographic.shape)

    points_geographic = points_geographic.reshape(-1, points_geographic.shape[-1])
        
    points_Mollweide = np.zeros(shape=points_geographic.shape,
                                dtype=points_geographic.dtype)

    alpha_tol = 1.e-6

    def alpha_eq(x):
        return np.where(np.pi/2 - np.abs(points_geographic[...,1]) < alpha_tol, points_geographic[...,1], 2 * x + np.sin(2 * x) - np.pi * np.sin(points_geographic[...,1]))

    alpha = root(fun=alpha_eq, x0=points_geographic[...,1], method='krylov', tol=1.e-10)

    points_Mollweide[...,0] = 2 * np.sqrt(2) * (points_geographic[...,0] - np.pi) * np.cos(alpha.x) / np.pi
    points_Mollweide[...,1] = np.sqrt(2) * np.sin(alpha.x)

    points_Mollweide = points_Mollweide.reshape(final_shape_Mollweide)

    return points_Mollweide

def Mollweide_to_geographic_point(points_Mollweide):
    """
    Transform points on the unit sphere from Mollweide projection coordiantes (x,y)
    to geographic coordinates (ra,dec).
    
    INPUTS
    ------
    points_Mollweide: numpy array
        The Mollweide projection coords (x,y).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    
    RETURNS
    -------
    points_geographic: numpy array
        The geographic coords (ra,dec).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    """
    if points_Mollweide.ndim == 1:
        points_geographic = np.zeros((2), dtype=points_Mollweide.dtype)
    else:
        points_geographic = np.zeros((points_Mollweide.shape[0], 2), dtype=points_Mollweide.dtype)

    alpha = np.arcsin(points_Mollweide[...,1]/np.sqrt(2))

    points_geographic[...,0] = np.pi + (np.pi * points_Mollweide[...,0]) / (2*np.sqrt(2)*np.cos(alpha))
    points_geographic[...,1] = np.arcsin((2*alpha + np.sin(2*alpha))/np.pi)

    return points_geographic

def Cartesian_to_geographic_vector(points, dpoints):
    """
    Transform vectors in the tangent plane of the unit sphere from
    Cartesian coords (d_x,d_y,d_z) to geographic coords (d_ra,d_dec).
    
    INPUTS
    ------
    points: numpy array
        The Cartesian coords (x,y,z).
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints,3)].
    dpoints: numpy array
        The Cartesian coords (d_x,d_y,d_z) which
        satisfy x*d_x+y*d_y+z*d_z = 0.
        Either a single point or many with shape 
        matching points.
    
    RETURNS
    -------
    tangent_vector: numpy array
        The coords (d_ra,d_dec) in radians.
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    """ 
    if points.ndim == 1:
        tangent_vector = np.zeros((2), dtype=dpoints.dtype)
    else:
        tangent_vector = np.zeros((len(points), 2), dtype=dpoints.dtype)
    
    x = points[... , 0]
    y = points[... , 1]
    z = points[... , 2]
    
    dx = dpoints[... , 0]
    dy = dpoints[... , 1]
    dz = dpoints[... , 2]
    
    tangent_vector[... , 0] = ( x*dy-y*dx ) / ( x**2+y**2 )
    tangent_vector[... , 1] = dz / ( np.sqrt( 1-z**2 ) )
    
    return tangent_vector

def geographic_to_Cartesian_vector(points, dpoints):
    """
    Transform vectors in the tangent plane of the unit sphere from
    geographic coords (d_ra,d_dec) to Cartesian coords (d_x,d_y,d_z).
    
    INPUTS
    ------
    points: numpy array
        The geographic coords (ra,dec).
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    dpoints: numpy array
        The geographic coords (d_ra,d_dec).
        Either a single point or many with shape 
        matching points.
    
    RETURNS
    -------
    tangent_vector: numpy array
        The coords (d_x,d_y,d_z).
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints,3)].
    """
    if points.ndim == 1:
        tangent_vector = np.zeros((3), dtype=dpoints.dtype)
    else:
        tangent_vector = np.zeros((len(points), 3), dtype=dpoints.dtype)
    
    theta = np.pi / 2 - points[... , 1]
    phi = points[... , 0]
    
    dtheta = - dpoints[... , 1]
    dphi = dpoints[... , 0]
    
    tangent_vector[...,0] = np.cos(theta) * np.cos(phi) * dtheta - np.sin(theta) * np.sin(phi) * dphi
    tangent_vector[...,1] = np.cos(theta) * np.sin(phi) * dtheta + np.sin(theta) * np.cos(phi) * dphi
    tangent_vector[...,2] = - np.sin(theta) * dtheta
    
    return tangent_vector
