import numpy as np

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
    
    
def Cartesian_to_geographic_point(points):
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
    new_points = np.zeros((len(points), 2))
    
    theta = np.arccos( points[..., 2] / np.linalg.norm(points, axis=1) )
    phi = np.arctan2( points[..., 1], points[..., 0] )
    
    new_points[...,0] = phi
    new_points[...,1] = np.pi/2 - theta
    
    if len(points.shape) == 1:
        return new_points[0]
    else:
        return new_points    
    
    
    
    
    
    
    
    
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
