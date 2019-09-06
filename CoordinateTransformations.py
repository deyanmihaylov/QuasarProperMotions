import numpy as np

def geographic_to_Cartesian_point(points):
    """
    Transform points on the unit sphere from geographic 
    coords (ra, dec) to Cartesian coords (x,y,z)
    
    INPUTS
    ------
    points: numpy array
        The coords ra and dec in radians.
        Either a single point [shape=(2,)], or
        a list of points [shape=(Npoints,2)].
    
    RETURNS
    -------
    new_points: numpy array
        The coord (x,y,z) with x^2+y^2+z^=1.
        Either a single point [shape=(3,)], or
        a list of points [shape=(Npoints,3)].
    """
    
    # I don't think any of these lines are needed anymore?
    #if len ( points.shape ) == 1:
    #    nrows = 1
    #else:
    #    nrows = points.shape[0]
        
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
    
def Cartesian_to_geographic_point ( points ):
    if len ( points.shape ) == 1:
        nrows = 1
    else:
        nrows = points.shape[0]
        
    new_points = np.zeros ( ( len ( points ) , 2 ))
    
    theta = np.arccos( points[..., 2] / np.linalg.norm(points, axis=1) )
    phi = np.arctan2( points[..., 1], points[..., 0] )
    
    new_points[...,0] = phi
    new_points[...,1] = np.pi / 2 - theta
    
    if len ( points.shape ) == 1:
        return new_points[0]
    else:
        return new_points    
    
    
    
    
    
    
    
def Cartesian_to_geographic_vector ( points , dpoints ):
    if points.ndim == 1:
        tangent_vector = np.zeros ( ( 2 ) , dtype = float)
    else:
        tangent_vector = np.zeros ( ( len(points) , 2 ) , dtype = float)
    
    x = points[... , 0]
    y = points[... , 1]
    z = points[... , 2]
    
    dx = dpoints[... , 0]
    dy = dpoints[... , 1]
    dz = dpoints[... , 2]
    
    tangent_vector[... , 0] = ( x * dy - y * dx ) / ( x ** 2 + y ** 2 )
    tangent_vector[... , 1] = dz / ( np.sqrt ( 1 - z ** 2 ) )
    
    return tangent_vector

def geographic_to_Cartesian_vector ( points , dpoints ):
    tangent_vector = np.zeros ( ( len(points) , 3 ) , dtype = float)
    
    theta = np.pi / 2 - points[... , 1]
    phi = points[... , 0]
    
    dtheta = - dpoints[... , 1]
    dphi = dpoints[... , 0]
    
    tangent_vector[...,0] = np.cos (theta) * np.cos (phi) * dtheta - np.sin (theta) * np.sin (phi) * dphi
    tangent_vector[...,1] = np.cos (theta) * np.sin (phi) * dtheta + np.sin (theta) * np.cos (phi) * dphi
    tangent_vector[...,2] = - np.sin (theta) * dtheta
    
    return tangent_vector
