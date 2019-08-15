import numpy

def geographic_to_Cartesian_point ( points ):
    if len ( points.shape ) == 1:
        nrows = 1
    else:
        nrows = points.shape[0]
        
    new_points = numpy.zeros ( ( len ( points ) , 3 ))
    
    theta = numpy.pi / 2 - points[... , 1]
    phi = points[... , 0]
    
    new_points[...,0] = numpy.sin ( theta ) * numpy.cos ( phi )
    new_points[...,1] = numpy.sin ( theta ) * numpy.sin ( phi )
    new_points[...,2] = numpy.cos ( theta )
    
    if len ( points.shape ) == 1:
        return new_points[0]
    else:
        return new_points
    
def Cartesian_to_geographic_vector (points , dpoints):
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
    
    tangent_vector[... , 0] = ( x * dy - y * dx ) / ( x ** 2 + y ** 2 )
    tangent_vector[... , 1] = dz / ( numpy.sqrt ( 1 - z ** 2 ) )
    
    return tangent_vector