import numpy

from data_load import covariant_matrix

import config as c
from utils import *

def generate_scalar_bg ( data , scale=1.0 , err_scale=1.0 ):

    par = {}
    
    for l in range(1,c.Lmax+1):
        for m in range(0, l+1):
            if m==0:
                par[ 'Re[a^E_' + str(l) +'0]' ] = 0.
                par[ 'Re[a^B_' + str(l) +'0]' ] = 0.
            else:
                par[ 'Re[a^E_' + str(l) + str(m) + ']' ] = 0.
                par[ 'Im[a^E_' + str(l) + str(m) + ']' ] = 0.
                par[ 'Re[a^B_' + str(l) + str(m) + ']' ] = 0.
                par[ 'Im[a^B_' + str(l) + str(m) + ']' ] = 0.
                
    par[ 'Re[a^E_10]' ] = scale * 1.0
    
    model_pm = generate_model ( par , data.VSH )
    
    data.proper_motions = model_pm

    data.proper_motions_err[:,0] = err_scale * numpy.reciprocal (numpy.cos(data.positions[:,1]))
    data.proper_motions_err[:,1] = err_scale * numpy.ones ( len(data.proper_motions_err) , dtype=None )
    data.proper_motions_err_corr = numpy.zeros ( data.proper_motions_err_corr.shape, dtype=None )
    data.covariance = covariant_matrix ( data.proper_motions_err , data.proper_motions_err_corr )
    data.covariance_inv = numpy.linalg.inv ( data.covariance )
    
    return data , par

def generate_gr_bg ( data , scale=1. , err_scale=1. ):
    variance = numpy.array( [ 0.0 , 
                              0.3490658503988659 ,
                              0.03490658503988659 ,
                              0.006981317007977318 , 
                              0.0019946620022792336 ] )
    
    par = {}
    
    for l in range(1,c.Lmax+1):
        for m in range(0, l+1):
            if m==0:
                par[ 'Re[a^E_' + str(l) +'0]' ] = scale * ( numpy.random.normal (0.0 , numpy.sqrt(variance[l-1])) )
                par[ 'Re[a^B_' + str(l) +'0]' ] = scale * ( numpy.random.normal (0.0 , numpy.sqrt(variance[l-1])) )
            else:
                par[ 'Re[a^E_' + str(l) + str(m) + ']' ] = scale * ( numpy.random.normal (0.0 , numpy.sqrt(variance[l-1])) )
                par[ 'Im[a^E_' + str(l) + str(m) + ']' ] = scale * ( numpy.random.normal (0.0 , numpy.sqrt(variance[l-1])) )
                par[ 'Re[a^B_' + str(l) + str(m) + ']' ] = scale * ( numpy.random.normal (0.0 , numpy.sqrt(variance[l-1])) )
                par[ 'Im[a^B_' + str(l) + str(m) + ']' ] = scale * ( numpy.random.normal (0.0 , numpy.sqrt(variance[l-1])) )

    model_pm = generate_model ( par , data.VSH )
    
    data.proper_motions = model_pm

    data.proper_motions_err[:,0] = err_scale * numpy.reciprocal (numpy.cos(data.positions[:,1]))
    data.proper_motions_err[:,1] = err_scale * numpy.ones ( len(data.proper_motions_err) , dtype=None )
    data.proper_motions_err_corr = numpy.zeros(data.proper_motions_err_corr.shape, dtype=None)
    data.covariance = covariant_matrix ( data.proper_motions_err , data.proper_motions_err_corr )
    data.covariance_inv = numpy.linalg.inv ( data.covariance )
    
    return data , par
