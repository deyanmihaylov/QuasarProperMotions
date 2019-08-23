import numpy

import config as c
from VectorSphericalHarmonicsVectorized import VectorSphericalHarmonicE, VectorSphericalHarmonicB

from CoordinateTransformations import Cartesian_to_geographic_vector, geographic_to_Cartesian_point

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# def plot_coeffs(coeffs, outfile, Lmax, projection='mollweide', proper_motion_scale=1):
#     """
#     method to plot smooth vector field on sphere

#     INPUTS
#     ------
#     coeffs: dict
#         The dictionary object that contains the named coefficients
#     outfile: str
#         file destination
#     """

#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection=projection)

#     eps = 0.01
#     DEC = numpy.arange(-80, 80+eps, 20) * (numpy.pi/180.)
#     RA = numpy.arange(-170, 170+eps, 20) * (numpy.pi/180.)

#     for ra in RA:
#         for dec in DEC:

#             ax.plot([ra], [dec], 'ro', markersize=1, alpha=0.3)

#             n = geographic_to_Cartesian_point ( numpy.array([ra, dec]) )

#             pm = numpy.zeros(3)
#             for l in numpy.arange(1, Lmax+1):
#                 for m in numpy.arange(0, l+1):
#                     if m==0:
#                         pm += coeffs['Re_a^E_'+str(l)+'0'] * numpy.real(VectorSphericalHarmonicE(l, 0, n))
#                         pm += coeffs['Re_a^B_'+str(l)+'0'] * numpy.real(VectorSphericalHarmonicB(l, 0, n))
#                     else:
#                         pm += 2*coeffs['Re_a^E_'+str(l)+str(m)] * numpy.real(VectorSphericalHarmonicE(l, m, n))
#                         pm -= 2*coeffs['Im_a^E_'+str(l)+str(m)] * numpy.imag(VectorSphericalHarmonicE(l, m, n))
#                         pm += 2*coeffs['Re_a^B_'+str(l)+str(m)] * numpy.real(VectorSphericalHarmonicB(l, m, n))
#                         pm -= 2*coeffs['Im_a^B_'+str(l)+str(m)] * numpy.imag(VectorSphericalHarmonicB(l, m, n))

#             pm_geo = Cartesian_to_geographic_vector(n, proper_motion_scale*pm)

#             ra_vals = [ra-pm_geo[0], ra+pm_geo[0]]
#             dec_vals = [dec-pm_geo[1], dec+pm_geo[1]]

#             ax.plot(ra_vals, dec_vals, 'r-', alpha=0.8)

                    
#     # plot grid lines 
#     plt.grid(True)
    
#     plt.savefig(outfile)

    
    
    
    
    





    
    
    
    
    
    
