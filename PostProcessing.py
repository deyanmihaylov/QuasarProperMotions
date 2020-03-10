import numpy as np

import Utils as U
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt



# def Quad(aQlm, names):
#     Quad = 0.0
#     for i, name in enumerate(names):
#         if name[4]=='1' and name[2]=='E':
#             Quad += aQlm[i]**2
#     return Quad

def C_l_GR(l):
    """
    TO DO: push this to a file
    """
    
    C_l = [0, 4.386490845, 0.4386490845, 0.08772981690, 0.02506566197,
           0.008952022133, 0.003730009222, 0.001740670970, 0.0008861597667,
           0.0004833598727, 0.0002788614650, 0.0001685426437, 0.0001059410903,
           0.00006886170871, 0.00004607658451, 0.00003162118544,
           0.00002219030558, 0.00001588358715, 0.00001157232778,
           8.566528356e-6, 6.433361216e-6, 4.894948752e-6,
           3.769110539e-6, 2.934107589e-6, 2.307161523e-6,
           1.831080574e-6, 1.465766469e-6, 1.182721909e-6,
           9.614384554e-7, 7.869838970e-7, 6.483674151e-7,
           5.374168414e-7, 4.479979048e-7, 3.754649107e-7,
           3.162699923e-7, 2.676822837e-7, 2.275841278e-7,
           1.943218322e-7, 1.665954245e-7, 1.433765500e-7]

    return C_l[l-1]

def C_l_B(l):
    if l == 1:
        return 8.77298
    else:
        return 0.

def post_process_results(posterior_file, which_basis, Lmax, L, pol, limit):
    """
    Post process CPNest results

    INPUTS
    ------
    post_process_results: str
        the path to the posterior.dat file produced by CPNest
    mod_basis: bool
        whether the modified basis of functions is used

    """
    with open(posterior_file) as f: 
        coeff_names = f.readline().split()[1:-2]
    
    almQ_posterior_samples = np.loadtxt(posterior_file)
    almQ_posterior_samples = almQ_posterior_samples[:, 0:6]

    if pol == "GR":
        diag_of_M = [[0. if C_l_GR(l) == 0 else 1./C_l_GR(l)] * 2*(2*l+1) for l in range(1, Lmax+1)]
    elif pol == "B":
        diag_of_M = [[0. if C_l_B(l) == 0 else 1./C_l_B(l), 0.] * (2*l+1) for l in range(1, Lmax+1)]

    diag_of_M_flat = [coeff for coeffs in diag_of_M for coeff in coeffs]
    M = np.diag(diag_of_M_flat)
    
    Q = np.einsum('...i,ij,...j->...', almQ_posterior_samples, M, almQ_posterior_samples)

    Q_limit = np.percentile(Q, limit)

    if which_basis == "vsh":
        chi_squared_limit = U.chi_squared_limit(len(coeff_names), limit)

        A_limit = np.sqrt(Q_limit/chi_squared_limit)

        print (A_limit)
    elif which_basis == "orthogonal":
        X = np.einsum("li,lk,kj->ij", L, M, L)
        
        generalized_chi_squared_limit = U.generalized_chi_squared_limit(len(coeff_names), X, limit)

        A_limit = np.sqrt(Q_limit/generalized_chi_squared_limit)

        print(A_limit)

        
        
        
        
        
        
# def plot_vector_field(coeffs, scale=1):
#     """
#     Smooth vector field plot.
#     Cylindrical projection
#     """
#     eps = 1.0e-3
#     ra_vals = np.arange(15,350+eps,15)
#     dec_vals = np.arange(-80,80+eps,10)
    
#     for ra in ra_vals:
#         for dec in dec_vals:
            
#             start = geographic_to_Cartesian_point(np.array([[np.pi*ra/180, np.pi*dec/180]]))[0] 
#             end = start
#             for name in coeffs.keys():
#                 l, m = int(name[4]), int(name[6:])
#                 if 'E' in name:
#                     end += scale * coeffs[name] * RealVectorSphericalHarmonicE(l, m, start)
#                 else:
#                     end += scale * coeffs[name] * RealVectorSphericalHarmonicB(l, m, start)
#             end = Cartesian_to_geographic_point(np.array([end]))[0]
#             if end[0]<0:
#                 end[0]+=2*np.pi
#             plt.plot([ra, 180*end[0]/np.pi], [dec, 180*end[1]/np.pi], linestyle='-', color='cyan')
            
#     Sag_Astar = (266.41684, -29.00781)
#     plt.plot([Sag_Astar[0]], [Sag_Astar[1]], marker='+', color='red', markersize=12)
    
#     plt.xlim(0,360)
#     plt.ylim(-90,90)
    
#     plt.xlabel(r"ra$^{\circ}$")
#     plt.ylabel(r"dec$^{\circ}$")
    
#     plt.show()
