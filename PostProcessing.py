# import numpy as np

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt



def Quad(aQlm, names):
    Quad = 0.0
    for i, name in enumerate(names):
        if name[4]=='2':
            Quad += aQlm[i]**2
    return Quad


def post_process_results(posterior_file, basis, L):
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
    print(coeff_names)
    exit()
    a_posteior_samples = np.loadtxt(posterior_file)
    
    Q = np.array([ Quad(sample, coeff_names) for sample in a_posteior_samples])

    plt.hist(Q, bins=np.linspace(0, np.max(Q), 30))

    plt.xlim(0, np.max(Q))

    plt.xlabel("Q [mas^2/yr^2]")
    plt.ylabel("Probability")

    plt.tight_layout()

    output = posterior_file[0:-13] + "Q_histogram.png"
    plt.savefig(output)

    plt.clf()

    Q90 = np.sort(Q)[int(0.9*len(Q))]

    LimitsFile = posterior_file[0:-13] + "Limits.txt"

    with open(LimitsFile, 'w') as text_file:
        text_file.write("Q90: {0}\n".format(Q90))

    C2 = 4. * np.pi**2 / 9.

    if mod_basis:
        M = np.zeros((len(coeff_names),len(coeff_names)))

        for i, name_x in enumerate(coeff_names):
            A_x = 1. if name_x[4]=='2' else 0.

            for j, name_y in enumerate(coeff_names):
                A_y = 1. if name_x[4]=='2' else 0.

                M[i,j] = A_x * A_y

        X = np.einsum("li,lk,kj->ij", L, M, L)

        z = np.random.multivariate_normal(np.zeros(len(coeff_names)), np.diag(np.ones(len(coeff_names))), size=10000)

        y = np.einsum("...i,...ij,...j->...", z, X, z)

        y90 = np.sort(y)[int(0.9*len(y))]

        A90 = np.sqrt(Q90/(C2 * y90))
    else:
        chi_90 = 16.

        A90 = np.sqrt(Q90/(C2*chi_90))

    with open(LimitsFile, 'a') as text_file:
        text_file.write("A90: {0}\n".format(A90))
        
        
        
        
        
        
def plot_vector_field(coeffs, scale=1):
    """
    Smooth vector field plot.
    Cylindrical projection
    """
    eps = 1.0e-3
    ra_vals = np.arange(15,350+eps,15)
    dec_vals = np.arange(-80,80+eps,10)
    
    for ra in ra_vals:
        for dec in dec_vals:
            
            start = geographic_to_Cartesian_point(np.array([[np.pi*ra/180, np.pi*dec/180]]))[0] 
            end = start
            for name in coeffs.keys():
                l, m = int(name[4]), int(name[6:])
                if 'E' in name:
                    end += scale * coeffs[name] * RealVectorSphericalHarmonicE(l, m, start)
                else:
                    end += scale * coeffs[name] * RealVectorSphericalHarmonicB(l, m, start)
            end = Cartesian_to_geographic_point(np.array([end]))[0]
            if end[0]<0:
                end[0]+=2*np.pi
            plt.plot([ra, 180*end[0]/np.pi], [dec, 180*end[1]/np.pi], linestyle='-', color='cyan')
            
    Sag_Astar = (266.41684, -29.00781)
    plt.plot([Sag_Astar[0]], [Sag_Astar[1]], marker='+', color='red', markersize=12)
    
    plt.xlim(0,360)
    plt.ylim(-90,90)
    
    plt.xlabel(r"ra$^{\circ}$")
    plt.ylabel(r"dec$^{\circ}$")
    
    plt.show()
