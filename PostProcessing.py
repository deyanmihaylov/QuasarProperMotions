import numpy as np




def Quad(aQlm, names):
    Quad = 0.0
    for i, name in enumerate(names):
        if name[5]=='2':
            Quad += aQlm[i]**2
    return Quad


def post_process_results(posterior_file):
    """
    Post process CPNest results

    INPUTS
    ------
    post_process_results: str
        the path to the posterior.dat file produced by CPNest
    """

    with open(posterior_file) as f: 
        coeff_names = f.readline().split()[1:-2]

    a_posteior_samples = np.loadtxt(posterior_file)
    
    Q = np.array([ Quad(sample, coeff_names) for sample in a_posteior_samples])
    
    plt.hist(Q)

    plt.xlabel("Q [mas^2/yr^2]")
    plt.ylabel("Probability")

    plt.tight_layout()

    output = posterior_file[0:-12] + "Q_histogram.png"
    plt.savefig(output)

    plt.clf()

