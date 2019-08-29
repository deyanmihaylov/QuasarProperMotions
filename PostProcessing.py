import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def Quad(aQlm, names):
    Quad = 0.0
    for i, name in enumerate(names):
        if name[4]=='2':
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


    C2 = 4 * np.pi**2 / 9

    chi_90 = 16.

    A90 = np.sqrt(Q90/(C2*chi_90))

    with open(LimitsFile, 'a') as text_file:
        text_file.write("A90: {0}\n".format(A90))
