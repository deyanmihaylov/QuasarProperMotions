import numpy as np

def generate_model(almQ, basis):
    """
    Generate model of PMs from a dictionary of a^Q_lm coefficients and VSH bank
    """
    print(almQ[1][0]['E']*basis[1][0]['E'])
    model = np.sum([almQ[l][m][Q] * basis[l][m][Q] for l in almQ.keys() for m in almQ[l].keys() for Q in ['E', 'B']], axis=0)

    return model
