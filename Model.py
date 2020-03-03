import numpy as np

def generate_model(almQ, basis):
    """
    Generate model of PMs from a dictionary of a^Q_lm coefficients and VSH bank
    """
    model = np.sum([almQ[key] * basis[key] for key in almQ.keys()], axis=0)

    return model
