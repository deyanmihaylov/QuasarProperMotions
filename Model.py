import numpy as np

def generate_model(coeffs, basis):
    """
    Generate model of PMs from a dictionary of a^Q_lm coefficients and VSH bank
    """
    return np.sum([ coeffs[name.replace("Y","a")]*basis[name] for name in basis.keys()], axis=0)
