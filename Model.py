import numpy as np

def generate_model(coeffs, VSH_bank):
    """
    Generate model of PMs from a dictionary of a^Q_lm coefficients and VSH bank                                                                      
    """
    return np.sum([ coeffs[name.replace("Y","a")]*VSH_bank[name] for name in VSH_bank.keys()])
