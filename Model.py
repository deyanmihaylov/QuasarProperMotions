import numpy as np

def generate_model(
    almQ: dict,
    basis: dict,
):
    """
    Generate model of proper motions from a dictionary of almQ
    coefficients and some spherical harmonics basis
    INPUTS
    ------
    almQ: dict
    	Dictionary containing the vector spherical harmonics
        coefficients
    basis: dict
    	Dictionary containing the vector spherical harmonics
    """
    model = np.sum(
        [almQ[key] * basis[key] for key in basis.keys()],
        axis=0,
    )
    return model
