import numpy as np

def generate_model ( coeffs , VSH_bank ):
    # Generate model of PMs from a^Q_lm coefficients and VSH                                                                      

    v_Q = numpy.sum ( [ numpy.sum ( [
                        coeffs['Re[a^' + Q + '_' + str(l) + '0]'] * VSH_bank['Re[Y^' + Q + '_' + str(l) + '0]']
                        + 2 * numpy.sum ( [
                        coeffs['Re[a^' + Q + '_'+str(l)+str(m) + ']'] * VSH_bank['Re[Y^' + Q + '_' + str(l) + str(m) + ']']
                        - coeffs['Im[a^'+ Q + '_'+str(l)+str(m) + ']'] * VSH_bank['Im[Y^' + Q + '_' + str(l) + str(m) + ']']
                        for m in range ( 1 , l + 1 ) ] , axis=0 )
                    for l in range ( 1 , c.Lmax + 1 ) ] , axis=0 )
                for Q in [ 'E' , 'B' ] ] , axis=0 )

    return v_Q
