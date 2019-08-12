import numpy as np

vsh_E_coeffs = [ [ 
                            param['Re_a^E_'+str(l)+'0']+0*(1j)   
                            if m==0 else if m>0   
                            param['Re_a^E_'+str(l)+str(m)]+(1j)*param['Im_a^E_'+str(l)+str(m)]
                            else
                            ((-1)**(-m)) * ( param['Re_a^E_'+str(l)+str(-m)]-(1j)*param['Im_a^E_'+str(l)+str(-m)] )
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]
vsh_B_coeffs = [ [ 
                            par['Re_a^B_'+str(l)+'0']+0*(1j)   
                            if m==0 else if m>0  
                            par['Re_a^B_'+str(l)+str(m)]+(1j)*par['Im_a^B_'+str(l)+str(m)]
                            else
                            ((-1)**(-m)) * ( param['Re_a^B_'+str(l)+str(-m)]-(1j)*param['Im_a^B_'+str(l)+str(-m)] )
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]
