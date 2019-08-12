import numpy as np


def CoefficientsFromParams_General(param):

    vsh_E_coeffs = [ [ 
                            param['Re_a^E_'+str(l)+'0']+0*(1j)   
                            if m==0 else if m>0   
                            param['Re_a^E_'+str(l)+str(m)]+(1j)*param['Im_a^E_'+str(l)+str(m)]
                            else
                            ((-1)**(-m)) * ( param['Re_a^E_'+str(l)+str(-m)]-(1j)*param['Im_a^E_'+str(l)+str(-m)] )
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]

    vsh_B_coeffs = [ [ 
                            param['Re_a^B_'+str(l)+'0']+0*(1j)   
                            if m==0 else if m>0  
                            param['Re_a^B_'+str(l)+str(m)]+(1j)*param['Im_a^B_'+str(l)+str(m)]
                            else
                            ((-1)**(-m)) * ( param['Re_a^B_'+str(l)+str(-m)]-(1j)*param['Im_a^B_'+str(l)+str(-m)] )
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]
    
    return vsh_E_coeffs, vsh_B_coeffs

  
  
  
  
  
  
def CoefficientsFromParams_Lmax1(params):
  
    vsh_E_coeffs = [ [param['Re_a^E_1-1']+(1j)*param['Im_a^E_1-1'], 
                      param['Re_a^E_10']+(1j)*0, 
                      param['Re_a^E_11']+(1j)*param['Im_a^E_11'] ]
                    
    vsh_B_coeffs = [ [param['Re_a^B_1-1']+(1j)*param['Im_a^B_1-1'], 
                      param['Re_a^B_10']+(1j)*0, 
                      param['Re_a^B_11']+(1j)*param['Im_a^B_11'] ] 
                    
    return vsh_E_coeffs, vsh_B_coeffs
                    
                    
