import numpy as np


def CoefficientsFromParams_General(param):

    N = len(param.keys())
    Lmax = -1 + int(np.sqrt( 1 + N//2 ))

    vsh_E_coeffs = [ [
                            param['Re_a^E_'+str(l)+'0']+0*(1j) 
                            if m==0 else 
                            param['Re_a^E_'+str(l)+str(m)]+(1j)*param['Im_a^E_'+str(l)+str(m)] 
                            if m>0 else
                            ((-1)**(-m)) * ( param['Re_a^E_'+str(l)+str(-m)]-(1j)*param['Im_a^E_'+str(l)+str(-m)] )  
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]

    vsh_B_coeffs = [ [ 
                            param['Re_a^B_'+str(l)+'0']+0*(1j)   
                            if m==0 else
                            param['Re_a^B_'+str(l)+str(m)]+(1j)*param['Im_a^B_'+str(l)+str(m)]
                            if m>0 else
                            ((-1)**(-m)) * ( param['Re_a^B_'+str(l)+str(-m)]-(1j)*param['Im_a^B_'+str(l)+str(-m)] )
                       for m in np.arange(-l, l+1)] for l in np.arange(1, Lmax+1)]
    
    return vsh_E_coeffs, vsh_B_coeffs

  
  
  
  
  
  
def CoefficientsFromParams_Lmax1(param):
  
    vsh_E_coeffs = [ [-param['Re_a^E_11']+(1j)*param['Im_a^E_11'], 
                      param['Re_a^E_10']+(1j)*0, 
                      param['Re_a^E_11']+(1j)*param['Im_a^E_11'] ] ]
                    
    vsh_B_coeffs = [ [-param['Re_a^B_11']+(1j)*param['Im_a^B_11'], 
                      param['Re_a^B_10']+(1j)*0, 
                      param['Re_a^B_11']+(1j)*param['Im_a^B_11'] ] ]
                    
    return vsh_E_coeffs, vsh_B_coeffs




def CoefficientsFromParams_Lmax4(param):

    vsh_E_coeffs = [

                     [-param['Re_a^E_11']+(1j)*param['Im_a^E_11'],
                      param['Re_a^E_10']+(1j)*0,
                      param['Re_a^E_11']+(1j)*param['Im_a^E_11'] ] 

                     ,

                     [param['Re_a^E_22']-(1j)*param['Im_a^E_22'],
                      -param['Re_a^E_21']+(1j)*param['Im_a^E_21'],
                      param['Re_a^E_20']+(1j)*0,
                      param['Re_a^E_21']+(1j)*param['Im_a^E_21'],
                      param['Re_a^E_21']+(1j)*param['Im_a^E_21'] ]

                     ,

                     [-param['Re_a^E_33']+(1j)*param['Im_a^E_33'],
                      param['Re_a^E_32']-(1j)*param['Im_a^E_32'],
                      -param['Re_a^E_31']+(1j)*param['Im_a^E_31'],
                      param['Re_a^E_30']+(1j)*0,
                      param['Re_a^E_31']+(1j)*param['Im_a^E_31'],
                      param['Re_a^E_32']+(1j)*param['Im_a^E_32'],
                      param['Re_a^E_33']+(1j)*param['Im_a^E_33'] ]

                     ,

                     [param['Re_a^E_44']-(1j)*param['Im_a^E_44'],
                      -param['Re_a^E_43']+(1j)*param['Im_a^E_43'],
                      param['Re_a^E_42']-(1j)*param['Im_a^E_42'],
                      -param['Re_a^E_41']+(1j)*param['Im_a^E_41'],
                      param['Re_a^E_40']+(1j)*0,
                      param['Re_a^E_41']+(1j)*param['Im_a^E_41'],
                      param['Re_a^E_42']+(1j)*param['Im_a^E_42'],
                      param['Re_a^E_43']+(1j)*param['Im_a^E_43'],
                      param['Re_a^E_44']+(1j)*param['Im_a^E_44'] ]

                   ]


    vsh_B_coeffs = [

                     [-param['Re_a^B_11']+(1j)*param['Im_a^B_11'],
                      param['Re_a^B_10']+(1j)*0,
                      param['Re_a^B_11']+(1j)*param['Im_a^B_11'] ]

                     ,

                     [param['Re_a^B_22']-(1j)*param['Im_a^B_22'],
                      -param['Re_a^B_21']+(1j)*param['Im_a^B_21'],
                      param['Re_a^B_20']+(1j)*0,
                      param['Re_a^B_21']+(1j)*param['Im_a^B_21'],
                      param['Re_a^B_21']+(1j)*param['Im_a^B_21'] ]

                     ,

                     [-param['Re_a^B_33']+(1j)*param['Im_a^B_33'],
                      param['Re_a^B_32']-(1j)*param['Im_a^B_32'],
                      -param['Re_a^B_31']+(1j)*param['Im_a^B_31'],
                      param['Re_a^B_30']+(1j)*0,
                      param['Re_a^B_31']+(1j)*param['Im_a^B_31'],
                      param['Re_a^B_32']+(1j)*param['Im_a^B_32'],
                      param['Re_a^B_33']+(1j)*param['Im_a^B_33'] ]

                     ,

                     [param['Re_a^B_44']-(1j)*param['Im_a^B_44'],
                      -param['Re_a^B_43']+(1j)*param['Im_a^B_43'],
                      param['Re_a^B_42']-(1j)*param['Im_a^B_42'],
                      -param['Re_a^B_41']+(1j)*param['Im_a^B_41'],
                      param['Re_a^B_40']+(1j)*0,
                      param['Re_a^B_41']+(1j)*param['Im_a^B_41'],
                      param['Re_a^B_42']+(1j)*param['Im_a^B_42'],
                      param['Re_a^B_43']+(1j)*param['Im_a^B_43'],
                      param['Re_a^B_44']+(1j)*param['Im_a^B_44'] ]

                   ]
                     
    return vsh_E_coeffs, vsh_B_coeffs                    
                    
