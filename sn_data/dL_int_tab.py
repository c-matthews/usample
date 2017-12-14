#
#    generate a table of luminosity distances (without c/H0 factor) for 
#    zCMB of each supernova in the JLA sample using colossus package
#
#   
import numpy as np


def Rmm(a, b, func, m, **kwargs):
    """
    Auxiliary function computing tableau entries for Romberg integration
    using recursive relation, but implemented non-recursively
    
    Parameters
    -----------------
    func - python function object
            function to integrate
    a, b - floats
            integration interval            
    m    - integer
            iteration level; accuracy order will be equal to 2(m+1)
            in this implementation there is no need for k on input
            
    kwargs - python dictionary 
            array of keyword arguments to be passed to the integrated function
            
    Returns
    ---------
    
    I(m)   - float
              estimate of the integral using scheme of order 2*m+2
    I(m-1) - float
              estimate of the integral using scheme of order 2*m
    """
    assert(m >= 0)
    
    ba = b - a;
    hk = ba / 2**(np.arange(m+1)) # vector of step sizes

    Rkm = np.zeros((m+1,m+1)) 

    Rkm[0,0] = 0.5 * ba * (func(a, **kwargs) + func(b, **kwargs))
        
    for k in range(1,m+1):
        # first compute R[k,0]
        trapzd_sum = 0.
        for i in range(1, 2**(k-1)+1):
            trapzd_sum += func(a + (2*i-1)*hk[k], **kwargs)
            
        # we can reuse Rkm[k-1,0] but we need to divide it by 2 to account for step decrease 
        Rkm[k,0] = Rkm[k-1,0] * 0.5 + hk[k] * trapzd_sum
        
        # then fill the tableau up to R[k,k]
        for md in range(1,k+1):
            fact = 4.**md
            Rkm[k,md] = (fact * Rkm[k,md-1] - Rkm[k-1,md-1])/(fact - 1)

          
    return Rkm[m,m], Rkm[m,m-1] # return the desired approximation and best one of previous order 

def romberg(func, a, b, rtol = 1.e-4, mmax = 8, verbose = False, **kwargs):
    """
    Romberg integration scheme to evaluate
            int_a^b func(x)dx 
    using recursive relation to produce higher and higher order approximations
    
    Code iterates from m=0, increasing m by 1 on each iteration.
    Each iteration computes the integral using scheme of 2(m+2) order of accuracy 
    Routine checks the difference between approximations of successive orders
    to estimate error and stops when a desired relative accuracy 
    tolerance is reached.
    
    - Andrey Kravtsov, 2017

    Parameters
    --------------------------------
    
    func - python function object
            function to integrate
    a, b - floats
            integration interval
    rtol - float 
            fractional tolerance of the integral estimate
    mmax - integer
            maximum number of iterations to do 
    verbose - logical
            if True print intermediate info for each iteration
    kwargs - python dictionary
             a list of parameters with their keywords to pass to func
               
    Returns
    ---------------------------------
    I    - float
           estimate of the integral for input f, [a,b] and rtol
    err  - float 
           estimated fractional error of the estimated integral

    """
    assert(a < b)
    
    for m in range(1, mmax):
        Rmk_m, Rmk_m1 = Rmm(a, b, func, m, **kwargs)
            
        if Rmk_m == 0:
            Rmk_m = 1.e-300 # guard against division by 0 
            
        etol = 1.2e-16 + rtol*np.abs(Rmk_m)
        err = np.abs(Rmk_m-Rmk_m1)

        if verbose: 
            print("m = %d, integral = %.6e, prev. order = %.6e, frac. error estimate = %.3e"%(m, Rmk_m, Rmk_m1, err/Rmk_m))

        if (m>0) and (np.abs(err) <= etol):
            return Rmk_m, err/Rmk_m
        
    print("!!! Romberg warning: !!!")
    print("!!! maximum of mmax=%d iterations reached, abs(err)=%.3e, > required error rtol = %.3e"%(mmax, np.abs(err/Rmk_m), rtol))
    return Rmk_m, err/Rmk_m
    
def dlum_func(a, **kwargs):
    """
    auxiliary function to compute the integrand of d_c integral
    
    Parameters
    -----------
    a: float
        expansion factor
    kwargs: pythong dictionary of keyword parameters
        contains "Om0" - mean matter density in units of critical at z=0
                 "OmL" - mean vacuum density
                 "Omk" - mean curvature density, Omk = 1 - Om0 - OmL
                 
    Returns
    --------
        float
        value of the integrand for the input a and cosmological parameters
        
    """
    a2i = 1./(a * a) 
    a2Hai = a2i / np.sqrt(kwargs["Om0"]/a**3 + kwargs["OmL"] + kwargs["Omk"] * a2i)
    return a2Hai

def dlum(z, Om0, OmL, rtol=1.e-10):
    """
    Compute luminosity distance (without the c/H_0 factor) for input redshift and Om0, OmL
    
    Parameters
    ----------
    z: float
        redshift
    Om0: float
        mean matter density in units of critical at z=0
    OmL: float
        mean vacuum density
    rtol: float
        fractional error tolerance with which to compute d_L
        
    Returns
    --------
    float: 
        value of d_L for input z and Om0, OmL and rtol without c/H_0 factor
    
    """
    Omk = 1. - Om0 - OmL
    kwargs = {"Om0": Om0, "OmL": OmL, "Omk": Omk}
    zp1 = 1.0 + z
    a = 1. / zp1
    dc = romberg(dlum_func, a, 1., rtol = rtol, mmax = 8, verbose = False, **kwargs)[0]
    
    if np.abs(Omk) < 1.e-15:
        return dc * zp1
    elif Omk > 0:
        sqrtOmk = np.sqrt(Omk)
        return np.sinh(sqrtOmk * dc) / sqrtOmk * zp1
    else:
        sqrtOmk = np.sqrt(-Omk)
        return np.sin(sqrtOmk * dc) / sqrtOmk * zp1

if __name__ == '__main__':
    
    # define parameters of the Om0-OmL grid
    Ng = 60 # number of grid cells in 1d
    Ommin = 0.; Ommax = 1.2; # range of values of Omegas
    dOm = (Ommax - Ommin)/Ng # step
    Om = np.arange(Ommin, Ommax, dOm)
    Oml = np.arange(Ommin, Ommax, dOm)
    # read redshifts of SNe 
    zCMB, zhel = np.loadtxt('jla_lcparams.txt', usecols=(1, 2),  unpack=True)
    Ns = np.size(zCMB)

    intsp = []; intsp2 = []
    inttab = np.zeros((np.size(zCMB),Ng,Ng));
    dtab = np.zeros((Ng,Ng))
    # compute grid of d_L for the specified grid of Om0-OmL
    for iz, zs in enumerate(zCMB):
        for iOl, Old in enumerate(Oml):
            for iOm, Omd in enumerate(Om):
                dtab[iOm,iOl] = dlum(zs, Omd, Old)
        inttab[iz,:,:] = dtab
        print("processed redshift N = %d, z = %.3f"%(iz, zs))

    np.save('dL_int_tab_740x60x60.npy',inttab)
