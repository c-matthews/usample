import numpy as np

def GetGR(x):
    
    xs = np.shape(x)
     
    ndim = xs[2]
    
    gr = np.zeros( (ndim, 1) )
    
    for ii in range(ndim):
         
        xi=np.squeeze( x[:,:,ii] )
        
        gr[ii] = compute_gr( xi )
        
    return np.max( gr )


def compute_gr(x):
    
    # http://astrostatistics.psu.edu/RLectures/diagnosticsMCMC.pdf
    # page 23
    
    xs = np.shape(x) 
    
    N = xs[0]
    Nw = xs[1]
    
    W =  np.sum(  (x - np.mean(x,0) )**2 )  / ( Nw * (N-1) )
    
    
    B_N = np.sum(  (np.mean(x,0) - np.mean(x) )**2 )  / ( Nw  - 1 )
    
    sig2 = (N-1) * W / N +  B_N
    
    R = (Nw +1 ) * sig2 / (W* Nw )  - (N -1 ) / (Nw * N )
    
    R = np.sqrt( sig2 / W )
    
    
    return R -1.0
     
    
        
        
    
    
    
    
    