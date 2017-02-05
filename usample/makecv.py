import numpy as np 

class Makecvfn:
    
    def __init__(self,v1, v2 ):
        
        self.v1 = np.array(v1)
        
        self.v2 = np.array(v2)
        
        self.r = self.v2-self.v1 
        
        self.rlen2 = 1.0*np.sum( self.r*self.r )
        
    def getcv(self,pp):
    
        p = np.array(pp)
    
        if (p.ndim==1):
            cv = np.dot( self.r , p-self.v1  )
        else:
            rr = np.expand_dims(self.r,axis=1)  
            cv = np.dot(   (p-np.expand_dims(self.v1,axis=1)).T ,  rr  )
            cv=cv.flatten()
            
        cv = cv / self.rlen2
        
        cv = np.fmax( cv , 0 )
        
        cv = np.fmin( cv , 1 )
         
        
        return cv
    
        
        
    
    