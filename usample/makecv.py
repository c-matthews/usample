import numpy as np 

def getcv(pp, opts):
    
    if (opts is None):
        return 0
    
    cvtype, vals = opts
    
    if (cvtype=="line"): 
        return getcv_line(pp, vals)
    
    return 0

def getic( cv, opts):
    
    cvtype, vals = opts
    
    if (cvtype=="line"): 
        return getpp_line(cv, vals)
    
    return 0


def getcv_line(pp,vals):
    v1,v2 = vals
    
    r = v2 - v1
    rlen2 = np.sum( r*r )*1.0
    
    p = np.array(pp)

    if (p.ndim==1):
        cv = np.dot( r , p-v1  )
    else:
        rr = np.expand_dims(r,axis=1)  
        cv = np.dot(   (p-np.expand_dims(v1,axis=1)).T ,  rr  )
        cv=cv.flatten()
        
    cv = cv / rlen2
    
    cv = np.fmax( cv , 0 )
    
    cv = np.fmin( cv , 1 )
    
    return cv

def getpp_line(cv, vals):

    v1,v2 = vals

    pp = v1 + cv*(v2-v1)
    
    return pp



#class Makecvfn:
    
    #def __init__(self,v1, v2 ):
        
        #self.v1 = np.array(v1)
        
        #self.v2 = np.array(v2)
        
        #self.r = self.v2-self.v1 
        
        #self.rlen2 = 1.0*np.sum( self.r*self.r )
        
    #def getcv(self,pp):
    
        #p = np.array(pp)
    
        #if (p.ndim==1):
            #cv = np.dot( self.r , p-self.v1  )
        #else:
            #rr = np.expand_dims(self.r,axis=1)  
            #cv = np.dot(   (p-np.expand_dims(self.v1,axis=1)).T ,  rr  )
            #cv=cv.flatten()
            
        #cv = cv / self.rlen2
        
        #cv = np.fmax( cv , 0 )
        
        #cv = np.fmin( cv , 1 )
         
        
        #return cv
    
        
        
    
    