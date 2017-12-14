import numpy as np 

def getcv(pp, opts):
    
    if (opts is None):
        return retzero(pp) 
    
    cvtype, vals = opts
    
    if (cvtype=="line"): 
        return getcv_line(pp, vals)
    
    if (cvtype=="grid"): 
        return getcv_grid(pp, vals)
    
    return retzero(pp) 

def getic( cv, opts):
    
    cvtype, vals = opts
    
    if (cvtype=="line"): 
        return getpp_line(cv, vals)
    
    if (cvtype=="grid"): 
        return getpp_grid(cv, vals)
    
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

def getcv_grid(pp,vals):
    v1,v2,v3 = vals
    
    cv1 = getcv_line(pp,[v1,v2])
    cv2 = getcv_line(pp,[v1,v3])
    
    return [cv1,cv2]

def getpp_line(cv, vals):

    v1,v2 = vals

    pp = v1 + cv*(v2-v1)
    
    return pp

def getpp_grid(cv, vals):

    v1,v2,v3 = vals

    pp = v1 + cv[0]*(v2-v1) + cv[1]*(v3-v1)
    
    return pp


def retzero(pp): 
    if (pp.ndim==1):
        return 0 
    return np.zeros( np.shape(np.array(pp))[1]  )


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
    
        
        
    
    