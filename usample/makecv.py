import numpy as np 

def get_cv(pp, opts):
    """
    Convert the point into the collective variable coordinate system.

    Parameters
    ----------
    pp : array
        The point to convert to a CV.
    opts : list
        A list giving the details of the CV. 

    Returns
    -------
    cv : array
        The value of the CV.
    """ 
    if (opts is None):
        return retzero(pp) 
    
    cvtype, vals = opts
    
    if (cvtype=="line"): 
        return get_cv_line(pp, vals)
    
    if (cvtype=="grid"): 
        return get_cv_grid(pp, vals)
    
    return retzero(pp) 

def get_ic( cv, opts):
    """
    Return a point in the true space that has a given CV.

    Parameters
    ----------
    cv : array
        The CV point to convert into a normal point.
    opts : list
        A list giving the details of the CV. 

    Returns
    -------
    pp : array
        The value of the point.
    """  
    cvtype, vals = opts
    
    if (cvtype=="line"): 
        return get_pp_line(cv, vals)
    
    if (cvtype=="grid"): 
        return get_pp_grid(cv, vals)
    
    return 0


def get_cv_line(pp,vals):
    """
    Find the value of the CV when it is defined by a line.

    Parameters
    ----------
    pp : array
        The point to convert to a CV.
    opts : list
        A list giving the details of the CV. 

    Returns
    -------
    cv : array
        The value of the CV.
    """ 
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

def get_cv_grid(pp,vals):
    """
    Find the value of the CV when it is defined as a grid.

    Parameters
    ----------
    pp : array
        The point to convert to a CV.
    opts : list
        A list giving the details of the CV. 

    Returns
    -------
    cv : list
        The value of the CV.
    """ 
    v1,v2,v3 = vals
    
    cv1 = get_cv_line(pp,[v1,v2])
    cv2 = get_cv_line(pp,[v1,v3])
    
    return [cv1,cv2]

def get_pp_line(cv, vals):
    """
    Return a point that gives a particular CV, when the collective variables are defined through a line.

    Parameters
    ----------
    cv : array
        The target cv.
    vals : list
        A list of the line defining the CV. 

    Returns
    -------
    pp : array
        The value of the point.
    """ 
    v1,v2 = vals

    pp = v1 + cv*(v2-v1)
    
    return pp

def get_pp_grid(cv, vals):
    """
    Return a point that gives a particular CV, when the collective variables are defined through a grid.

    Parameters
    ----------
    cv : array
        The target cv.
    vals : list
        A list of the line vectors defining the CV. 

    Returns
    -------
    pp : array
        The value of the point.
    """ 
    v1,v2,v3 = vals

    pp = v1 + cv[0]*(v2-v1) + cv[1]*(v3-v1)
    
    return pp


def retzero(pp): 
    """
    Returns the zero point.

    Parameters
    ----------
    pp : array
        A point with given dimension.

    Returns
    -------
    pp : array
        The zero point of matching dimension.
    """ 
    if (pp.ndim==1):
        return 0 
    return np.zeros( np.shape(np.array(pp))[1]  )
 
        
    
    