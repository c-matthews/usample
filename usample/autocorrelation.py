# -*- coding: utf-8 -*-
"""
Tools for analyzing the autocorrelation of a time series
"""
import numpy as np

def autocorrfxn(timeseries,lagmax):
    ts = np.asarray(timeseries)
    ts -= np.average(ts) # Set to mean 0
    N = len(timeseries)
    corrfxn = np.zeros(lagmax)
    for dt in range(lagmax):
        corrfxn[dt] = (np.dot(timeseries[0:N-dt],timeseries[dt:N])) # sum of ts[t+dt]*ts[t]

    if (corrfxn[0]>0):
        corrfxn /= corrfxn[0] # Normalize
    return corrfxn


def ipce(timeseries,lagmax=None):
    """
    Initial positive correlation time estimator
    """
    if (len(timeseries)<3):
        return 1,1,1
    timeseries = np.copy(timeseries)
    mean = np.average(timeseries)
    if lagmax == None:
        lagmax = len(timeseries)/2
    corrfxn = autocorrfxn(timeseries,lagmax)
    i = 0
    t = 0

    while i < 0.5*lagmax-1:
        gamma =  corrfxn[2*i] + corrfxn[2*i+1]
        if gamma < 0.0:
#            print('stop at %d'%(2*i))
            break
        else:
            t += gamma
        i += 1
    tau = 2*t - 1
    var = np.var(timeseries)
    sigma = np.sqrt(var * tau / len(timeseries))
    return tau, mean, sigma

def _cte(timeseries,maxcorr):
    timeseries = np.copy(timeseries)
    mean = np.average(timeseries)
    corrfxn = autocorrfxn(timeseries,maxcorr)
    tau = 2*np.sum(corrfxn)-1
    var = np.var(timeseries)
    sigma = np.sqrt(var * tau / len(timeseries))
    return tau, mean, sigma


def icce(timeseries,lagmax=None):
    """
    Initial convex correlation time estimator
    """
    timeseries = np.copy(timeseries)
    if lagmax == None:
        lagmax = len(timeseries)/2
    corrfxn = autocorrfxn(timeseries,lagmax)
    t = corrfxn[0] + corrfxn[1]
    i = 1
    gammapast = t
    gamma = corrfxn[2*i] = corrfxn[2*i+1]
    while i < 0.5*lagmax-2:
        gammafuture =  corrfxn[2*i+2] + corrfxn[2*i+3]
        if gamma > 0.5*(gammapast+gammafuture) :
            print('stop at %d'%(2*i))
            break
        else:
            t += gamma
            gammapast = gamma
            gamma = gammafuture
        i += 1
    tau = 2*t - 1
    var = np.var(timeseries)
    mean = np.average(timeseries)
    sigma = np.sqrt(var * tau / len(timeseries))
    return tau, mean, sigma

def _get_iat_method(iatmethod):
    """Control routine for selecting the method used to calculate integrated
    autocorrelation times (iat)

    Parameters
    ----------
    iat_method : string, optional
        Routine to use for calculating said iats.  Accepts 'ipce', 'acor', and 'icce'.

    Returns
    -------
    iatroutine : function
        The function to be called to estimate the integrated autocorrelation time.

    """
    if iatmethod=='acor':
        from acor import acor
        iatroutine = acor
    elif iatmethod == 'ipce':
        from .autocorrelation import ipce
        iatroutine = ipce
    elif iatmethod == 'icce':
        from .autocorrelation import icce
        iatroutine = icce
    return iatroutine
