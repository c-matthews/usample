import matplotlib.pylab as plt
import numpy as np

def plot_pretty(dpi=175,fontsize=9):
    # import pyplot and set some parameters to make plots prettier
    import matplotlib.pyplot as plt

    plt.rc("savefig", dpi=dpi)
    plt.rc('text', usetex=True)
    plt.rc('font', size=fontsize)
    plt.rc('xtick.major', pad=5); plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5); plt.rc('ytick.minor', pad=5)
    plt.rc('xtick', direction='in') 
    plt.rc('ytick', direction='in')
    return


import scipy.optimize as opt
from matplotlib.colors import LogNorm

def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def plot_2d_dist(x,y, xlim,ylim,nxbins,nybins, weights=None, xlabel='x',ylabel='y', clevs=None, smooth=None, fig_setup=None, savefig=None):
    if fig_setup == None:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        #ax = plt.add_subplot(1,1,1)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
    #
    if xlim[1] < 0.: ax.invert_xaxis()

    if weights == None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); 
             
    X,Y = np.meshgrid(xbins,ybins) 
    if smooth != None:
        from scipy.signal import wiener
        H = wiener(H, mysize=smooth)

    H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
    
    pcol = ax.pcolormesh(X,Y,(Hmask), vmin=1.e-15*np.max(Hmask), cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')

    if clevs != None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
                   
        ax.contour(H, linewidths=np.linspace(1,2,len(lvls)), colors='black', levels = sorted(lvls), 
                    norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup == None:
        plt.show()
    return

if __name__ == '__main__':
    plot_pretty()
