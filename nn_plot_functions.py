from __future__ import division  # Ramin
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def patchplane(ax, x, y, z, alpha):
    pc = Poly3DCollection([list(zip(x,y,z))],alpha=alpha)       # Create PolyCollection from coords
    pc.set_facecolor('yellow')                             # Set facecolor to mapped value
    pc.set_edgecolor('k')                           # Set edgecolor to black
    ax.add_collection3d(pc)                         # Add PolyCollection to axes
    return pc

def plotplane(ax,comps,B,alpha):    # components (e.g. PCs), bias 
    px = [-B/comps[0], 0, 0]
    py = [0, -B/comps[1], 0]
    pz = [0, 0, -B/comps[2]]       
    patchplane(ax, px, py, pz,alpha) 

def plot_decisionplane(ax,comps,B):
    tmp = np.linspace(-ax.get_xlim3d()[1]+1,ax.get_ylim3d()[1]+1,300)
    z = lambda x,y: (-B - comps[0]*x - comps[1]*y) / comps[2]
    x,y = np.meshgrid(tmp,tmp) 
    Z = z(x,y)
    Z[ Z < ax.get_zlim3d()[0]] = np.nan
    Z[ Z > ax.get_zlim3d()[1]] = np.nan
    ax.plot_surface(x, y, Z,  alpha=.3,color=(0,0,0))

def plot_decisionline(ax,comp_x,comp_y,B):
    xmin = ax.get_xlim()[0]
    xmax = ax.get_xlim()[1]
    yval1 = (-B - comp_x*xmin )/comp_y
    yval2 = (-B - comp_x*xmax )/comp_y
    ax.plot([xmin,xmax],[yval1,yval2],color='k',linewidth=3,alpha=0.1)
        

def vecref(ref,vec,SCALE):  # (helper function for matplotlib plot and plot3D) Vector relative to reference point
    ref = ref.squeeze()
    vec = vec.squeeze()
    out = []
    for i in range(ref.size):
        out.append( np.array([ ref[i] , ref[i] + vec[i]*SCALE ]).squeeze() )
    return out
        
def plot_linear_mode_2D( ax, xstar, v, SCALE, **kwargs ):
    out = vecref(  xstar,  +v  , SCALE)        # + direction
    ax.plot(out[0],out[1], **kwargs )
    out = vecref(  xstar,  -v  , SCALE)        # - direction
    ax.plot(out[0],out[1], **kwargs )

def plot_linear_mode_3D( ax, xstar, v, SCALE, **kwargs ):
    out = vecref(  xstar,  +v  , SCALE)        # + direction
    ax.plot3D(out[0], out[1], out[2], **kwargs )
    out = vecref(  xstar,  -v  , SCALE)        # - direction
    ax.plot3D(out[0],out[1], out[2], **kwargs )


def linearcolor(x):  # 1D-3D inputs
    
    if x.size > 2:
        clr = 'g'
    else:
        maxclr = 0.9
        maxval = 1.5

        clr = np.zeros(3)

        for c in range(x.size):
            if np.abs(x[c]) > maxval:
                clr[c]         = maxclr
            else:
                clr[c] = maxclr * np.abs(x[c])/maxval
    
    return clr

def colorbarplot(cmap,vmin,vmax,label):  # plots a colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)
    cmap = cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=ax, orientation='horizontal', label=label)

def colorbarmapper(x,xmin,xmax,cmap):  # get (r,g,b) for x values, given colormap
    colormap = cm.get_cmap(cmap)
    # (just for clarity) explicitly apply min and max values
    # x[x > xmax] = xmax       # assign ceiling value
    # x[x < xmin] = xmin       # floor vaglue
    xnorm  = (x-xmin) / (xmax-xmin)  # scale to [0,1]
    rgba   = colormap(xnorm)
    clrs   = rgba[:,0:3]     
    return clrs        


def bincenters(binedges):
    binsize = binedges[1] - binedges[0]
    return binedges[:-1]+binsize/2

def nanhist(centers,vals):
    
    # helper function for area-based histogram plots

    # identify edges of positive histogram areas #
    naninds             = np.full((vals.shape),False)
    naninds[vals==0]    = True             
    isedge          = np.diff( (vals > 0).astype(int) )
    naninds[np.where(isedge==+1)[0]] = False
    naninds[np.where(isedge==-1)[0]+1] = False

    # replace zeros with nans #
    vals                = np.array(vals,dtype=np.float64)
    vals[naninds]       = np.nan
    centers[naninds]    = np.nan

    return centers, vals

def padhist(cents,vals,n):

    binsize = np.unique(np.diff(cents))[0]

    cents = np.append(np.array(cents[0]-binsize/2),cents)  # front
    cents = np.append(cents,np.array(cents[-1]+binsize/2))  # back

    vals = np.append(np.array(0),vals)
    vals = np.append(vals,np.array(0))

    n = np.append(np.array(0),n)
    n = np.append(n,np.array(0))

    return cents, vals, n

def area_histogram(ax,vals,binedges,lw,clr,alpha1,alpha2,zorder1,zorder2,normalize=False):
    n, edges    = np.histogram(vals,binedges)
    centers     = bincenters(edges)
    cents, vals = nanhist(centers,n)
    if vals[0] > 0 or vals[-1] > 0:     
        cents, vals, _ = padhist(centers,vals,n)
    if normalize:
        vals = vals/np.nansum(vals)
        n    = n/np.nansum(n)
        maxy = np.max(n)
    else:
        maxy = 5*np.ceil(np.max(n)/5)
    ax.plot(cents,vals,linewidth=lw,color=clr,alpha=alpha1,zorder=zorder1)
    ax.fill_between(cents,vals,facecolor=clr,edgecolor=None,alpha=alpha2,zorder=zorder2)

    return maxy


def Plot_FP_distances(D):

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(1700,400,1100,500) 

    T_plot = D['Out_fd'].shape[0]

    ax = plt.subplot(2,1,1)  
    clrs = ['r','g','b','m','y','c','slategrey','bisque','darkorange','plum','deepskyblue','salmon','peru','cadetblue','hotpink','k','k','k']
    for b in range(49):
        for fp in range(  D['Out_fd'].shape[2]  ):
            plt.scatter(    np.arange(T_plot), D['Out_fd'][:,b,fp].squeeze() , c = clrs[fp]       )
            plt.plot(       np.arange(T_plot), D['Out_fd'][:,b,fp].squeeze() , color = clrs[fp]      )
    ax.set_xlim([0,T_plot-1])

    ax = plt.subplot(2,1,2)  
    for b in range(49):
        plt.scatter(  np.arange(T_plot),       D['Out_f'][:,b] , c = 'k'       )
        plt.plot(     np.arange(T_plot),       D['Out_f'][:,b] , color = 'k'    )
    ax.set_xlim([0,T_plot-1])

    fig.subplots_adjust(hspace=0.5)

    return ax    