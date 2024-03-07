
#%% Construct and run model ###############################

from __future__ import division
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
mpl.use("Qt5Agg") # set the backend  
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
matplotlib.use("Qt5Agg") # set the backend
matplotlib.rcParams.update({'font.size': 20})
matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"
import matplotlib.patches as patch
from mpl_toolkits.mplot3d import Axes3D
mpl.rcParams.update({'font.size': 20})
mpl.rcParams['font.sans-serif'] = "Helvetica"
mpl.rcParams['font.family'] = "sans-serif"
from importlib import reload 
import sys
import _tkinter  
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions.normal import Normal
import copy

# local #
import task_train_test      as tk
import nn_models            as nm
import nn_plot_functions    as npf
import nn_analyses          as nna
import ti_functions         as ti
import postproc             as pp

################# Basic settings ################
System              = 1    # 1: original                 (1 real, 2 oscillatory)
                           # 2: real part zeroed out     (oscillatory only, real eigenvalue is zeroed out)
Filter_IC           = 0    # 0: raw IC  (i.e. in 3D)
                           # 1: filter the IC Real (non-oscillatory) mode
EXTERNAL_PULSE      = 0                # at 100th timestep, get external impulse along 3rd L eigenvector 
    # initial conditions #
xinit               = torch.Tensor([0,-1,1])            # (0,-1,+1)
# xinit                 = torch.from_numpy(q.v[0])          # <1st R eigenvector>         i.c. with simple x trajectories
# xinit                 = torch.from_numpy(q.v[1])          # <2nd R eigenvector>       i.c. with simple x trajectories
xinit                 = torch.from_numpy(q.v[2])          # <3rd R eigenvector>       i.c. with simple x trajectories
# xinit                 = torch.from_numpy(q.w[2])          # <3rd L eigenvector>       i.c. with simple x trajectories
# xinit               = torch.Tensor([0,0,0])               
# xinit               = torch.Tensor([2,-2,3])

################# Plot Settings  ###########################
Osci_scale          = 4
Osci_numgrid        = 60   #   number of grid spacings for flow field

State_1D_plot       = System   # 1: Original system, 2: Reduced system (osci only)
State_3D_plot       = System   # 1: Original system, 2: Reduced system (osci only)
Osci_plane_plot     = System   # 1: Original system, 2: Reduced system (osci only)


if Filter_IC == 0:
    filterstring = 'original IC'
elif Filter_IC == 1:
    filterstring = 'osci-filtered IC'


class params():
   def __init__(self):

        # basic #
        self.N             = 3       # 3D system
        self.U             = 3       # 3D input
        self.tau           = 1       # time constant (1 s)
        self.dt            = 0.01    # time step     (10 ms)
        self.T             = 20      # total time    (20 s)

        # linear dynamics #
        self.xstar         = np.transpose( np.array([ 0, 0, 0 ], dtype='float64') )  # center
        self.Jac           = np.array( [ [-1,-10,-10], [1,0,0], [0,1,0]  ] ,dtype='float64')
        # self.Jac           = np.array( [ [-1,-.5,-1],[1,0,0],[0,1,0]  ] ,dtype='float64')
        # self.Jac           = np.array( [ [-1,-2,-3],[1,2,0],[.3,4,1]  ] ,dtype='float64')
        # self.Jac           = np.float64( 2*np.random.randn( 3,3 ) )

        # diagonalization #
        l, R               = np.linalg.eig(self.Jac)    # l: eigvals, R: right eigvector matrix (cols are eigvecs)
        L                  = np.linalg.inv(R)           # L: left eigvector matrix (rows are eigvecs)
        self.l             = l
        self.R             = R
        self.L             = L

def reduce_p( p ):
    
    # Takes linear system (in p) and eliminates the real-only mode #
    #   creates params object q 

    # copy #
    q       = copy.deepcopy(p)

    # I.  Reduce Jacobian + identify modes #
    Jac              = q.Jac
        # diagonalize #  xdot = R @ E @ L * x
    l, R             = np.linalg.eig(Jac)  # l: eigvals, R: right-eigvec matrix (cols are eigvecs)
    L                = np.linalg.inv(R)    # L: left-eigvec matrix (rows are eigvecs)
        # identify real + oscillatory modes # 
    r                = np.where( np.isreal(l) )[0]
    o1               = np.where( np.invert( np.isreal(l) ) )[0][0]
    o2               = np.where( np.invert( np.isreal(l) ) )[0][1]

    #   zero-out the real mode #    
    l[r]             = 0  # zero out
    E                = np.complex64(np.diag( l ))
        # reconstitute #
    q.Jac            = np.float64(   np.dot(R,np.dot(E,L))  )


    # II.  Define basis of the new system #

    # From LEFT eigenvectors, get LINEAR FUNCTIONS (projections / filters ?) of state x, which evolve simply #
    # "LEFT eigenvectors give LINEAR FUNCTIONS of state that are simple, for any initial condition"
    w1               = np.real(L[o1,:])                 # osci axis 1
    w2               = np.imag(L[o1,:])                 # osci axis 2
    w3               = np.real(L[r,:]).flatten()        # real axis

    # From RIGHT eigenvectors, get INITIAL CONDITIONS from which trajectories are simple  #
    # "RIGHT eigenvectors are INITIAL CONDITIONS from which resulting motion (x trajectory) is simple (i.e., remains on line or in plane)"
    v1               = np.real(R[:,o2])                 # osci axis 1
    v2               = np.imag(R[:,o2])                 # osci axis 2
    v3               = np.real(R[:,r]).flatten()        # real axis

    # Define oscillatory subspace (+ orthonormal basis vectors via QR), made from R eigenvectors
    q.oscbasis       = nna.Linearbasis_2D( q.xstar, v1, v2 )  

    q.v              = (v1,v2,v3)
    q.w              = (w1,w2,w3)
    q.R              = R
    q.L              = L
    q.l              = l


    return q


class LM():   # Linear Model #

    def __init__( self, p  ):                                
                
        # Linear parms #
        self.Jac      = Variable( torch.from_numpy( p.Jac   ),   requires_grad=False )      # Jacobian
        self.xstar    = Variable( torch.from_numpy( p.xstar ),   requires_grad=False )    # fixed point

    def forward(self, p, u, *xinit):
        out  = self.dynamical( p, u, *xinit)
        return out   

    def dynamical(self, p, u, *xinit):

        # (basic values) #
        numdt   = u.shape[0]                       # trial / input duration
        B       = u.shape[1]                       # batch size
        numFP   = 1                      # no. of fixed points

        # Initialize outputs #
        xs      = np.zeros( (numdt, p.N, B) )            # cell activation [T, N, B]  time, state variable, condition

        # 1. Set initial condition  #
        if len(xinit) == 0:     # none specified outside of .forward()
            # x0   = 10 * torch.randn(B,p.N)            # train
            x0     = torch.Tensor([0,-1,1])         
            x0     = x0[None,:]                     #torch.Tensor([0,-1,1])
        else:                    
            x0     = xinit[0]   # [B,N]
            if xinit[0].numel() == 3:
                x0 = x0[:,None]     # [N,B]
            else:
                x0 = xinit[0]

        # 2. Identify the FP to adopt linearized dynamics #
        xstar   = self.xstar[:,None]   # [1,3]
 
        # 4. Calculate y0, set #     (i.e. initial condition for y, the displacement from FP) #
        y0    = x0 - xstar        # [B,N]
        y     = y0                # [N,B] 

        # Run simulation #
        for ti in range(numdt):

            # i.  Linear dynamics #    (Jacobian, A)
            Ay = torch.zeros( p.N, B )
            for b in range(B):   
                A         = self.Jac   # Jacobian
                Ay[:,b]   = A.mm( y[:,b,np.newaxis] ).flatten()                             

            # Dynamical update #
            # print(u[ti,:,:].shape)
            # print(Ay.shape)
            y               = y + (p.dt/p.tau) * ( Ay + u[ti,:,:].T  )    # new activation, u[ti,:,:]:[U,B], wu:[N,U] 
            #      [N,B]                [N,B] [N,B] [N,1]  [N,B]          [N,B]

            # De-center from FP to recover x #
            x               = y + xstar
   
            xs[ti,:,:]      = x.detach().numpy()

        return xs   


#%%######################### Study #################################

# 1. Construct the linear models #
p               = params()      # params for original system
q               = reduce_p(p)   # params for real-zeroed (reduced) system
model_lin       = LM( p )       # model
model_lin_r     = LM( q )       # real-zeroed model

# 2a. Make a blank input #
numdt      = int( p.T / p.dt )
B          = 1
u          = tk.get_blank_input( numdt, 3, B )

# 2b. (optional) Construct an impulse input along the 3rd R eigenvector  #
if EXTERNAL_PULSE:
    B          = 1
    u          = tk.get_blank_input( numdt, 1, B )
    u          = Variable( torch.zeros( [numdt, B, 3] ), requires_grad=False) 
    u[100:101,:,:] = 100 * torch.from_numpy( q.v[2] ).T    # instantaneous pulse along 3rd

# 3. (optional) Filter x0 (IC) to keep only the components that lie within the Oscillatory subspace #
if Filter_IC:  
    xinit      = xinit[None,:]
    xinit      = q.oscbasis.filter(data = xinit, center=True).flatten()

# 4. Run the two linear model #
xs          = model_lin.dynamical( p, u, xinit )    # Original model
xsr         = model_lin_r.dynamical( p, u, xinit )  # Real-zeroed model

# 5. Report eigvals #
l, R             = np.linalg.eig(p.Jac)  # l: eigvals, R: right-eigvec matrix (cols are eigvecs)
L                = np.linalg.inv(R)    # L: left-eigvec matrix (rows are eigvecs)
E                = np.complex64( np.diag( np.zeros((p.N)) ) )
print('eigenvalues:')
print('%s' % (np.array2string(l[0])))
print('%s' % (np.array2string(l[1])))
print('%s' % (np.array2string(l[2])))






#%% State space 1D  ###########################

if State_1D_plot == 1:
    string = 'Original system'
    data = xs
elif State_1D_plot == 2:
    string = 'Osci-only system'
    data = xsr


fig = plt.figure(constrained_layout=True,figsize=(14,5))
plt.get_current_fig_manager().window.setGeometry(100,200,1000,300) 
plt.show(block=False)
plt.suptitle('State Space (x)\n%s, %s' % (string,filterstring) ,fontsize=22)
for xx in range(3):  # zoom scale
    ax = plt.subplot(1,3,xx+1)  
    ax.plot( (p.dt) * np.arange(numdt) , data[:,xx,0] )
    ax.set_xlim([0,p.T])
    ax.plot( [0,p.T] , [0,0], color='k', alpha=0.2 )
    ax.set_ylim([-10,10])
    ax.set_xlabel('Time (sec)',fontsize=20)
    ax.set_ylabel('Amplitude',fontsize=20)
    ax.set_title( 'x%d' % (xx) , fontsize=20)

plt.draw()     
plt.show()
plt.pause(0.0001)

plt.ion()
plt.ioff()
plt.ion()

#%% State space 3D ######################################

if State_3D_plot == 1:
    string = 'Original system'
    data = xs
elif State_3D_plot == 2:
    string = 'Osci-only system'
    data = xsr

plt.ion()

fig     = plt.figure(constrained_layout=True,figsize=(7,7))
plt.get_current_fig_manager().window.setGeometry(1150,50,800,800) 
ax = fig.gca(projection='3d')
mpl.interactive(True)
plt.suptitle('State Space (x)\n%s, %s' % (string,filterstring) ,fontsize=22)


    # initial condition #
ax.scatter3D(   data[0,0,0], data[0,1,0], data[0,2,0], s=150, marker='o',color='g', linewidth=2, alpha=0.6)       # Trajectory
    # trajectory #
ax.plot3D(      data[:,0,0], data[:,1,0], data[:,2,0], color='k', linewidth=2, alpha=0.2, label='hi')     # Trajectories
ax.plot3D(      xsr[:,0,0], xsr[:,1,0], xsr[:,2,0], color='k', linewidth=2, alpha=0.05, label='hi')       # Osci-only linear system trajectories
    # FP #
ax.scatter3D(p.xstar[0], p.xstar[1], p.xstar[2], s=200, marker='x', color='k', linewidth=1, alpha=0.9)       # Fixed Point
    # origin #
ax.scatter3D( 0, 0, 0, s=100, marker='o', color='k', alpha=0.5)       # Fixed Point

    # oscillatory axis #  (via R eigvecs)
SCALE = 2
npf.plot_linear_mode_3D( ax, p.xstar, q.v[0], SCALE=SCALE, color=[0,0,1], linewidth=2)
npf.plot_linear_mode_3D( ax, p.xstar, q.v[1], SCALE=SCALE, color=[.5,.5,1], linewidth=2)
npf.plot_linear_mode_3D( ax, p.xstar, q.v[2], SCALE,       color='r', linewidth=2)

    # The 3rd L eigvec # -- Normal vector to Oscillatory Plane
w3   = q.L[2,:]/np.linalg.norm(q.L[2,:])
ax.plot3D( [0,w3[0]] , [0,w3[1]], [0,w3[2]], color='k', alpha=0.5  )
ax.scatter3D( w3[0] , w3[1], w3[2], color='m', s=100, alpha=0.5  )      
w3n   = w3 / np.linalg.norm(w3)             # normalized L eigvec
b    = np.dot( np.real(w3n), xs[0,:,0].squeeze()  )
ax.plot3D( [0,b*w3n[0]] , [0,b*w3n[1]], [0,b*w3n[2]], color='k', alpha=0.5  )



# limits #
ax.set_xlim3d(-2, 2)
ax.set_ylim3d(-2, 2)
ax.set_zlim3d(-2, 2)

# labels #
ax.set_xlabel('x0',fontweight='bold',fontsize=20)
ax.set_ylabel('x1',fontsize=20)
ax.set_zlabel('x2',fontsize=20)
ax.set_xticks(np.arange(-2,3,1))
ax.set_yticks(np.arange(-2,3,1))
ax.set_zticks(np.arange(-2,3,1))

ax.view_init(elev=10, azim=15)


ax.set_box_aspect(aspect = (1,1,1))
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))


plt.show()




#%% Oscillatory (2D) subspace   #####################################

if Osci_plane_plot == 1:             # Original model
    data = xs
elif Osci_plane_plot == 2:           # Reduced model
    data = xsr

data2             = nna.project_data(   data.squeeze(), q.oscbasis, center=True )

fig = plt.figure(constrained_layout=True,figsize=(14,5))
plt.get_current_fig_manager().window.setGeometry(1970,50,700,700) 
plt.suptitle('Oscillatory subspace \n%s, %s' % (string,filterstring) ,fontsize=22)

# figure basics #
ax          = plt.subplot(1,1,1)  
gmax        = Osci_scale
gmin        = -Osci_scale
grid        = (Osci_numgrid,gmin,gmax)

# plot flow field #
arrowsize = 8
nna.plot_flowfield_simple( ax, grid, p, q.oscbasis, model_lin, arrowsize )  ###############################

# # plot flow test #
# t_flow      = p.t1
# arrowsize   = 0.01
# nna.plot_flowtest( ax, data2[-1], p, oscbasis, model_lin , t_flow , arrowsize ) ##################################

# trajectory #
plt.plot(       data2[:,0],  data2[:,1],  color=(.7,.7,.7), alpha=0.6)                          # traj  (line) 
plt.scatter(    data2[:,0],  data2[:,1],  s=10, color=(.7,.7,.7), alpha=0.8)                          # traj  (line) 

# early events   
plt.scatter(    data2[0,0],    data2[0,1], s=100, color='g',  marker='o', alpha=0.5)       # i1      (green)
# plt.scatter(    data2[t1,0],    data2[t1,1],    color=clr[LR],  marker='.', s=mksize,zorder=100)       # i1      (green)
# plt.scatter(    data2[t1+1,0],  data2[t1+1,1],  color='k',  marker='.', s=mksize,zorder=100)       # i1      (green)

ax.scatter(0,0,color='k', marker='o',s=100)  # plot origin
ax.set_aspect('equal', 'box')
ax.set_xlim([gmin,gmax])
ax.set_ylim([gmin,gmax])
ax.set_xlabel('Axis 1', fontsize=24, fontweight='bold')
ax.set_ylabel('Axis 2', fontsize=24, fontweight='bold')


fig.subplots_adjust(hspace=1)
plt.tight_layout()



#%% State space 1D: L eigenvectors (w) ##

# w1               = np.real(L[o1,:])
# w2               = np.imag(L[o1,:])
# w3               = np.real(L[r,:])

ws      = np.matmul( p.L, xs.squeeze().transpose() )    # mode amplitude, original sys
wsr     = np.matmul( p.L, xsr.squeeze().transpose() )    # mode amplitude, reduced sys

fig = plt.figure(constrained_layout=True,figsize=(14,5))
plt.get_current_fig_manager().window.setGeometry(100,800,1000,600) 
plt.show(block=False)

plt.suptitle('Left eigenvectors (w) \n%s' % (filterstring),fontsize=24)


for sys in [0,1]:
    if sys == 0:    # original
        string = 'Original system'
        data = ws
    elif sys == 1:  # reduced
        string = 'Osci-only system'
        data = wsr
    for ww in range(3):  # zoom scale
        ax = plt.subplot(2,3,sys*3+ww+1)  
        ax.plot( (p.dt) * np.arange(numdt) , data[ww,:] )
        ax.plot( [0,p.T] , [0,0], color='k', alpha=0.2 )
        ax.set_xlim([0,p.T])
        ax.set_ylim([-5,5])
        ax.set_xlabel('Time (sec)',fontsize=20)
        if ww == 0:
            ax.set_ylabel('%s\nAmplitude' % (string),fontsize=20)
        if sys == 0:
            ax.set_title( 'w%d' % (ww) , fontsize=20)

# %%
