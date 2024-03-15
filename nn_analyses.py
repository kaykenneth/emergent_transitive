from __future__ import division
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
import time

import task_train_test      as tk



### PCA ###################################################

def PCA_Calc(Data, Reference_TT):
    # Data: RNN activations
    # Reference_TT: index of the time/input condition for which to calc PCA
    pca = PCA()   
    pca.fit(Data[Reference_TT])                            # input to PCA is [samples,features]
    return pca

def PCA_plot(Data):      

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D


    figg = plt.figure(figsize=(10,8))
    plt.get_current_fig_manager().window.setGeometry(100,600,400,400) 

    N       = Data['DATA'][0].shape[1]
    pca     = Data['pca']

    p3 = plt.subplot(1,1,1)
    if N < 10:
        MAXPC = N
    else:
        MAXPC = 10
    p3.bar(np.linspace(0,MAXPC-1,MAXPC),100*pca.explained_variance_ratio_[0:MAXPC],color='black')
    top3VE = 100*np.sum(pca.explained_variance_ratio_[0:3])
    print('Variance explained (PC1,2,3) = %d %%' % top3VE)
    p3.set_aspect('auto')
    p3.set_xlim([-0.5,MAXPC])
    plt.xticks(ticks=range(MAXPC))
    p3.set_ylim([0,100])
    p3.set_xlabel('PC #',fontsize=18)
    p3.set_ylabel('% var',fontsize=18)     




### Fixed-point Finding (FPF) functions ########################################

class params_search(object):  
    def __init__(self,**kwargs):
        # Default parameters for FPF optimization #
        self.lr              = 0.005   # learning rate of optimizer
        self.maxloss         = 1e-30   # maximum loss of FPF
        self.numepochs       = 20000   # numepochs to execute
        self.numepochs_stop  = 10000   # no. epochs to attempt loss improvement before stopping
        self.add_rand_seeds  = False       
        self.rand_batchsize  = 50  # (optional) default (nan): # of test samples in Data, otherwise randomly choose # seeds
        self.rand_numbatch   = 50       # 1: trial times, 2: random
        if len(kwargs) != 0:           # (optional) manually set params
            for key,val in kwargs.items():
                setattr(self, key, val)

class params_detect(object):
    def __init__(self,**kwargs):
        # Default parameters for FP detection #
        self.tol_q           = 8    # energy tolerance for FP detection
        self.tol_unique      = 1       # distance (Euclidean, hidden activation space) tolerance for unique SP/FP
        find_baseline_fp     = False
        if len(kwargs) != 0:           # manual
            for key,val in kwargs.items():
                setattr(self, key, val)

def FPF_search(model,p,X_input,**kwargs):

    # model:        RNN model
    # p             RNN model parameters, used in nn_models
    # Data:         RNN activations to use as seeds: [B,N]    B: batch/samples, N: hidden dim
    # pfp:          Parameters for finding fixed pts

    f = params_search(**kwargs)

    # (memorize for later restoration)
    wugrad = model.wu.requires_grad

    # I. Freeze gradients in the RNN
    model.J.requires_grad = False
    model.wu.requires_grad = False
    model.wz.requires_grad = False
    model.b.requires_grad = False
    model.bz.requires_grad = False    

    # (get task function)
    get_test_data = tk.Get_test_data(p)

    # II. Optimize #
    
    xstar = []

    if f.add_rand_seeds:
        numbatch = f.rand_numbatch + 1
    else:
        numbatch = 1   

    startTime     = time.time()

    for BB in range(numbatch):

        running_loss  = 0
        loss_min      = np.inf

        # Obtain seeds
        if BB == 0:     # trial times

            print('FPF: Selecting Trial times')
            # Numepochs = 20000            # always do 50000 training epochs for this setting
            Numepochs = f.numepochs            # always do 50000 training epochs for this setting

            # run model #
            with torch.no_grad():
                t_ext = 100
                X, _, _, _,_,_  = get_test_data(p,X_input,t_ext)       # X: [T,U,B]
                input       = torch.Tensor(np.transpose( X , (0,2,1) ))    # input: [T,B,U]
                D           = tk.run_model( model, p, input, hidden_output = True )
                h           = D['Out_h']   # [T,B,U]

            # select trial time states #
            B = 6 * h.shape[1]  # Five time events * number of input conditions
            Data = np.zeros((B ,p.N))
            k = 0
            for tt in [0, p.t1, p.t1+int(p.d1/2), p.t2, p.T-1, h.shape[0]-1]:  # critical times
                for cc in range(p.n_items**2):
                    Data[k,:] = h[tt,cc,:]  # [T,B,N]
                    k += 1
               
        elif BB > 0:   # random sample

            print('FPF: Random batch #%d' % (BB))
            Numepochs = f.numepochs 
            
            B        = f.rand_batchsize  
            Data     = np.zeros((B ,p.N))
            for b in range(B):
                with torch.no_grad():
                    t_ext = 0
                    if p.jit != 0:
                        j = np.random.randint(np.round(p.d1/2))
                    X, _, _, _,_,_  = get_test_data(p,X_input,t_ext,offsets=[-1,-1])       # X: [T,U,B]
                    input       = torch.Tensor(np.transpose( X , (0,2,1) ))    # input: [T,B,U]
                    D           = tk.run_model( model, p, input, hidden_output = True )
                    h           = D['Out_h']   # [T,B,U]
                    h           = np.reshape( h, (h.shape[0]*h.shape[1],p.N) )
                    ind         = np.random.choice( np.arange( h.shape[0] ) , size=1, replace=False )
                    Data[b,:]   = h[ind,:]

        h0_all       = np.reshape( Data, (B,p.N), order='C')  
        input        = torch.tensor( np.zeros([1,p.U,B]), dtype=torch.float32, requires_grad=False)     # 0-vector input
        h0           = torch.tensor( h0_all, dtype=torch.float32, requires_grad=True)
        # h0           = Variable(torch.from_numpy(h0_all), dtype=torch.float32, requires_grad=True) 

        # II. Optimizer & Loss #
        optimizer = optim.Adam([h0], lr=f.lr)  # Use Adam optimizer #
        criterion = nn.MSELoss()

        for i in range(Numepochs):

            # print('FPF batch %d epoch %d' % (BB,i))
            optimizer.zero_grad()   # zero the gradient buffers

            # Apply 1-step recurrent (forward) function from the trained network
            _, h1 = model.forward(p, input, True, False, h0)
            loss = criterion(h1[0][-1,:,:].t(), h0 )  # [B,N]   # original
            # loss = criterion(h1[0][-1,:,:].t().detach(), h0 )  # h0 branch only: readout FPs! -- this is probably defective, since only takes one branch of gradients

            loss.backward()
            optimizer.step()    # Does the update

            # if loss improved, save h0 #
            lossval = loss.item()
            if lossval < loss_min:
                xstar_batch           = h0.detach().numpy()  # best observed
                loss_min              = lossval
                eps_no_improvement    = 0
            else:
                eps_no_improvement    += 1

            # Stop optimization if loss over 1000 trials is below max loss #
            running_loss += lossval  # accumulates loss, i.e. adds to running_loss
            if i % 1000 == 999:
                running_loss /= 1000  
                if running_loss > f.maxloss:
                    # print('FPF: Step {}, Loss {:0.9f}'.format(i+1, running_loss))
                    print('FPF: Step {}, Loss {:e}'.format(i+1, running_loss))
                else:      
                    # print('FPF: Step {}, Loss {:0.9f} * Done!'.format(i+1, running_loss))
                    print('FPF: Step {}, Loss {:e} * Done!'.format(i+1, running_loss))
                    break
                running_loss = 0  # reset running loss

            # Stop optimization if loss hasn't improved for a number of epochs #
            if eps_no_improvement == f.numepochs_stop:
                print('Early stopping! %d epochs without improvement' % (f.numepochs_stop))
                break

        xstar += [xstar_batch]  # save output of this batch

    xstar = np.vstack(xstar)  # concatenate output from all batches together

    print("FPF time: ",time.time() - startTime)

    # V. Calc q (energy) of each candidate FP #
    xstar_q       = np.zeros([xstar.size])
    with torch.no_grad():
        input   = torch.tensor( np.zeros([1,p.U,xstar.shape[0]]), dtype=torch.float32, requires_grad=False)     # 0-vector input
        _, h1   = model.forward(p, input,True,False,torch.tensor(xstar,requires_grad=False) )
    h1val = h1[0][-1,:,:].t().detach().numpy()   # [B,N]
    h0val = xstar
    xstar_q = 0.5*np.sum((h1val-h0val)**2,axis=1) # energy
    xstar_q = xstar_q.flatten()

    # candidate FPs #
    fps             = fixedpoints(f)  # instantiate an object
    fps.num         = xstar.shape[0]         # number of candidate FPs
    fps.xstar       = xstar     # [B,N]
    fps.q           = xstar_q   # [B]
    
    # restore this (for bookkeeping)
    model.wu.requires_grad = wugrad  

    return fps



def FPF_detect(fps, model, p, X_input, pca, **kwargs):     # Detect unique FPs (Golub-Sussillo) + Linearize

    f = params_detect(**kwargs)

    B           = fps.xstar.shape[0]
    xstar       = fps.xstar
    q           = fps.q

    # a.  Sort from Slowest to Fastest
    inds        = np.argsort(q)   
    Xstar, Q    = (xstar[inds,:],q[inds])

    # b.  Detect FP + disregard non-unique FP candidates nearby
    print('Detecting unique Fixed Points..')
    inds   = Q < 10**(-f.tol_q)         # apply speed threshold
    Xstar, Q    = (Xstar[inds,:], Q[inds])
    inds_delete = np.full(Q.size,False,dtype=bool)
    for xx in range(Q.size):
        if inds_delete[xx] == True:
            continue
        for yy in range(xx+1,Q.size):
            if inds_delete[yy] == False:

                fp1 = Xstar[xx,:]
                fp2 = Xstar[yy,:]

                if True:    # unit activations
                    hdist = np.abs(fp1-fp2)               # Golub Sussillo
                    if np.any(hdist < f.tol_unique):
                        # print('%0.1f' % (hdist))
                        inds_delete[yy] = True
                # elif True:  # PCA
                #     fp1  = pca.transform(fp1[np.newaxis,:])
                #     fp2  = pca.transform(fp2[np.newaxis,:])                
                #     hdist = np.linalg.norm(fp1[:,:3]-fp2[:,:3],ord=2)       # Euclidean in PC1-10 space
                #     if hdist < f.tol_unique:
                #         inds_delete[yy] = True
                
                # else:
                #     print('%0.1f' % (hdist))

    ind         = np.invert(inds_delete)
    num_fp      = np.sum(ind)
    Xstar, Q    = (Xstar[ind,:],Q[ind])

    # d. Detect FP closest to item 1 delays, 1-input state #
    if f.detect_delay:

        # 1. Obtain activation state at the time of item 1 #
        _, _, _, X, input, _     = tk.parms_test(p, X_input, TT = 0)  # get appropriate input
        T_select                 = p.t2-1        # maximal time
        # T_select                 = 15            # minimal time for 95% jitter, 
        # T_select                 = int(p.t2/2)   # half time
        # T_select                 = 0             # negative control
        with torch.no_grad():
            D     = tk.run_model( model, p, input, hidden_output = True )
            Out_h = D['Out_h']   # [T,B,U]
            xs    = Out_h[T_select,:,:].squeeze().detach().cpu().numpy()  # baseline state (here, at default item 1 time, t1)
        xbases = xs[ 0:p.n_items, :]        #  unique trials for each possible Item 1

        for item1 in range(p.n_items):

            xbase = xbases[item1,:]

            # 2. Calculate Euc dists between candidate fps and baseline state #
            dists       = np.linalg.norm(xstar - xbase, ord=2, axis=1) 
            cands       = q < 10**(-f.tol_q)         # speed threshold
            ii          = np.argsort(dists)   # sort by dist, ascending  
            iii         = ii[cands]         

            # 3. Identify closest candidate FP #
            ind_b       = iii[0]            # index of FP nearest baseline
            xstar_bfp   = xstar[ind_b,:]    # state of this FP
            q_bfp       = q[ind_b]          # speed of this FP

            # # 4. Also identify any previously detected FP are near this candidate FP -- to delete them below#
            # dists_prev   = np.linalg.norm(Xstar - xstar_bfp, ord=2, axis=1)
            # Distance_Tol = 1
            # # (if already detected, then delete from existing list) #
            # indd = np.array([])  # inds to delete
            # if np.isin(ind_b, np.where(ind)[0] ): # look for ind in Fps inds   
            #     indd = np.append( indd, np.where( ind == ind_b )[0] )
            # if np.any(dists_prev < Distance_Tol):
            #     indd = np.append( indd, np.where(dists_prev < Distance_Tol)[0] )
            # Xstar   = np.delete(Xstar,indd,axis=0)
            # Q       = np.delete(Q,indd)
        
            # 4. Add to list (at the top) #
            Xstar   = np.vstack( (xstar_bfp[np.newaxis,:], Xstar) )  
            Q       = np.insert(Q, 0, q_bfp)
            num_fp  = Xstar.shape[0]    
            print('delay FP detected: item #%d, speed: %0.1f' % (item1, -np.log10(q_bfp) ) )     

    # c. Detect FP closest to baseline, 0-input state #
    if f.detect_baseline:

        # 1. Obtain activation state at default time of item 1 #
        _, _, _, X, input, _     = tk.parms_test(p, X_input, TT = 2)  # get blank input
        with torch.no_grad():
            D     = tk.run_model( model, p, input, hidden_output = True )
            Out_h = D['Out_h']   # [T,B,U]
            xbase = Out_h[p.t1,:,:].squeeze().detach().cpu().numpy()  # baseline state (here, at default item 1 time, t1)
            
        # 2. Calculate Euc dists between candidate fps and baseline state #
        cands       = np.where(q < 10**(-f.tol_q))[0]         # speed threshold to get candidates
        dists       = np.linalg.norm(xstar[cands,:] - xbase, ord=2, axis=1) 
        distinds    = np.argsort(dists)          # sort by dist, ascending  

        # 3. Identify closest candidate FP #
        if distinds.size > 0:
            ind_b       = cands[distinds[0]]            # index of FP nearest baseline
            xstar_bfp   = xstar[ind_b,:]    # state of this FP
            q_bfp       = q[ind_b]          # speed of this FP

            # 4. Also identify any  previously detected FP are near this candidate FP -- to delete
            Distance_Tol = 1
            dists_prev   = np.linalg.norm(Xstar - xstar_bfp, ord=2, axis=1)
            # (if already detected, then delete from existing list to put near front) #
            indd    = np.array([],dtype='int64')  
            ind_FP  = np.where(ind)[0]          # indices of detected FP in fps 
            if np.isin(ind_b, ind_FP ): # look for current index in Fps indices  
                indd = np.append( indd, np.where( ind_FP == ind_b )[0] )  # flag it to delete
            if np.any(dists_prev < Distance_Tol):
                indd = np.append( indd, np.where(dists_prev < Distance_Tol)[0] )
            if indd.size > 0:  # if any found, delete
                Xstar   = np.delete(Xstar,indd,axis=0)
                Q       = np.delete(Q,indd)
        
            # 4. Add to list (at the top) #
            Xstar   = np.vstack( (xstar_bfp[np.newaxis,:], Xstar) )  
            Q       = np.insert(Q, 0, q_bfp)
            num_fp  = Xstar.shape[0]
            print('baseline FP detected, speed: %0.1f' % (-np.log10(q_bfp) ) )   
        else:
            print('no baseline FP detected!' )   

    print('Detected %d unique FPs (%d candidates)' % (num_fp,B))

    # Linearize at each detected FP #
    
    # initialize outputs #
    Jac     = np.full([num_fp,p.N,p.N],np.nan,dtype='float64')
    eigval  = np.full([num_fp,p.N],np.nan,dtype='complex64')
    eigvec  = np.full([num_fp,p.N,p.N],np.nan,dtype='complex64')

    for FP in range(num_fp):

        ## Method 1 ##  (numerical)  (autograd.grad) ###################
        # print('calculating Jacobian numerically')
        fp  = torch.from_numpy( Xstar[np.newaxis,FP,:] )  # (np.newaxis is needed to allow matrix mult in model.forward)
        fp.requires_grad = True
        fp.to(p.Device)
        
        # run model at fixed point #
        input   = torch.tensor(np.zeros([1,p.U,1]), dtype=torch.float32, requires_grad=False)  # [T,U,B]
        _, h1   = model.forward(p, input, True, False, fp)
        x1      = h1[0]
        deltax  = (x1[-1,:,:].t() - fp).flatten()  # dynamical output

        # calculate jacobian #
        jacT = torch.zeros(p.N, p.N)
        for i in range(p.N):   # iterate over each "output" dimension, calculating input gradients (columns)       
            # select the output dimension #                                                                                                            
            output      = torch.zeros(p.N)                                                                                                          
            output[i]   = 1                           
            # differentiate (calc partial derivative with respect to each input dimension) #                                                                                       
            jacT[:,i]   = torch.autograd.grad( deltax, fp, grad_outputs=output, retain_graph=True )[0]
        jac = (p.tau / p.dt) * jacT.detach().numpy().T  # note 1. convert units to tau  (e.g. 10 deltat / 1 tau) 
                                                        # note 2. jac is standard: a row corresponds to partials (of a single output dim) wrt each input dim
        eval, evec  = np.linalg.eig(jac)  # (cols of evec are R eigvectors)

        # ## Method 2 ##  (analytic)   (for tanh nonlinearity) ####################
        # print('calculating Jacobian analytically')
        # Jrec = model.J.cpu().detach().numpy()
        # jac = np.zeros((p.N,p.N))  # [i,j,FP]
        # for xx in range(p.N):
        #     for yy in range(p.N):
        #         if xx == yy:
        #             kronecker = 1
        #         else:
        #             kronecker = 0
        #         jac[xx,yy] = -kronecker + Jrec[xx,yy] * tanh_deriv(Xstar[FP,yy])
        # eval, evec = np.linalg.eig(jac.squeeze())    

        # # (sanity check) #
        # if np.imag(np.sum(jac[:])) != 0:
        #     print('jacobian is not real?')
        # else:
        #     jac = np.real(jac)

        # install #
        eigval[FP,:]     = eval
        eigvec[FP,:,:]   = evec   # cols are R eigvecs
        Jac[FP,:,:]      = jac     
        
    # Detect (slowest) Saddle + Attractors #
    stable          = np.all(  np.real( eigval ) < 0 , axis=1 )    # stable FPs
    reals           = np.isclose( 0, np.imag( eigval ) )           # (indices of all pure real eigvals)
    imags           = np.invert( reals )           # (indices of all pure real eigvals)
    unstables       = np.real( eigval ) > 0                        # (indices of all unstable eigvals)
    real_unstable   = np.sum( np.logical_and( reals, unstables )  , axis=1 ) # no. of real & unstable for each FP
    imag_unstable   = np.sum( np.logical_and( imags, unstables )  , axis=1 ) 
    real_unst_1     = real_unstable == 1
    imag_unst_0     = imag_unstable == 0
    Saddles         = np.where(np.logical_and(real_unst_1,imag_unst_0))[0]   # saddle is FP w/ single real-unstable mode
    Attractors      = np.where(stable)[0]               # saddle is FP w/ single real-unstable mode

    # Output #
    Fps             = fixedpoints(f)
    Fps.num         = num_fp    # number of detected FPs
    Fps.xstar       = Xstar     # [FP,N]
    Fps.q           = Q         # [FP]
    Fps.Jac         = Jac       # [FP,N,N]
    Fps.eigval      = eigval    # [FP,N]
    Fps.eigvec      = eigvec    # [FP,N,N]  R eigvecs
    Fps.Saddles     = Saddles
    Fps.Attractors  = Attractors

    return Fps    # fps: from all optimizations, Fps: unique 


class fixedpoints(object):
    def __init__(self,parameters):
        self.num         = None
        self.xstar       = None
        self.q           = None
        self.Jac         = None
        self.eigval      = None
        self.eigvec      = None
        self.parameters  = parameters

def tanh_deriv(x):
    return 1-(np.tanh(x))**2

# Plot Histogram of q for each FP ###
def plot_histogram_fp_speeds(fps,Fps,qmin=5,qmax=10):

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    import nn_plot_functions   as npf

    plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(1500,20,700,400) 
    vals   = -np.log10(fps.q)
    vals[vals > 999999] = 20    # (sometimes get inf) replace with 20
    plt.hist(vals,bins=200,color='k')
    plt.xlabel('-log(speed)')
    plt.ylabel('# FPF initializations')
    for pp in range(Fps.num):
        val = -np.log10(Fps.q[pp])
        if val > 999999:
            val = 20
        plt.plot([val,val],[0,100],color='r',linewidth=2)
    # colorbar #
    npf.colorbarplot('jet',qmin,qmax,'-log(speed)')  # plot colorbar

#### FP colors ####
clrs = ['#ff6c03','k','k','k','y','c','slategrey','bisque','darkorange','plum','deepskyblue','salmon','peru','cadetblue','hotpink','k','k','k']
#####################

# Plot eigenvalues of all detected Fps #
def plot_eigenvalues(p, Fps):

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(200,400,2000,500) 
    for f in range(Fps.num):
        if f >= 16:
            print('not plotting more than 12 eigvals')
            break
        else:
            clr = clrs[f]

        ax = plt.subplot(2,8,f+1)

        eigval = Fps.eigval[f,:]

        if 0:
            freq   = np.imag(eig)       # cycle / tau
            ax.plot([0, 0], [-4, 4], '-',color=(0.5,0.5,0.5))
            ax.set_ylabel('Imaginary')   
        else: 
            freq   = np.imag(eigval)  * ( p.d1 * p.dt / p.tau) / (2*np.pi)  # cycle / delay
            ax.plot([0, 0], [-2.5, 2.5], '-',color=(0.5,0.5,0.5))
            # minreal = np.min(np.real(eigval))-0.1
            # maxreal = np.max(np.real(eigval))+0.1
            minreal = -2
            maxreal = 2
            ax.plot([minreal, maxreal], [-0.5, -0.5], ':',color=(0.8,0.8,0.8), alpha=1)
            ax.plot([minreal, maxreal], [+0.5, +0.5], ':',color=(0.8,0.8,0.8), alpha=1)
            ax.plot([minreal, maxreal], [-1, -1], '-',color=(0.5,0.5,0.5), alpha=0.5)
            ax.plot([minreal, maxreal], [+1, +1], '-',color=(0.5,0.5,0.5), alpha=0.5)
            ax.set_ylabel('Imag (cyc/delay)')
    
        ax.scatter( np.real(eigval),  freq,  alpha=1,  color=clr, s=30 )
        ax.plot([0, 0], [-2.5, 2.5], '-', linewidth=2, color=(0.5,0.5,0.5))
        # ax.set_aspect('equal', 'box')
        ax.set_xlim([minreal,maxreal])
        ax.set_ylim([-1.5,1.5])
        ax.set_yticks(np.arange(-1.5,2,0.5))
        ax.set_yticklabels(['-1.5','-1','-0.5','0','0.5','1','1.5'])
        ax.set_xticks(np.arange(minreal,maxreal+0.5,0.5))
        ax.set_xticklabels(['-2','','-1','','0','','1','','2'])
        ax.set_xlabel('Real')
        speed = -np.log10(Fps.q[f])
        plt.title('%d \n %0.1f' % (f,speed),fontweight='bold')
 
    # fig.subplots_adjust(hspace=1,vspace=1)    
    fig.tight_layout()     


def plot_eig(p, Neurals, Variant_plot=[0,1]):

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    # l: [eigvals, models]
    nummodels = Neurals['l'].shape[1]
    numeigs   = Neurals['l'].shape[0]

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(50+Variant_plot[0]*500,100,1000,500) 

    plt.suptitle('LSQ linear dynamics\ncollected across RNNs', fontsize=24, fontweight='bold')

    ax = plt.subplot(1,2,1)

    for m in range(Neurals['Model_variant'].size):

        if ~np.isin(Neurals['Model_variant'][m],np.array(Variant_plot)):
            continue

        clr = np.random.rand(3)
        eig = Neurals['l'][:,m]

        freq   = np.imag(eig)  * ( p.d1 * p.dt / p.tau) / (2*np.pi)  # cycle / delay
        real   = np.real(eig)
    
        ax.scatter( real,  freq,  alpha=0.02, marker='o', facecolors='k', linewidth=None, s=90 )

    ax.set_ylabel('Oscillatory frequency\n(1/delay)')


    if 0:
        minreal = -2
        maxreal = 2
        ax.plot([minreal, maxreal], [-0.5, -0.5], ':',color='k', alpha=.25)
        ax.plot([minreal, maxreal], [+0.5, +0.5], ':',color='k', alpha=.25)
        ax.plot([minreal, maxreal], [-1, -1], '-',color='k', alpha=0.25)
        ax.plot([minreal, maxreal], [+1, +1], '-',color='k', alpha=0.25)
        ax.set_ylim([-1,1])
        ax.set_yticks(np.arange(-1,1.5,0.5))
        ax.set_yticklabels(['-1','-0.5','0','0.5','1'])
        ax.set_aspect(2/1)    
        ax.plot([0, 0], [-1, 1], '-',color='k',alpha=0.25)
    else:
        minreal = -2
        maxreal = 2
        ax.plot([minreal, maxreal], [-0.5, -0.5], ':',color='k', alpha=.25)
        ax.plot([minreal, maxreal], [+0.5, +0.5], ':',color='k', alpha=.25)
        ax.plot([minreal, maxreal], [-1, -1], '-',color='k', alpha=0.25)
        ax.plot([minreal, maxreal], [+1, +1], '-',color='k', alpha=0.25)
        ax.set_ylim([-1.5,1.5])
        ax.set_yticks(np.arange(-1.5,2,0.5))
        # ax.set_yticklabels(['-1','-0.5','0','0.5','1'])
        ax.set_aspect(2/1.5)    
        ax.plot([0, 0], [-1.5, 1.5], '-',color='k',alpha=0.25)

    ax.plot([-2, 2], [0,0], '-',color='k',zorder=0,alpha=0.25)

    ax.set_xlim([minreal,maxreal])
    ax.set_xticks(np.arange(minreal,maxreal+0.5,0.5))
    ax.set_xticklabels(['-2','','-1','','0','','1','','2'])
    ax.set_xlabel('Exponential growth/decay')
    # plt.title('%d \n %0.1f' % (f,speed),fontweight='bold')
    plt.show()
    title = 'Linear fit (black) to RNN data\n\n'
    plt.title(title,fontweight='bold',fontsize=12)

    # fig.subplots_adjust(hspace=1,vspace=1)    
    fig.tight_layout()    





def find_unstable_modes(Fps):  # Find (unstable) Complex modes #
    
    osc_l = []   # [FP][M]
    osc_v = []   # [FP][R/I,N,M]
    real_l = []  # [FP][M]
    real_v = []  # [FP][1,N,M]
    line_l = []  # [FP][1]   # currently detecting only 1 mode -- the one nearest stability line
    line_v = []  # [FP][1,N,1]

    for type in range(2):

        for ff in range(Fps.num):   # iterate through each detected FP

            # real and oscillatory #

            inds_unstable = np.real(Fps.eigval[ff,:]) > 0

            if type == 0:
                inds_type = np.imag(Fps.eigval[ff,:]) == 0   # pure real (exp growth)
            elif type == 1:
                inds_type = np.imag(Fps.eigval[ff,:]) > 0    # complex

            # 1. Identify via eigenvalues #
            inds    = np.logical_and(inds_unstable,inds_type)
            inds    = np.where(inds)[0]                                       #  
            inds    = inds[ np.flip ( np.argsort(np.real(Fps.eigval[ff,inds])) ) ]  # sort by Real part, big to small
            M       = inds.size         #  No. of identified modes #
            l       = Fps.eigval[ff,inds]  # lambda

            # 2. Get eigenvectors #
            N = Fps.eigvec.shape[1]
            v   = np.zeros((2,N,M))  # [R/I,N,M]  R/I: Real/Imag, M: mode
            for m in range(M):
                re      = np.real( Fps.eigvec[ff,:,inds[m]] ) 
                im      = np.imag( Fps.eigvec[ff,:,inds[m]] )
                # if type == 1:  # if complex, then orthogonalize 
                #     reim    = np.concatenate((re[:,np.newaxis],im[:,np.newaxis]),axis=1)
                #     ri,_    = np.linalg.qr(reim)
                #     re,im   = (ri[:,0],ri[:,1])
                v[0,:,m] = re
                v[1,:,m] = im 

            if type == 0:
                real_l.append( l )      
                real_v.append( v ) 
            elif type == 1:
                osc_l.append( l )      
                osc_v.append( v ) 


            # line attractor #

            absreal         = np.abs(np.real(Fps.eigval[ff,:]))  # ind closest to 0
            nonosc          = np.where(np.imag(Fps.eigval[ff,:]) == 0)[0]
            ind             = np.argmin(absreal[nonosc])  # internal index of nearest-0 & real mode
            ind_line        = nonosc[ind]

            l               = Fps.eigval[ff,ind_line]
            v               = np.zeros((2,N,1))  # [R/I,N,M]  R/I: Real/Imag, M: mode
            re              = np.real( Fps.eigvec[ff, :, ind_line ] ) 
            im              = np.imag( Fps.eigvec[ff, :, ind_line ] )
            v[0,:,0]        = re
            v[1,:,0]        = im 
            line_l.append( l )      
            line_v.append( v )             

    Fps.real_l = real_l
    Fps.real_v = real_v
    Fps.osc_l = osc_l
    Fps.osc_v = osc_v
    Fps.line_l = line_l
    Fps.line_v = line_v
    
    return Fps



def pca_project_fp(fps,Fps,pca):  # Project FP and Eigvectors into PC space ##

    N = fps.xstar.shape[1]  # no. dimensions

    # project fixed points #    
    fps.xstar_pc = pca.transform(fps.xstar)
    Fps.xstar_pc = pca.transform(Fps.xstar)

    # project Osci
    osc_v_pc = []   # [FP][R/I,N,M]
    for FP in range(Fps.num):   # iterate each FP  
        if Fps.osc_v[FP].size > 0:  # check if are any oscillatory modes
            M      = Fps.osc_v[FP].shape[2]  #  No. of oscillatory modes
            vecs   = np.zeros([2,N,M]) # initialize vector pair
            for m in range(M): # iterate this FP's unstable oscillatory modes
                vecs[:,:,m] = np.matmul(Fps.osc_v[FP][:,:,m], np.transpose(pca.components_))
                vecs[:,:,m] = vecs[:,:,m] / np.linalg.norm(vecs[:,:,m],ord=2)  # normalize
            osc_v_pc.append( vecs )
        else:
            osc_v_pc.append( np.array([]) )

    # project Real
    real_v_pc = []   # [FP][R/I,N,M]
    for FP in range(Fps.num):   # iterate each FP  
        if Fps.real_v[FP].size > 0:  # check if are any unstable real modes
            M      = Fps.real_v[FP].shape[2]  #  No. of unstable real modes
            vecs   = np.zeros([2,N,M]) # initialize vector pair  (real,imag) - if real only, then imag is merely 0s
            for m in range(M): # iterate this FP's unstable oscillatory modes
                vecs[:,:,m] = np.matmul(Fps.real_v[FP][:,:,m], np.transpose(pca.components_))
                vecs[:,:,m] = vecs[:,:,m] / np.linalg.norm(vecs[:,:,m],ord=2)  # normalize
            real_v_pc.append( vecs )
        else:
            real_v_pc.append( np.array([]) )

    # project Line
    line_v_pc = []   # [FP][R/I,N,M]
    for FP in range(Fps.num):   # iterate each FP  
        if Fps.line_v[FP].size > 0:  # check if are any unstable real modes
            M      = Fps.line_v[FP].shape[2]  #  No. of unstable real modes
            vecs   = np.zeros([2,N,M]) # initialize vector pair  (real,imag) - if real only, then imag is merely 0s
            for m in range(M): # iterate this FP's unstable oscillatory modes
                vecs[:,:,m] =  np.matmul(Fps.line_v[FP][:,:,m], np.transpose(pca.components_))
                vecs[:,:,m] = vecs[:,:,m] / np.linalg.norm(vecs[:,:,m],ord=2)  # normalize
            line_v_pc.append( vecs )
        else:
            line_v_pc.append( np.array([]) )

    
    Fps.real_v_pc   = real_v_pc
    Fps.osc_v_pc    = osc_v_pc
    Fps.line_v_pc   = line_v_pc

    return fps, Fps


def plot_fps_3D(ax,qmin,qmax,fps,Basis,pcp):  # plot detected FP in PC space
    
    import nn_plot_functions   as npf
    
    clrs = npf.colorbarmapper(-np.log10(fps.q),qmin,qmax,'jet')         # get colormap colors
    fps_proj = project_data( fps.xstar, Basis, True )                   # project into basis
    ax.scatter(fps_proj[:, pcp[0]], fps_proj[:, pcp[1]], fps_proj[:,pcp[2]], zorder=20, s=25, marker='o', c=clrs.squeeze(), alpha=1)


def plot_FP_3D(ax, Fps, Basis, pcp):  # plot detected FP in PC space
    for FP in range(Fps.num):
        if FP >= 16:
            break
        else: 
            clr = clrs[FP]

        if FP == 0:
            lw = 6
            size = 320
        else:
            lw = 3
            size  = 150

        dat = project_data( Fps.xstar[np.newaxis,FP,:], Basis, True).squeeze()
        x = dat[pcp[0]]
        y = dat[pcp[1]]
        z = dat[pcp[2]]
        ax.scatter(x,y,z, zorder=20, s=size, marker='x', c=clr, linewidth=lw, alpha=1)

        # xfp,yfp,zfp = ( Fps.xstar_pc[FP,pcp[0]], Fps.xstar_pc[FP,pcp[1]], Fps.xstar_pc[FP,pcp[2]] )        
        # ax.scatter(xfp,yfp,zfp, zorder=20, s=150, marker='x', c=clr, linewidth=3, alpha=1)
 
def plot_FP_2D(ax, Fps, Basis, c1, c2):  # plot detected FP in PC space
    for FP in range(Fps.num):
        if FP >= 12:
            clr = 'k'
        else:
            clr = clrs[FP]
        dat = project_data( Fps.xstar[np.newaxis,FP,:], Basis, center=True).squeeze()
        ax.scatter( dat[c1], dat[c2], zorder=20, s=150, marker='x', c=clr, linewidth=3, alpha=1)

def plot_FP_modes_2D(ax,Fps,FP_mode_toplot,Basis,c1,c2):  

    SCALE = 400
    import nn_plot_functions   as npf
    components = np.array([c1,c2])

    for FP in FP_mode_toplot:

        if FP >= 12:
            clr = 'k'
        else:
            clr = clrs[FP]

        # the FP location #
        center = project_data( Fps.xstar[np.newaxis,FP,:], Basis, True).squeeze()
        # ax.scatter( center[c1] , center[c2], zorder=20, s=150, marker='x', c=clrs[FP], linewidth=3, alpha=1)
        ax.scatter( center[c1] , center[c2], zorder=20, s=500, marker='x', c='k', linewidth=3, alpha=1)

        # Fps             = fixedpoints(f)
        # Fps.num         = num_fp    # number of detected FPs
        # Fps.xstar       = Xstar     # [FP,N]
        # Fps.q           = Q         # [FP]
        # Fps.Jac         = Jac       # [FP,N,N]
        # Fps.eigval      = eigval    # [FP,N]
        # Fps.eigvec      = eigvec    # [FP,N,N]         
        
        # identify Pure Real modes #
        inds = np.where( np.logical_and( np.real(Fps.eigval[FP]) > 0, \
                                         np.imag(Fps.eigval[FP]) == 0 ) )[0]   # no. of Real-only modes for this FP
        M = inds.size
        for m in range(M):
            print('plotting unstable real mode #%d' % (m))
            clr = np.array([M-m,M-m,M-m])/(M-m+1)
            i   = inds[m]
            eigvec = Fps.eigvec[np.newaxis,FP,:,i]
            vector = project_data( eigvec, Basis, False).squeeze()

            out = npf.vecref(  center[components],  +vector[components]  , SCALE)        # + direction
            ax.plot(out[0],out[1], color=clr,linewidth=2)
            out = npf.vecref(  center[components],  -vector[components]  , SCALE)        # - direction
            ax.plot(out[0],out[1], color=clr,linewidth=2)

def plot_FP_modes_2D(ax,Fps,FP_mode_toplot,Basis,c1,c2):  

    SCALE = 400
    import nn_plot_functions   as npf
    components = np.array([c1,c2])

    for FP in FP_mode_toplot:

        if FP >= 12:
            clr = 'k'
        else:
            clr = clrs[FP]

        # the FP location #
        center = project_data( Fps.xstar[np.newaxis,FP,:], Basis, True).squeeze()
        # ax.scatter( center[c1] , center[c2], zorder=20, s=150, marker='x', c=clrs[FP], linewidth=3, alpha=1)
        ax.scatter( center[c1] , center[c2], zorder=20, s=500, marker='x', c='k', linewidth=3, alpha=1)

        # Fps             = fixedpoints(f)
        # Fps.num         = num_fp    # number of detected FPs
        # Fps.xstar       = Xstar     # [FP,N]
        # Fps.q           = Q         # [FP]
        # Fps.Jac         = Jac       # [FP,N,N]
        # Fps.eigval      = eigval    # [FP,N]
        # Fps.eigvec      = eigvec    # [FP,N,N]         
        
        # identify Pure Real modes #
        inds = np.where( np.logical_and( np.real(Fps.eigval[FP]) > 0, \
                                         np.imag(Fps.eigval[FP]) == 0 ) )[0]   # no. of Real-only modes for this FP
        M = inds.size
        for m in range(M):
            print('plotting unstable real mode #%d' % (m))
            clr = np.array([M-m,M-m,M-m])/(M-m+1)
            i   = inds[m]
            eigvec = Fps.eigvec[np.newaxis,FP,:,i]
            vector = project_data( eigvec, Basis, False).squeeze()

            out = npf.vecref(  center[components],  +vector[components]  , SCALE)        # + direction
            ax.plot(out[0],out[1], color=clr,linewidth=2)
            out = npf.vecref(  center[components],  -vector[components]  , SCALE)        # - direction
            ax.plot(out[0],out[1], color=clr,linewidth=2)


def plot_FP_modes_2D_pca(ax,Fps,FP_toplot,pc_x,pc_y):  # OUTDATED since in pipeline, pca is not calculated for each model

    import nn_plot_functions   as npf

    # ax: axis handle
    # Fps: object containing FP information
    # FP: which FP to plot
    # clr: color tuple
    # pcp: the 3 PCs to plot, e.g. [0,1,2]

    SCALE = 100
    pc_xy = np.array([pc_x, pc_y])
    # print(pc_xy)

    for FP in FP_toplot:

        center = Fps.xstar_pc[FP,pc_xy]
        ax.scatter(center[0],center[1], zorder=20, s=500, marker='x', c='pink', linewidth=3, alpha=1)

        if Fps.real_v_pc[FP].size > 0:
            M = Fps.real_v_pc[FP].shape[2]  # no. of Real-only modes for this FP
            for m in range(M):
                clr = np.array([M-m,M-m,M-m])/(M-m+1)
                vec = Fps.real_v_pc[FP][0,pc_xy,m]
                out = npf.vecref(center,+vec,SCALE)        # + dir
                ax.plot(out[0],out[1], color=clr,linewidth=2)
                out = npf.vecref(center,-vec,SCALE)        # - dir
                ax.plot(out[0],out[1], color=clr,linewidth=2)

        # if Fps.osc_v_pc[FP].size > 0:
        #     M = Fps.osc_v_pc[FP].shape[2]  # no. of Osci modes for this FP
        #     for m in range(M):
        #         clr     = np.array([M-m,M-m,M-m])/(M-m+1)
        #         real    = np.real(Fps.osc_l[FP][m])
        #         if real > 0.5:
        #             clr = 'r'
        #         else:
        #             clr = 'k'
        #         oscvec1 = Fps.osc_v_pc[FP][0,pc_xy,m]
        #         oscvec2 = Fps.osc_v_pc[FP][1,pc_xy,m]
        #         out = npf.vecref(center, oscvec1, SCALE)        # real
        #         ax.plot(out[0],out[1], color=clr,linewidth=2)
        #         out = npf.vecref(center, oscvec2, SCALE)        # real
        #         ax.plot(out[0],out[1], color=clr,linewidth=2)
        


def plot_FP_modes_3D(ax, Fps, FP_mode_toplot, Basis, pcp):  

    import nn_plot_functions   as npf

    for FP in FP_mode_toplot:

        # if FP >= 12:
        #     clr = 'k'
        # else:
        #     clr = clrs[FP]

        # the FP location #
        center = project_data( Fps.xstar[np.newaxis,FP,:], Basis, True).squeeze()
        # ax.scatter( center[c1] , center[c2], zorder=20, s=150, marker='x', c=clrs[FP], linewidth=3, alpha=1)
        # ax.scatter( center[pcp[0]], center[pcp[1]], center[pcp[2]], zorder=20, s=500, marker='x', c='pink', linewidth=3, alpha=1)

        # Fps             = fixedpoints(f)
        # Fps.num         = num_fp    # number of detected FPs
        # Fps.xstar       = Xstar     # [FP,N]
        # Fps.q           = Q         # [FP]
        # Fps.Jac         = Jac       # [FP,N,N]
        # Fps.eigval      = eigval    # [FP,N]
        # Fps.eigvec      = eigvec    # [FP,N,N]         

        SCALE           = 1.5
        nummodes        = Fps.eigval[FP,:].size
        Icount          = 0
        eigval_last     = None

        for M in range(nummodes):

            eigval = Fps.eigval[FP,M]
            
            if np.real( eigval ) < 0:   # skip plotting unstable mode
                continue
            else:

                if np.abs( np.imag( eigval ) ) > 0:   # oscillatory mode

                    # no need to plot complex conjugate #
                    if eigval_last == None:
                        eigval_last = eigval
                    elif np.all( eigval == np.conj(eigval_last) ):
                        Icount += 1 
                        continue

                    # plot #
                    clrs = ['g','b','m','k','k','k','k','k']
                    if Icount < len(clrs):
                        clr  = clrs[Icount]
                    else:
                        clr = 'k'
 
                    eigvec1 = np.real(  Fps.eigvec[np.newaxis,FP,:,M]  )
                    eigvec2 = np.imag(  Fps.eigvec[np.newaxis,FP,:,M]  )
                    vector1 = project_data( eigvec1, Basis, False).squeeze()
                    vector2 = project_data( eigvec2, Basis, False).squeeze()

                    out = npf.vecref(  center[pcp],  +vector1[pcp]  , SCALE)        # + direction
                    ax.plot3D(out[0],out[1], out[2], color=clr,linewidth=2)                
                    # out = npf.vecref(  center[pcp],  -vector1[pcp]  , SCALE)        # - direction
                    # ax.plot3D(out[0],out[1], out[2], color=clr,linewidth=2)

                    out = npf.vecref(  center[pcp],  +vector2[pcp]  , SCALE)        # + direction
                    ax.plot3D(out[0],out[1], out[2], color=clr,linewidth=2)                
                    # out = npf.vecref(  center[pcp],  -vector2[pcp]  , SCALE)        # - direction
                    # ax.plot3D(out[0],out[1], out[2], color=clr,linewidth=2)

                    Icount += 1 

                else:           # pure real mode

                    clr = '#FFB3DE'
                    # print('plotting unstable real mode' % (m))
                    
                    eigvec = Fps.eigvec[np.newaxis,FP,:,M]
                    vector = project_data( eigvec, Basis, False).squeeze()

                    out = npf.vecref(  center[pcp],  +vector[pcp]  , SCALE)        # + direction
                    ax.plot3D(out[0],out[1], out[2], color=clr,linewidth=3,alpha=1)
                    
                    out = npf.vecref(  center[pcp],  -vector[pcp]  , SCALE)        # - direction
                    ax.plot3D(out[0],out[1], out[2], color=clr,linewidth=3,alpha=1)
                
                eigval_last = eigval


# def pca_basic(D): # data: [samples,features]

#     # C               = np.cov(D, rowvar=False)
#     C               = np.matmul(D.T,D)
#     evals, evecs    = np.linalg.eig(C)    
#     ii              = np.argsort(evals)[::-1]
#     evals           = evals[ii]
#     pcs             = evecs[:,ii]  # eigenvectors are columns
#     var_exp         = evals / np.sum(evals) 

#     return pcs, var_exp   # pcs have components in columns




#### Oscillatory study  #############

class linearbasis(object):

    def __init__(self, matrix, c, topdim = None):
        self.c          = c                      # [N]   center
        self.matrix     = matrix                 # [N,N]
        self.topdim     = topdim                   
    
    def transform(self, data, center, inverse=False):
        # data:         [samples,N]
        # center:       True/False to center the data
        c = self.c if center == True else 0
        if not inverse:   # forward
            out = np.matmul( data - c, self.matrix )  # data: [samp,N], matrix [N,M]
        else:             # inverse
            out = np.matmul( data, np.linalg.inv(self.matrix) ) + c
        return out
    
    def filter(self, data, center):   # project to basis, eliminate 
        c = self.c if center == True else 0             # center
        data2 = np.matmul( data - c, self.matrix )      # project to new basis
        data2[:,self.topdim:] = 0                       # zero-out all but top dims
        out = np.matmul( data2, np.linalg.inv(self.matrix) ) + c    # project back
        return out
        

def PCA_filter_matrix(Data, pca, dim_keep=np.array([0]) ):
    # Data is list of data matrices
    # pca: scikit learn PCA object
    # dim_filt: numpy array of PCs to filter out
    N       = Data[0].shape[1]
    inds    = np.invert( np.isin( np.arange(N) ,  dim_keep )  )  
    out     = []
    for TT in range(len(Data)):
        dat   = Data[TT]
        datap = project_data(dat,   pca,  center=True)    
        datap[:,inds] = 0           # zero-out all but top dims
        dataf = project_data(datap, pca, center=True, inverse=True)
        out.append(dataf)
    return out




def project_data( data, basis, center, inverse=False , filter=False):  # xx
    
    if isinstance(basis, linearbasis):   # transform matrix
        if filter is False:
            out           = basis.transform( data, center, inverse )  # standard
        else:
            out           = basis.filter( data, center )
            out           = basis.transform( out, center, inverse )  # standard
    else:                               # scikit learn  (e.g. PCA) 
        if center:
            if not inverse:
                out       = basis.transform(data)
            else:
                out       = basis.inverse_transform(data)
        else:
            matrix = np.transpose(basis.components_)
            if not inverse:
                out       = np.matmul( data, matrix )      
            else:  
                out       = np.matmul( data, np.linalg.inv( matrix ) )      
    return out 
 

def Linearbasis_3D( center, v1, v2, v3 ):   #  Orthogonalize v1 & v2, instantiate linearbasis Object
    N = int( v1.size )      # ambient dimensionality
    v       = np.concatenate( ( v1[:,np.newaxis], v2[:,np.newaxis], v3[:,np.newaxis], np.random.rand( N, N-3 ) ) , axis=1 )
    w, _    = np.linalg.qr(v)  # qr orthogonalize
    basis   = linearbasis( w, center, topdim = 3 )   # set up basis
    return basis

def Linearbasis_2D( center, v1, v2 ):   #  Orthogonalize v1 & v2, instantiate linearbasis Object
    N = int( v1.size )      # ambient dimensionality
    v       = np.concatenate( ( v1[:,np.newaxis], v2[:,np.newaxis], np.random.rand( N, N-2 ) ) , axis=1 )
    w, _    = np.linalg.qr(v)  # qr orthogonalize
    basis   = linearbasis( w, center, topdim = 2 )   # set up basis
    return basis

def Linearbasis_1D( center, v1 ):   #  Orthogonalize v1 & v2, instantiate linearbasis Object
    N = int( v1.size )   # ambient dimensionality
    v1 = v1.squeeze()
    # QR #
    v       = np.concatenate( ( v1[:,np.newaxis], np.random.rand( N, N-1 ) ) , axis=1 )
    w, _    = np.linalg.qr(v)
    basis   = linearbasis( w, center, topdim = 1 )   # set up basis
    return basis    

def Get_Oscillatory_Mode( Lsp, FP_osc ):
    # Lsp:      linearized system parameters, dictionary from Define_linearized_system
    # FP_osc:   the index in Lsp of the FP being referenced
    #
    # returns the FP coordinate + the two oscillatory real axes
    Jac             = Lsp['Jac'][:,:,FP_osc]    # Jacobian of this FP       
    xstar           = Lsp['xstar'][:,FP_osc]    # state of this FP
    ind_osc         = np.min( np.where( Lsp['l_inds'][FP_osc] )[0]  ) # ind of one of the two conjugate oscillatory modes
    l_osc           = Lsp['l'][FP_osc][ind_osc]   # osc eigval
    o1              = np.real( Lsp['R'][:, ind_osc, FP_osc] )   # osci axis 1
    o2              = np.imag( Lsp['R'][:, ind_osc, FP_osc] )   # osxi axis 2 
    return xstar, o1, o2




def Project_Data_Tensor( Data, basis, Xs, filter=False ):   # Project N-D data into 2D oscillatory linearbasis

    # Data:   [T,N,B]

    data   = []     # projected data

    for TT in range(len(Data)):    
        
        T  = Xs[TT].shape[0]   # get time duration from each time/input condition
        B  = Xs[TT].shape[2]   # get # of conditions from each time/input condition
        N  = basis.c.size

        # # (if specified) #
        # if filter:
        #     dat = basis.filter( Data[TT], center=True )
        # else:
        #     dat = Data[TT]
        
        dat = Data[TT]
        
        # Center & Project # 
        rawdata = project_data( dat, basis, center=True )

        # (check dimension) #
        if len( rawdata.shape ) == 1:
            rawdata = rawdata[:,np.newaxis]

        # Initialize #
        data.append( np.zeros([T,N,B])  )    # [T,2,B]
        
        # Parse #
        for mm in range(B):
            a = mm * T
            b = (mm+1) * T
            data[TT][:,:,mm] = rawdata[a:b,:]
    
    return data    


def Project_Data_Matrix( Data, basis, filter=False ):   # Project N-D data into 2D oscillatory linearbasis

    # Data:             [samp,N]
    data   = []
    for TT in range(len(Data)):     
        d = Data[TT]
        if filter:      # filter top dimensions if specified
            d = basis.filter(Data[TT],center=True)
        dat = project_data( d, basis, center=True)
        data.append( dat )    # [B,N]
        
    return data   


def Project_Simulated_Data( Data, basis, label_strings ):  # runs model on multiple Time/Input conditions, stores activity

    # project the data #
    data_matrix               = Project_Data_Matrix( Data['data'], basis )
    data_tensor               = Project_Data_Tensor( Data['data'], basis, Data['Xs'] )

    # install into Data dict #
    Data[label_strings[0]]    = data_matrix         # [TT][samples,N_basis]
    Data[label_strings[1]]    = data_tensor         # [TT][T,N_basis,B] 
    Data[label_strings[2]]    = basis               #  linearbasis object



## Input vectors ###

def calc_input_vectors(p,model,X_input):

    # Returns input activation (feedforward) from neural model

    with torch.no_grad():        
        B            = p.n_items  # number of items
        X            = np.zeros([1, p.U, B])    # [T,U,B]  
        for ii in range(p.n_items):
            X[:,:,ii]    = X_input[ii,:]        # 1st item             
            input        = torch.Tensor(np.transpose( X , (0,2,1) ))        # input: [T,B,U]
            D = tk.run_model( model, p, input, hidden_output = True )
            Out_hi       = D['Out_h']   # [T,B,U]      
    
    inpvec = Out_hi.data.detach().numpy().squeeze()

    return inpvec


def plot_input_3D(p,ivec):
    MAXV            = 10  
    fig  = plt.figure(constrained_layout=True,figsize=(7,7))
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(40,200,700,700)
    ax   = plt.axes(projection="3d")
    ax   = Axes3D(fig)
    for ii in range(p.n_items):
        prop    = .75*(ii+1)/(p.n_items)    # proportion of full color intensity                
        clr = (prop,.75,prop) 
        ax.plot3D([0,ivec[ii,0]], \
                    [0,ivec[ii,1]], \
                    [0,ivec[ii,2]], color=clr, linewidth=1.5, marker='o',alpha=1)
    ax.scatter(0,0,0, color=(0,.65,0), marker='o',s=150) 
    plt.show()
    ax.grid(False)
    ax.set_xlim3d(-MAXV,MAXV)
    ax.set_ylim3d(-MAXV,MAXV)
    ax.set_zlim3d(-MAXV,MAXV)
    ax.scatter(0,0,0, color='k', marker='o',s=120)         
    ax.set_xlabel('C0',fontsize=16,fontweight='bold')
    ax.set_ylabel('C1',fontsize=16,fontweight='bold')
    ax.set_zlabel('C2',fontsize=16,fontweight='bold')
    xclr,yclr,zclr = (.95,.97,.99)    # axis plane
    ax.w_xaxis.set_pane_color((xclr, xclr, xclr,xclr))
    ax.w_yaxis.set_pane_color((yclr, yclr, yclr,yclr))
    ax.w_zaxis.set_pane_color((zclr, zclr, zclr,zclr))
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.zaxis.set_ticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

# Print dimensionality #
def print_dimensionality(X):
    # X: [samples x features]
    from sklearn.decomposition import PCA
    pca = PCA()    
    pca.fit( X )                        
    varexp = pca.explained_variance_ratio_
    str = np.array2string( np.round(varexp[:5]*100,decimals=1) )
    print( 'Dimensions, var exp: %s' % (str) )    


# Planar (2D) Flow field #

def plot_flowfield( ax, grid, p, oscbasis, model_lin, arrowsize = 1 ):
    
    # i. Construct 2D grid (in plane) #
    n, gmin, gmax  = grid   # no. of divisions, gmin, gmax: min + max in 2D space
        # initialize #
    s2      = np.zeros([(n+1)**2, p.N])     # 2D state      [s,N]
    v2      = np.zeros([(n+1)**2, p.N])     # 2D flow       [s,N]
    sN      = np.zeros([(n+1)**2, p.N])   # ND state      [s,N]
    vN      = np.zeros([(n+1)**2, p.N])   # ND flow       [s,N]
        # mesh #
    x = np.linspace(gmin, gmax, n+1)
    y = np.linspace(gmin, gmax, n+1)
    xx, yy = np.meshgrid(x, y) 
    s2[:,0] = xx.ravel()  # convert grid to coordinate pair #
    s2[:,1] = yy.ravel()

    # ii. Transform 2D to ND #
    sN     = project_data( data = s2, basis = oscbasis, center=True, inverse=True )

    print_dimensionality(sN)

    # (prelim) get blank input (1 time step)
    B      = (n+1) ** 2  #  no. of grid points
    u      = tk.get_blank_input(1, p.U, B)

    # iii. Run linear system w/ Initial Conditions (ND to ND)#
    with torch.no_grad():
        x0      = torch.tensor( np.transpose(sN) , dtype=torch.float32, requires_grad=False )   # initial condition
        D       = tk.run_linmodel( model_lin, p, u, True, x0 )   # zero input + initial hidden state
        x1      = D['Out_x'][-1,:,:]            # [T,B,N]
    vN = x1 - x0.t().detach().cpu().numpy()  # [B,N]  flow vector

    print_dimensionality(x1)

    # v. Transform (these directions) back to 2D subspace #
    v2     = project_data( data = vN, basis = oscbasis, center=False )  # don't center since these are directions
    xx_dir = v2[:,0]  
    yy_dir = v2[:,1]

    # vi. Plot arrows
    ax.quiver( xx.ravel(), yy.ravel(), xx_dir, yy_dir , color=(.6,.6,.6))
    #   quiver(x, y, u, v)



def plot_flowtest( ax, data2, p, oscbasis, linmodel , t , arrowsize = 1 ):
    
    # data2: [T,2,B]    single time point
    # s2:    [B,2]      single time point
    # t:                time point (to plot flow)
    # sN:    [B,2]            N-D state

    # i. Convert the 2D data to coordinate pairs (xx,yy)
    xx      = np.transpose( data2[t,0,:] ).ravel()
    yy      = np.transpose( data2[t,1,:] ).ravel()

    # ii. Transform the data from 2D >> ND #
    s2      = np.transpose( data2[t,:,:] )  # [B,2]
    sN      = project_data( data = s2, basis = oscbasis, center=True, inverse=True )
    sN      = np.transpose( sN )   # [N,B]
    s2_     = project_data( np.transpose(sN), oscbasis, center=True )
        
    # iii. Run linear system (ND >> ND) #
        # (prelim) get blank input (i.e. for 1 time step)
    numT       = 1  
    B          = sN.shape[1]
    u          = tk.get_blank_input(numT,p.U,B)
    with torch.no_grad():
        x0      = torch.tensor( sN , dtype=torch.float32, requires_grad=False ) # [N,B] initial condition
        D       = tk.run_linmodel( linmodel, p, u, True, x0 )              # 
        x1      = D['Out_x'][-1,:,:]              # [T,B,N] or [B,N]
        # x2      = D['Out_x'][-1,:,:]            # [T,B,N] or [B,N]
        # x1      = D['Out_x'][-2,:,:]            # [T,B,N]

    # iv. Calculate flow vector in ND #
    vN = x1 - x0.t().detach().cpu().numpy()        # N-D (flow) vector   [B,N]  
    # v1 = x2 - x1  # [B,N]  directions in ND space

    # v. Transform flow vector from ND >> 2D  #
    v2 = project_data(data = vN, basis = oscbasis, center=False)  # no need to center since these are directions
    xx_dir = v2[:,0]  
    yy_dir = v2[:,1]

    # vi. Plot arrows #
    ax.quiver( xx, yy, xx_dir, yy_dir, scale=arrowsize, color='g')

    # (plot) #
    s2__         = project_data(  np.transpose(x0), oscbasis, center=True )
    ax.scatter( s2[:,0], s2[:,1], 400, color=(1,0,0),alpha=0.3)                     # red dot
    ax.scatter( s2_[:,0], s2_[:,1], 200, color=(0,0,1),alpha=0.3)                   # blue dot
    ax.scatter( s2__[:,0], s2__[:,1], 600, color=(.8,.2,.5),alpha=0.3,marker='x')   # magenta 'x'
    s2_1         = project_data(  x1, oscbasis, center=True )  # predicted
    ax.scatter( s2_1[:,0], s2_1[:,1], 200, color=(.6,.6,1),alpha=0.3)                   # blue dot



def plot_flowfield_simple( ax, grid, p, oscbasis, model_lin, arrowsize = 1 ):
    
    # i. Construct 2D grid (in plane) #
    n, gmin, gmax  = grid   # no. of divisions, gmin, gmax: min + max in 2D space
        # initialize #
    s2      = np.zeros([(n+1)**2, p.N])     # 2D state      [s,N]
    v2      = np.zeros([(n+1)**2, p.N])     # 2D flow       [s,N]
    sN      = np.zeros([(n+1)**2, p.N])   # ND state      [s,N]
    vN      = np.zeros([(n+1)**2, p.N])   # ND flow       [s,N]
        # mesh #
    x = np.linspace(gmin, gmax, n+1)
    y = np.linspace(gmin, gmax, n+1)
    xx, yy = np.meshgrid(x, y) 
    s2[:,0] = xx.ravel()  # convert grid to coordinate pair #
    s2[:,1] = yy.ravel()

    # ii. Transform 2D to ND #
    sN     = project_data( data = s2, basis = oscbasis, center=True, inverse=True )

    # print_dimensionality(sN)

    # (prelim) get blank input (1 time step) for running model
    B      = (n+1) ** 2  #  no. of grid points
    u      = tk.get_blank_input(1,p.U,B)

    # iii. Run linear system w/ Initial Conditions (ND to ND)#
    with torch.no_grad():
        x0      = torch.tensor( np.transpose(sN) , dtype=torch.float32, requires_grad=False )   # initial condition
        x1      = model_lin.forward( p, u, x0 )   # zero input + initial hidden state
    vN = x1.squeeze() - x0.detach().cpu().numpy()  # [B,N]  flow vector
    vN = np.transpose(vN)

    # print_dimensionality(x1)

    # v. Transform (these directions) back to 2D subspace #
    v2     = project_data( data = vN, basis = oscbasis, center=False )  # don't center since these are directions
    xx_dir = v2[:,0]  
    yy_dir = v2[:,1]

    # vi. Plot arrows
    ax.quiver( xx.ravel(), yy.ravel(), xx_dir, yy_dir, scale=arrowsize )
    #   quiver(x, y, u, v)



def polarcoord(data, center = (0,0) ):   # data is in form [TT][T,2,B]    where 2 is the 2D plane coords
    
    out = []
    
    for l in range(len(data)):
        
        dat = np.copy(data[l])

        T = data[l].shape[0]
        B = data[l].shape[2]
        
        for t in range(T):
            for b in range(B):
                x = data[l][t,0,b] - center[0]
                y = data[l][t,1,b] - center[1]
                r       = np.power(x**2 + y**2,0.5)     
                phi     = np.arctan2(y,x)
                dat[t,0,b] = r
                dat[t,1,b] = phi 
        out.append( dat ) 

    return out


def largest_complex(vals):
    ind = np.argmin(np.where( np.imag(vals) > 0)[0])
    return ind


def Define_linearized_system( p, Fps, FP_indices, FP_linoption, plot_eigspectra = True ):

    numFPs            = len(FP_indices)  # No. of FPs to linearize

    # Initialize outputs #
    xstars           = np.full((p.N,numFPs),     np.nan)     # fixed point activation states
    Jacs             = np.full((p.N,p.N,numFPs), np.nan)     # jacobians
    Rs               = np.full((p.N,p.N,numFPs), np.nan, dtype='complex64')     # R eigvec matrices (eigvecs are cols)
    Es               = np.full((p.N,p.N,numFPs), np.nan, dtype='complex64')     # Diagonal matrix
    Ls               = np.full((p.N,p.N,numFPs), np.nan, dtype='complex64')     # L eigvec matrix (eigvels are rows)
    ls               = []           # [FP][N]       all eigenvalues for each FP
    ls_inds          = []           # [FP][N]       indices of retained eigvals (modes)

    for ff in range(numFPs):

        FP               = FP_indices[ff]
        linoption        = FP_linoption[ff]

        fp               = Fps.xstar[FP,:]     # x activation at this FP  [N]
        Jac              = Fps.Jac[FP,:,:]     # Jacobian at this FP      [N,N]

        l, R             = np.linalg.eig(Jac)  # l: eigvals, R: right-eigvec matrix (cols are eigvecs)
        L                = np.linalg.inv(R)    # L: left-eigvec matrix (rows are eigvecs)

        E                = np.complex64(np.diag(np.zeros((p.N))))

        if linoption == 1:  # unstable

            # inds             = np.logical_and(np.real(l) > -0.2 , np.real(l) < 0 )
            inds             = np.real(l) > 0
            for x in range(np.where(inds)[0].size):  # iterate eigvals
                ind = np.where(inds)[0][x]           # index of eigval
                E[ind,ind] = l[ind]                  # set new Diag matrix w/ eigvalue
            Jac2             = np.float32(np.matmul(R,np.matmul(E,L)))  # Reconstitute Jacobian

        elif linoption == 2: # osci only
            
            if 1:   # perpendicular
                inds1             = np.logical_and(np.abs(np.imag(l)) > 2.5, \
                                                np.abs(np.imag(l)) < 4 )
                inds2             = np.abs(np.imag(l)) > 0
                inds3             = np.real(l) > 0
                inds = np.logical_and(inds1,np.logical_and(inds2,inds3))
            elif 0:   # oblique
                inds1             = np.real(l) > 0
                inds2             = np.logical_and( np.abs(np.imag(l)) > .4, np.abs(np.imag(l)) < .6)
                inds = np.logical_and(inds1,inds2)
            else:   # unstable + oscillatory
                inds1             = np.abs(np.imag(l)) > 0
                inds3             = np.real(l) > 0
                inds = np.logical_and(inds1,inds3)

            for x in range(np.where(inds)[0].size):
                ind = np.where(inds)[0][x]
                E[ind,ind] = l[ind]

            Jac2             = np.float32(np.matmul(R,np.matmul(E,L)))  # mode-reduced
        
        elif linoption == 3:  # Fastest freq (near-unstable) Osci mode
            lower             = np.real(l) > 0
            upper             = np.real(l) < 1  # 0.07
            l_nearunstable    = l[np.logical_and(lower,upper)]   

            freq_delay        = np.imag(l)  * ( p.d1 * p.dt / p.tau) / (2*np.pi)  # cycle / delay
            l_osc_fastest     = l_nearunstable[ np.argmax(np.imag(l_nearunstable)) ]
            nearest_half      = np.argmin( np.abs( 0.5 - freq_delay[np.logical_and(lower,upper)]) )

            l_osc             = l_nearunstable[ nearest_half ]
            # l_osc             =  l_osc_fastest 


            inds1             = np.isclose( np.imag(l), np.imag(l_osc) )        # mode 1
            inds2             = np.isclose( np.imag(l), np.imag(-l_osc) )       # mode 2
            inds = np.logical_or(inds1,inds2)
            for x in range( np.where(inds)[0].size ):
                ind           = np.where(inds)[0][x]
                E[ind,ind]    = l[ind]
            Jac2             = np.float32( np.matmul( R, np.matmul(E,L) ) )  # mode-reduced
        
        elif linoption == 4:  # largest Osci, largest Real
            l_unstable        = l[np.real(l) > 0]   
            l_osc0            = l_unstable[np.argmax(np.imag(l_unstable))]
            
            l_real0           = l_unstable[np.argmax(np.real(l_unstable))]     # 1st biggest real
            l_real1           = l_unstable[np.argsort(np.real(l_unstable))][-2]  # 2nd biggest real
            l_real2           = l_unstable[np.argsort(np.real(l_unstable))][-3]  # 3rd biggest real


            inds1             = np.isclose(np.imag(l),np.imag(l_osc0))
            inds2             = np.isclose(np.imag(l),np.imag(-l_osc0))
            inds3             = np.isclose(np.real(l),np.real(l_real1))
            inds4             = np.isclose(np.real(l),np.real(l_real2))
            inds0             = np.isclose(np.real(l),np.real(l_real0))

            inds = np.logical_or(inds1,inds2)  # oscis
            # inds = np.logical_or(inds,inds3)   # real
            # inds = np.logical_or(inds,inds4)   # real
            inds = np.logical_or(inds,inds0)   # real

            for x in range(np.where(inds)[0].size):
                ind = np.where(inds)[0][x]
                E[ind,ind] = l[ind]
            Jac2             = np.float32(np.matmul(R,np.matmul(E,L)))  # mode-reduced

        elif linoption == 5: # all unstable except largest Real
            unstable          = np.real(l) > 0
            l_unstable        = l[np.real(l) > 0]   
            l_real0           = l_unstable[np.argmax(np.real(l_unstable))]     # 1st biggest real
            not_real0          = np.real(l) < l_real0

            inds             = np.logical_and(unstable,not_real0)

            for x in range(np.where(inds)[0].size):
                ind = np.where(inds)[0][x]
                E[ind,ind] = l[ind]
            Jac2             = np.float32(np.matmul(R,np.matmul(E,L)))  # mode-reduced

        elif linoption == 99:
            E                = np.diag(l)   # sanity check #
            Jac2             = np.float32(np.matmul(R,np.matmul(E,L)))  # mode-reduced
            inds = np.isscalar(l) 
        else:
            Jac2 = Jac
            inds = np.isscalar(l) 

        xstars[:,ff]    = fp
        Jacs[:,:,ff]    = Jac2
        Rs[:,:,ff]      = R
        Ls[:,:,ff]      = L
        Es[:,:,ff]      = E
        ls.append(l)
        ls_inds.append(inds)


    # plot reduced eigenspectrum #
    if plot_eigspectra and numFPs <= 8:
        
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Qt5Agg") # set the backend
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"
        matplotlib.rcParams['font.family'] = "sans-serif"
        import matplotlib.patches as patch
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(constrained_layout=True,figsize=(14,5))
        plt.get_current_fig_manager().window.setGeometry(100,600,2000,600) 
        clrs = ['#de5c0b','k','k','k','y','c','slategrey','bisque','darkorange','plum','deepskyblue','salmon','peru','cadetblue','hotpink','k','k','k']
        
        for f in range(numFPs):
            FP = FP_indices[f]
            ax = plt.subplot(2,8,f+1)
            eig, _   = np.linalg.eig( Fps.Jac[FP] )
            eig2, _  = np.linalg.eig( Jacs[:,:,f] )
            if 0:
                freq   = np.imag(eig)       # cycle / tau
                freq2  = np.imag(eig2)
                plt.ylabel('Imag (cyc / tau)')
                ax.plot([0, 0], [-4, 4], '-',color=(0.5,0.5,0.5))
            else: 
                freq   = np.imag(eig)  * ( p.d1 * p.dt / p.tau) / (2*np.pi)  # cycle / delay
                freq2  = np.imag(eig2) * ( p.d1 * p.dt / p.tau) / (2*np.pi)  
                plt.ylabel('Delay (cyc / delay) ')
                ax.plot([0, 0], [-2.5, 2.5], '-',color=(0.5,0.5,0.5))
                minreal = np.min(np.real(eig))-0.1
                maxreal = np.max(np.real(eig))+0.1
                ax.plot([minreal, maxreal], [-0.5, -0.5], ':',color=(0.8,0.8,0.8), alpha=0.8)
                ax.plot([minreal, maxreal], [+0.5, +0.5], ':',color=(0.8,0.8,0.8), alpha=0.8)
                ax.plot([minreal, maxreal], [-1, -1], '-',color=(0.5,0.5,0.5), alpha=0.5)
                ax.plot([minreal, maxreal], [+1, +1], '-',color=(0.5,0.5,0.5), alpha=0.5)
            plt.scatter( np.real(eig),  freq,  alpha=0.4,color=clrs[FP])
            plt.scatter( np.real(eig2), freq2, marker= 'x',alpha=0.8,color='k')
            plt.xlabel('Real')
            speed = -np.log10(Fps.q[FP])
            plt.title('%d \n %0.1f' % (FP,speed),fontweight='bold')

        # fig.subplots_adjust(hspace=1,vspace=1)    
        fig.tight_layout()  

    # output #
    Lsp = {}  
    Lsp['numFPs']      = numFPs
    Lsp['xstar']       = np.float32(xstars)
    Lsp['Jac']         = np.float32(Jacs)
    Lsp['R']           = Rs
    Lsp['E']           = Es
    Lsp['L']           = Ls
    Lsp['l']           = ls
    Lsp['l_inds']      = ls_inds

    return Lsp 


def PCA_filter(X, topdims=10 ):  

    # Transforms to PC, filters for top PCs, then converts back to original axes #
    #   (maintains center of data)
    #
    # X: 2D or 3D array where axis=1 has features to reduce dimensions
    # topdims: top PCs to keep

    if np.ndim(X) == 2:
        X_cat = X
    elif np.ndim(X) == 3:
        dim_a = X.shape[0]
        dim_c = X.shape[2]
        N     = X.shape[1]
        X_cat = np.full((dim_a*dim_c,N),np.nan)
        c     = 0
        for aa in range(dim_a):
            for cc in range(dim_c):
                X_cat[c,:] = X[aa,:,cc]
                c+=1

    pca = PCA( )    
    pca.fit( X_cat )

    # Xmean = np.mean(X_cat, axis=0)

    if np.ndim(X) == 2:
        X_pc        = pca.transform(X)              # transform data to PC space
        X_pc        = X_pc[:,(topdims+1):] = 0      # filter smaller PCs
        X_filtered  = pca.inverse_transform(X_pc)   # transform back to original axes
    elif np.ndim(X) == 3:  
        X_pc        = np.full(X.shape, np.nan)    
        X_filtered  = np.full(X.shape, np.nan) 
        for cc in range(dim_c):
            X_pc[:,:,cc]            = pca.transform( X[:,:,cc] )
            X_pc[:,(topdims+1):,cc] = 0
            X_filtered[:,:,cc]      = pca.inverse_transform(X_pc[:,:,cc])

    return X_filtered, pca

def PCA_inverse(X, pca ):  

    if np.ndim(X) == 3:
        dim_a = X.shape[0]
        dim_c = X.shape[2]
        N     = X.shape[1]
        X_cat = np.full((dim_a*dim_c,N),np.nan)
        c     = 0
        for aa in range(dim_a):
            for cc in range(dim_c):
                X_cat[c,:] = X[aa,:,cc]
                c+=1
        N   = pca.components_.shape[1]  # ambient dimensionality
        X_inv  = np.full( (X.shape[0],N,dim_c), np.nan) 
        for cc in range(dim_c):
            X_inv[:,:,cc]      = pca.inverse_transform(X[:,:,cc])
    else:
        breakpoint()

    return X_inv

def PCA_top_dims(X, topdims=10 ):  

    # Transforms to PCA and truncates all but top PC dimensions
    #
    # X: 2D or 3D array where axis=1 has features to reduce dimensions
    # topdims: top PCs to keep

    if np.ndim(X) == 2:
        X_cat = X
    elif np.ndim(X) == 3:
        dim_a = X.shape[0]
        dim_c = X.shape[2]
        N     = X.shape[1]

        X_cat = np.full((dim_a*dim_c,N),np.nan)
        c     = 0
        for aa in range(dim_a):
            for cc in range(dim_c):
                X_cat[c,:] = X[aa,:,cc]
                c+=1

    pca     = PCA( n_components=topdims )    
    pca.fit( X_cat )
    # Xmean = np.mean(X_cat, axis=0)

    if np.ndim(X) == 2:
        X_filt = pca.transform(X)
    elif np.ndim(X) == 3:  
        X_filt = np.full((dim_a,topdims,dim_c), np.nan)    
        for cc in range(dim_c):
            X_filt[:,:,cc] = pca.transform( X[:,:,cc] )

    return X_filt, pca





def Plot_Dimensionality(p, pca, neff, TT = 0, model_title=''):

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    tvec = tk.get_timevector(p,TT)

    fig0 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(200,200,700,600) 

    ax   = fig0.add_subplot(3,1,1)
    ax.plot(tvec, neff, color='k', linewidth=3)
    ax.set_xlim( [ tvec[0], tvec[-1] ])
    ax.set_ylim([0,10])
    ax.set_xlabel('Time')
    ax.set_ylabel('N_eff')
    ax.set_title(model_title,fontsize=14)

    ax   = fig0.add_subplot(3,1,2)
    ax.plot(tvec, neff/np.max(neff), color='k', linewidth=3)
    ax.set_xlim( [ tvec[0], tvec[-1] ])
    ax.set_ylim([0,1])
    ax.set_xlabel('Time')
    ax.set_ylabel('N_eff')

    ax   = fig0.add_subplot(3,1,3)
    MAXPC = 10
    ax.bar(np.linspace(0,MAXPC-1,MAXPC),100*pca.explained_variance_ratio_[0:MAXPC],color='black')
    top3VE = 100*np.sum(pca.explained_variance_ratio_[0:3])
    print('Variance explained (PC1,2,3) = %d %%' % top3VE)
    ax.set_aspect('auto')
    ax.set_xlim([-0.5,MAXPC])
    plt.xticks(ticks=range(MAXPC))
    ax.set_ylim([0,100])
    ax.set_xlabel('PC #',fontsize=18)
    ax.set_ylabel('% var',fontsize=18)        

    fig0.subplots_adjust(hspace=.5)
 

### Feedforward weight studies ##########





# %% Study WEIGHTS ################### of MLP, LOGISTIC, or LINEAR #


def Unit_responses(p, F, model):
    
    import copy

    M                        = p.n_items

    INDS,_,_                 = tk.Get_trial_indices(p)
    itemmat                  = tk.trial_parser_itemnum(p)
    taskfunc                 = tk.Get_taskfunc(p)

    q                   = copy.deepcopy(p)   # copy so we can change noise level
    q.sig               = 0
    q.noise_test        = False

    ZZ      = 2      # two output neurons
    zrow     = np.full((2,M,ZZ), np.nan)                      # [itempos,M,unit]
    zmat     = np.full((M,M,ZZ), np.nan, dtype=np.float32)    # [M,M,unit]

    N      = p.N      # two output neurons
    hrow     = np.full((2,M,N), np.nan)                      # [itempos,M,unit]
    hmat     = np.full((M,M,N), np.nan, dtype=np.float32)    # [M,M,unit]

    for ITEM in [0,1,2]:
        if ITEM == 0:
            toggle      = [1,0]
        elif ITEM == 1:
            toggle      = [0,1]
        elif ITEM == 2:
            toggle      = [1,1]
        _, _, _, _, input, _     = tk.parms_test(q, F, 0, toggle)
        
        # run network #
        with torch.no_grad():
            D     = tk.run_model( model, q, input, hidden_output = True )
            z   = D['Out_z'].squeeze()   # [T,B,Z]
            h   = D['Out_h'].squeeze()   # [T,B,N]

        # get output unit activities #
        for zz in range(2):
            mat = np.reshape(z[:,zz],(M,M)) #    z[:,0]
            if ITEM == 0:
                zrow[ITEM,:,zz] = mat[0,:]
            elif ITEM == 1:
                zrow[ITEM,:,zz] = mat[:,0]                
            else:
                zmat[:,:,zz]    = mat

        # get hidden unit activities #
        for nn in range(N):
            mat = np.reshape(h[:,nn],(M,M)) #    z[:,0]
            if ITEM == 0:
                hrow[ITEM,:,nn] = mat[0,:]
            elif ITEM == 1:
                hrow[ITEM,:,nn] = mat[:,0]                
            else:
                hmat[:,:,nn]    = mat

    Out = (zrow, zmat)
    Hid = (hrow, hmat)

    return Out, Hid
    
    
    
def Plot_output_unit_responses(p, Out):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    row, mat = Out

    # Plot #
    fig0 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(80,100,600,300) 
    fig0.suptitle('Transitive Inference, Output responses', fontsize=17, weight='bold')
    
    # Single item #
    for cc in range(2):
        ax1 = fig0.add_subplot(1,2,cc+1)
        ax1.imshow(row[:,:,cc],cmap='bwr',vmin=-5,vmax=5)   # Plot 3rd test pair stim 
        ax1.set_yticks([0,1])
        ax1.set_yticklabels(['1','2'])
        ax1.set_title('Output unit %d\nactivation' % (cc),fontsize=18)
        ax1.set_xlabel('Item')
        # ax1.set_ylabel('Position')
        ax1.set_xticks(np.arange(p.n_items))
        ax1.set_xticklabels(['A','B','C','D','E','F','G'])
        fig0.subplots_adjust(hspace=1.5)
        import nn_plot_functions    as npf
        npf.colorbarplot('bwr',-5,5,'unit')  # plot colorbar
        plt.draw()     
        plt.pause(0.0001)    

    ########## 2. Scatter plot ##########
    fig1 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(80,100,1000,1000) 
    fig1.suptitle('Transitive Inference, Output unit responses', fontsize=17, weight='bold')

    import ti_functions as ti
    MAX = 5

    # row: [itempos,M,unit]
    for cc in range(2):    # 0: output 1, 1: output 2
        ax1 = fig1.add_subplot(1,2,cc+1)
        for mm in range(p.n_items):
            clr = ti.embedcolor(mm, p.n_items, 3)
            if mm == p.n_items-1:
                edgecolor='k'
            else:
                edgecolor='None'
            ax1.scatter(row[0,mm,cc],row[1,mm,cc],s=300,facecolors=clr[0],edgecolors=edgecolor,alpha=1,zorder=100)   # Plot 3rd test pair stim 
            ax1.set_xlim([-MAX,MAX])
            ax1.set_ylim([-MAX,MAX])
            ax1.set_xticks(np.arange(-MAX,MAX+1,1))
            ax1.set_yticks(np.arange(-MAX,MAX+1,1))
            ax1.set_xticklabels(['-5','','','','','0','','','','','5'])
            ax1.set_yticklabels(['-5','','','','','0','','','','','5'])
            ax1.set_title('Output %g\nactivation' % (cc), fontsize=18)
            plt.draw()     
            plt.pause(0.0001)
        ax1.plot([-MAX,MAX],[0,0],linewidth=2,alpha=0.15,color='k',zorder=0)
        ax1.plot([0,0],[-MAX,MAX],linewidth=2,alpha=0.15,color='k',zorder=1)
        ax1.set_aspect('equal', 'box')
        plt.draw()     
        plt.pause(0.0001)    


def Plot_hidden_unit_responses(p, Hid, model):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    row, mat = Hid   # the data 
                     # row: [itempos,M,unit]
    ro_weights = model.readout.weight.detach().numpy()          # get readout weights from PyTorch nn    hmat     = np.full((M,M,N), np.nan, dtype=np.float32)    # [M,M,unit]
    diffweight = np.diff( ro_weights, axis=0 ).flatten()
    I          = np.argsort( diffweight )

    ########## 1. Color plot ##########
    fig0 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(80,100,600,600) 
    fig0.suptitle('Transitive Inference, Hidden unit responses', fontsize=17, weight='bold')
    
    N_plot = 10

    # Single item #
    plotcount = 0
    for cc in [0,1,2,3,4,95,96,97,98,99]:
        if cc < 50:
            coord = 1 + plotcount*2
        else:
            coord = 2 + np.abs(95-cc)*2
        ax1 = fig0.add_subplot(5,2,coord)
        ax1.imshow(row[:,:,I[cc]],cmap='bwr')   # Plot 3rd test pair stim 
        ax1.set_xticks(range(p.n_items))
        ax1.set_yticks([0,1])
        ax1.set_yticklabels(['1','2'])
        # ax1.set_title('Unit %d response' % (cc))
        # ax1.set_xlabel('Item #')
        # ax1.set_ylabel('Item position')
        ax1.set_xticks(np.arange(p.n_items))
        if plotcount == 4 or plotcount == 9:
            ax1.set_xticklabels(['A','B','C','D','E','F','G'])
        else:
            ax1.set_xticklabels([])
        plt.draw()     
        plt.pause(0.0001)
        plotcount+=1    


    ########## 2. Scatter plot ##########
    fig0 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(400,100,1000,1000) 
    fig0.suptitle('Transitive Inference, Hidden unit responses', fontsize=17, weight='bold')

    import ti_functions as ti

    # row: [itempos,M,unit]
    for uu in range(2):    # 0: output 1, 1: output 2
        ax1 = fig0.add_subplot(1,2,uu+1)
        if uu == 0:
            inds = diffweight < 0
        else:
            inds = diffweight > 0
        for mm in range(p.n_items):
            if mm == p.n_items-1:
                edgecolor='k'
            else:
                edgecolor='None'
            clr = ti.embedcolor(mm, p.n_items, 3)
            ax1.scatter(row[0,mm,inds],row[1,mm,inds],s=200,facecolors=clr[0],edgecolors=edgecolor,alpha=0.5,zorder=100)   # Plot 3rd test pair stim 
            # ax1.set_xlim([-1,1])
            # ax1.set_ylim([-1,1])
            ax1.set_xlim([-1.25,1.25])
            ax1.set_ylim([-1.25,1.25])

            ax1.set_xticks(np.arange(-1.25,1.5,.25))
            ax1.set_yticks(np.arange(-1.25,1.5,.25))
            ax1.set_xticklabels(['','-1','','','','0','','','','1',''])
            ax1.set_yticklabels(['','-1','','','','0','','','','1',''])
            ax1.set_title('Units to output %g\n' % (uu), fontsize=18)
            plt.draw()     
            plt.pause(0.0001)
        ax1.plot([-1.25,1.25],[0,0],linewidth=2,alpha=0.15,color='k',zorder=0)
        ax1.plot([0,0],[-1.25,1.25],linewidth=2,alpha=0.15,color='k',zorder=1)
        ax1.set_aspect('equal', 'box')



def Plot_hidden_pca_responses(p, Hid, groups_toplot, COLOR_EMBED):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    from sklearn.decomposition import PCA

    M               = p.n_items
    _, mat          = Hid        # mat: [M,M,unit]
    data            = np.reshape( mat, (M*M,p.N) )
    pca             = PCA()    
    pca.fit( data )                            
    pdata           = pca.transform(data)
    INDS,_,_        = tk.Get_trial_indices(p)
    taskfunc        = tk.Get_taskfunc(p)
    
    N_plot = 10

    # Plot #
    fig0 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(80,100,1200,1600) 
    fig0.suptitle('Transitive Inference, Output responses', fontsize=17, weight='bold')

    # PC Variance explained #
    MAXPC = 10
    ax1 = fig0.add_subplot(4,2,1)
    ax1.bar(np.linspace(0,MAXPC-1,MAXPC),100*pca.explained_variance_ratio_[0:MAXPC],color='black')
    # ax1.imshow(row[:,:,cc],cmap='bwr',vmin=-10,vmax=10)   # Plot 3rd test pair stim 
    # ax1.set_yticks([0,1])
    # ax1.set_yticklabels(['1','2'])
    # ax1.set_title('Unit %d response' % (cc))
    ax1.set_xlabel('PC #')
    ax1.set_ylabel('Variance explained')
    plt.draw()     
    plt.pause(0.0001)      


    # PC projections in 2D #
    for pcs in range(7):
        PCS = ([0,1],[0,2],[1,2], [3,4],[3,5],[4,5],[6,7])
        pc = PCS[pcs]
        ax = fig0.add_subplot(4,2,2+pcs)
        for gg in groups_toplot:
            ii, clr0, mksize, _, _ = taskfunc.groupplot(gg,INDS)
            for LR in [0,1]:
                if ii[LR].size == 0:
                    continue
                for pp in ii[LR]:
                    if COLOR_EMBED == 0:
                        clr = clr0[LR]
                    elif COLOR_EMBED > 0:
                        clr = taskfunc.embedcolor(p,pp,COLOR_EMBED)
                    ax.scatter(pdata[pp,pc[0]],pdata[pp,pc[1]], linewidth=1.5,color=clr)  # trajectory
        # ax1.scatter(pdata[0,:],pdata[1,:],color='black')
        # ax1.set_yticks([0,1])
        # ax1.set_yticklabels(['1','2'])
        # ax1.set_title('Unit %d response' % (cc))
        ax.set_xlabel('PC%d' % (pc[0]))
        ax.set_ylabel('PC%d' % (pc[1]))
        ax.set_aspect('equal','box')
    fig0.subplots_adjust(hspace=.5)