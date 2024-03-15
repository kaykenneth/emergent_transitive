from __future__ import division 
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.decomposition import PCA
import time
import copy
import pickle, os

import nn_models as nm



## Common Input functions ##########################

def get_items(p):               # sensory items #       
    np.random.seed(p.seedB)
    X_input = np.random.randn( np.sum(p.n_items), p.U ) # Stim item x stim dim
    if False:  # orthogonalize
        u, s, vh    = np.linalg.svd(X_input, full_matrices=False)
        s_ones      = (np.sum(s) / p.n_items) * np.diag( np.ones(p.n_items) )
        X_input     = np.matmul(u, np.matmul(s_ones, vh)).astype(np.float32)
    print(p.seedB)
    # print(X_input[0,0])
    return X_input    # [ num items, U]

def get_blank_data(p, t_ext):   # no input (np array) #
    X = np.zeros([int(p.T + t_ext), p.U , 1])    # X: [T,U,1]
    return X     # X: [T,U,B]

def get_blank_input(T,U,B):     # no input (torch) #
    # T: number of timesteps
    # U: input dimensionality
    # B: batch/condition number
    u          = Variable( torch.zeros( [T, B, U] ), requires_grad=False)       # X0: [T,U,B]
    return u

def Get_taskfunc(p):
    if p.Task == 1:     # TI
        import ti_functions as taskfunc
    elif p.Task == 2:   # AI
        import ai_functions as taskfunc
    elif p.Task == 3:   # IRI
        import iri_functions as taskfunc
    return taskfunc

def Get_test_data(p):
    taskfunc = Get_taskfunc(p)
    return taskfunc.get_test_data    

def Get_train_data(p):
    taskfunc = Get_taskfunc(p)
    return taskfunc.get_train_data    

def Get_trial_indices(p):
    taskfunc = Get_taskfunc(p)
    INDS,trains,tests        = taskfunc.trial_parser(p)  # get important indices (for transitive inference)
    return INDS, trains, tests

def trial_parser_itemnum(p):   # Gets indices of different types of trials#
    M       = p.n_items
    XV, YV  = np.meshgrid(np.arange(M),np.arange(M))  
    itemmat = np.concatenate((YV[:,:,np.newaxis],XV[:,:,np.newaxis]),axis=2)
    return itemmat  # [M,M,2]   each matrix has identity of item1, item2

## Common Output functions ###########################

def parse_output(p,Out_z):
    if p.Z == 1:    # MSE, 1-output  (binary choice at 0 and 1)
        Choiceval     = Out_z[-1,:,1]
        RT            = None
        if p.Model >= 7:
            Choice        = (Choiceval > 0) * 1.0          
        else:
            if p.Task == 3:
                Choice        = (Choiceval > 0) * 1.0       
            else:
                Choice        = (Choiceval > 0.5) * 1.0         
    elif p.Z == 2:   # Cross-entropy      (binary choice, two readout neurons)
        Choiceval      = Out_z[-1,:,1] - Out_z[-1,:,0]   # Choice 0 (TI: 1st spatial position is "larger")
                                                         # Choice 1 (TI: 2nd spatial position is "larger")
        Choice         = (Choiceval > 0) * 1.0              
        RT             = None
    elif p.Z == 3:  # MSE, 3-output
        Choice, RT   = nm.Parse_3_readouts(p,Out_z)  # Choiceval_2 is output neuron w/ non-linearity
        if 0:    # last time step
            Choiceval      = Out_z[-1,:,1] - Out_z[-1,:,0]   # Choice unit activity, 1st - 2nd, final timestep   # same (0) - diff (1)
        else:    # last half of choice period
            lasthalf       = int(p.d1/2)
            Choiceval      = np.mean(Out_z[-lasthalf:,:,1] - Out_z[-lasthalf:,:,0], axis=0)   # Choice unit activity, 1st - 2nd, final timestep   # same (0) - diff (1)
    return Choice, Choiceval, RT

## Get Inputs/Outputs for Various Trial Formats ("Time-Trial" (TT)) ####################################

def get_timevector(p,TT):
    T, t1, t2, _, _, _  = parms_test( p, None, TT )
    tvec                = np.arange(T)  #* p.dt
    return tvec

def parms_test(p, F, TT, toggle=[1,1]):

    # Default parms #
    fname       = 'test_input'    # 'test_input' or 'blank_input
    t_ext       = 0
    offsets     = [0,0]
    # toggle      = [1,1]   # [stim1,stim2]  if 0, zeros out input

    if TT == 0:             # Normal dur + input
        offsets     = [0,0]  #[int(p.d1/4),0]
    elif TT == 1:           # Extended dur + input
        t_ext       = 200
        offsets     = [0,0]  #[int(p.d1/4),0]
    elif TT == 2:           # Normal time, blank input
        fname       = 'blank_input'
        t_ext       = 0
    elif TT == 3:           # Extended time, blank input
        fname       = 'blank_input'
        t_ext       = 100
    elif TT == 4:           # Normal time, only 1st input
        offsets     = [0,0]
        toggle      = [1,0]
    elif TT == 5:           # Extended time, only 1st input
        offsets     = [0,0]
        toggle      = [1,0]
        t_ext       = 100
    elif TT == 6:           # Move 1st item forward maximally
        t_ext       = 0
        offsets     = [-1,0]
    elif TT == 7:           # Move 2nd item back maximally
        t_ext       = 0
        offsets     = [0,-1]

    elif TT == 8:           # Move 1st item forward 75% of way
        t_ext       = 0
        offsets     = [int(p.d1*0.67),0]
    elif TT == 9:           # Move 2nd item back 75% of delay
        t_ext       = 0
        offsets     = [0,int(p.d1*0.67)]

    elif TT == 10:           # no Stimulus
        t_ext       = -int(p.d1 + p.d1*0.5)
        offsets     = [0,0]
        toggle     = [1,0]
    elif TT == 11:           # No stimulus
        t_ext       = -p.d1
        offsets     = [0,0]
        toggle     = [1,0]

    elif TT == 12:          # 50%    
        offsets     = [int(p.d1*0.5),0]
        t_ext       = 0
    elif TT == 13:       
        offsets     = [0,int(p.d1*0.5)]
        t_ext       = 0

    elif TT == 14:          # 33%    
        offsets     = [int(p.d1/3),0]
        t_ext       = 0
    elif TT == 15:       
        offsets     = [0,int(p.d1/3)]
        t_ext       = 0

    elif TT == 16:           # (dummy condition)
        offsets     = [0,0]
        t_ext       = 50

    # Get parms + Input (tensor) #
    if F is not None:

        if fname == 'test_input':

            get_test_data = Get_test_data(p)
            
            if p.Z == 3:
                X, ztarg, _, _, j1, j2  = get_test_data(p, F['X_input'], t_ext, offsets, toggle)       # X: [T,U,B]
            elif p.Z == 2:
                X, ztarg, _, _, _, _  = get_test_data(p, F['X_input'], t_ext, offsets, toggle)       # X: [T,U,B]
                j1, j2  = (0,0)
                
        elif fname == 'blank_input':
            X               = get_blank_data(p,t_ext)       # X: [T,U,B]
            ztarg           = []
            j1, j2          = (0,0)

        input           = torch.Tensor(np.transpose( X , (0,2,1) ))    # input: [T,B,U]

    else:

        X     = None
        input = None
        ztarg = []
        j1 = j2 = 0

            
    T               = p.T + t_ext
    t1              = p.t1 + j1
    t2              = p.t2 - j2

    return T, t1, t2, X, input, ztarg


## Loading and saving trained models ###############################

def get_model( Location, localdir, Load_model, Save_model, jobname, TID):
    
    import sys

    if Location == 1:       # local  (directory: ./)

        from init import params, singleout, genfilename 
        p            = params()            
        omit_jobname = True
        TID          = -1  # dummy value

    elif Location == 2:     # cluster (directory: ./jobname)

        paramsdir = '%s/%s' % (localdir,jobname)
        sys.path.insert(0, paramsdir)
        from init import params, singleout, genfilename
        p = singleout(TID = TID)
        omit_jobname = False    

    filename                = genfilename(p, localdir, omit_jobname)
    F, (Dataset,fps_all)    = train_or_load(p, filename, Load_model)
    
    if Save_model:
        writefile(filename,[F])

    fps = None
    Fps = None

    if fps_all != None:
        print('(New model file) grabbing FPF outputs')
        fps                = fps_all[0]
        Fps                 = None
        # fps                = fps_all[i]
        # Fps                = Fps_all[i]
    else:
        print('(Non-cluster or Old file) running postprocessing on older')
        # F                               = pp.run_Performance(p, F, TT_list = range(16))       # updates F to have 'performance' 
        # fps                             = pp.run_FPF_search(p,F)
        # Fps                             = pp.run_FPF_detect(p,F,fps,pca = [])

    return p, F, filename, Dataset, fps, Fps


def select_model( p, F, Model_option ):

    # Select model  #
    numfrozen          = len(F['models_frozen'])
    if Model_option == -1:            # 'Best' model
        # print('Selecting model: Best')
        model          = F['model']
        model_epoch    = F['epoch']
        i              = numfrozen
    elif Model_option == -2:              # Lowest loss
        # print('Selecting model: Lowest Loss')
        model          = F['model_lowest']
        model_epoch    = F['epoch_lowest']
        i              = numfrozen + 1
    elif Model_option == -3:            # Task-proportion > 0.1 model
        # print('Selecting model: Minimal task-loss proportion')
        losses         = F['log_train_losses']
        epochs_frozen  = np.arange(0, numfrozen*p.nEpochsfreeze, p.nEpochsfreeze)
        taskprop       = losses[epochs_frozen,1] / losses[epochs_frozen,0]
        inds           = taskprop > 0.1
        tasklosses     = losses[epochs_frozen[inds],1]
        i              = np.argmin(tasklosses)
        model_epoch    = epochs_frozen[i]
        model          = F['models_frozen'][i]        
    elif Model_option == -4:            # Earliest performer model
        if 'model_earliest' in F.keys():
            model_epoch    = F['epoch_earliest']
            model          = F['model_earliest']
        else:               # if earliest performer not available, take lowest-loss
            model_epoch    = F['epoch']
            model          = F['model']
        i              = -1
    elif Model_option == -5:            # Earliest performer model
        model_epoch    = F['epoch_earliest_noise']
        model          = F['model_earliest_noise']
        i              = -1
    else:                               # Manually chosen model
        # i              = int( Model_option / p.nEpochsfreeze )
        # model          = F['models_frozen'][i] 
        # model_epoch    = i * p.nEpochsfreeze
        i = numfrozen-1
        model          = F['models_frozen'][i] 
        model_epoch    = i * p.nEpochsfreeze

    ep_last             = np.max(np.where( F['log_train_losses'][:,0] != -1))
    # print(ep_last)

    model_title         = 'Model option: %d // frozen: %d // epoch: %d' % (Model_option, i, model_epoch)

    # print('Model is from training epoch #%d' % (model_epoch))

    return model, model_epoch, i, model_title


def train_or_load(p, filename, load):

    F       = None   # model (rnn)
    Dataset = None   # data from simulation
    fps     = None   # fixed pts, candidates
    Fps     = None   # fixed pts, detected

    if load:
        out = readfile(filename)
        if len(out) == 0:
            F = train_model(p)   # train 
        elif len(out) == 1:
            F = out[0]           # load
        else:
            F       = out[0]           # load
            Dataset = out[1]
            fps     = out[2]
    else:
        F = train_model(p)   # train 

    return F, (Dataset,fps)

def readfile(filename, extension = ".p", silent=False):
    filename = filename + extension
    if not silent:
        print('Reading file (pickle) '+filename+'...')
    if os.path.isfile(filename):
        if extension == ".p":
            data = pickle.load(open(filename,"rb"))
        elif extension == ".txt":
            data = np.loadtxt(filename)
    else:
        data = []
    return data

def writefile(filename, results, extension = ".p"):
    filename = filename + extension
    print('Writing file (pickle) '+filename+'...')
    if extension == ".p":
        pickle.dump(results, open(filename,"wb"))
    elif extension == ".txt":
        np.savetxt(filename, results)

#### Plot losses ########################################

def plot_training_losses(F, p, filename, model_epoch):

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    log_train_losses    = F['log_train_losses']
    if 'epoch' in F:
        model_ep            = F['epoch']
        epoch_best          = F['epoch']
        epoch_lowest        = F['epoch_lowest']
        # ep_last             = F['epoch']  # may have trained more epochs, but this is lowest loss
        ep_last             = np.max(np.where( log_train_losses[:,0]  != -1))
    else:
        model_ep            = F['loss_min_epoch']
        epoch_best          = model_ep
        epoch_lowest        = model_ep
        ep_last             = np.max(np.where( log_train_losses[:,0]  != -1))

    xmax                = 1000*np.ceil(ep_last/1000)
    numfrozen           = len(F['models_frozen'])
    if numfrozen > 0:
        epochs_frozen       = np.arange(0,numfrozen*p.nEpochsfreeze,p.nEpochsfreeze)
    else:
        epochs_frozen   = []

    ##### print selected model's losses #######
    loss        = log_train_losses[model_epoch,0]
    taskloss    = log_train_losses[model_epoch,1]
    task        = 100*log_train_losses[model_epoch,1] / loss
    weight      = 100*log_train_losses[model_epoch,2] / loss
    meta        = 100*log_train_losses[model_epoch,3] / loss
    if weight   != weight:   weight  = 0 # string check
    if meta     != meta:     meta    = 0    # string check
    print('(Selected model) Task loss: %0.2f, Task %%: %d, Weight %%: %d, Meta %%: %d' % (taskloss,task,weight,meta) )

    ##### plot ######################
    figT  = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(150,0,500,1400) 
    plt.show(block=False)
        
    # plot total loss #
    axx = figT.add_subplot(7,1,1)
    axx.cla()
    axx.plot(log_train_losses[0:ep_last,0],color='k',zorder=100)    
    axx.axvline(x=epoch_best, color = [.75,.75,.75], zorder=0)   # chosen model
    axx.axvline(x=epoch_lowest, color = [.85,.85,.85], zorder=0)
    axx.axvline(x=model_epoch, color = [1,.8,.8], linewidth=5,zorder=0)  # red line

    axx.set_xlim([0,xmax])
    axx.set_yscale('log')
    axx.set_ylabel('Total loss')
    axx.set_title(filename,fontsize=11)
    ymin, ymax = axx.get_ylim()
    axx.vlines(x=epochs_frozen,ymin=ymin-.5,ymax=ymax+5, color=[.95,.95,.95],zorder=1)
    axx.set_ylim([ymin,ymax])

    # plot each loss term #
    #   # red:      Task error
    #   # green:    Weights (input/output), raw value without regularization coefficient
    #   # blue:     Metabolic (activity), raw value without regularization coefficient
    axes = [[],[],[],[],[],[],[]]
    clrs = ['','r','g','b']
    labels = ['','Error','Weight L2','Firing L2']
    for X in range(1,4):  # the 3 individual loss terms
        axes[X] = figT.add_subplot(7,1,1+X)
        if log_train_losses.shape[1] > 4:  # latest model file has larger log matrix
            if X == 2 or X == 3:
                index = X+2   # plots the raw value (not multiplied by regularization coefficient)
            else:
                index = X
        else:
            index = X
        values = log_train_losses[0:ep_last,index]
        axes[X].plot(values,color=clrs[X],linewidth=2)    
        axes[X].set_xlim([0,xmax])
        axes[X].set_ylabel(labels[X])
        if ~np.all(np.isnan(values[1:-1])):
            axes[X].set_yscale('log')
        axes[X].axvline(x=epoch_best, color = [.7,.7,.7], zorder=0)   # chosen model
        axes[X].axvline(x=epoch_lowest, color = [.9,.9,.9], zorder=0)   # chosen model
        axes[X].axvline(x=model_epoch, color = [1,.8,.8], linewidth=5,zorder=0)

        ymin, ymax = axes[X].get_ylim()
        axes[X].vlines(x=epochs_frozen,ymin=ymin-.5,ymax=ymax+5, color=[.95,.95,.95],zorder=1)
        axes[X].set_ylim([ymin,ymax])

    # plot proportions of loss (3 sources) #
    axes[4] = figT.add_subplot(7,1,5)    # Hidden neuron activity, example trial
    axes[4].cla()
    total = log_train_losses[0:ep_last,0]
    proportion =  log_train_losses[0:ep_last,1:4] / total[:,None]
    axes[4].plot(proportion[:,0], color=clrs[1] , alpha=0.6)  # error (task-based)
    axes[4].plot(proportion[:,1], color=clrs[2] , alpha=0.6)  # weight reg
    axes[4].plot(proportion[:,2], color=clrs[3] , alpha=0.6)  # activity reg
    axes[4].plot(proportion[:,2], color=clrs[3] , alpha=0.6)  # activity reg
    axes[4].plot(proportion[:,2], color=clrs[3] , alpha=0.6)  # activity reg
    axes[4].axvline(x=epoch_best, color = [.7,.7,.7], zorder=0)   # chosen model
    axes[4].axvline(x=epoch_lowest, color = [.9,.9,.9], zorder=0)   # chosen model
    axes[4].axvline(x=model_epoch, color = [1,.8,.8], linewidth=5, zorder=0)
    ymin, ymax = axes[4].get_ylim()
    axes[4].vlines(x=epochs_frozen,ymin=0,ymax=ymax+5, color=[.95,.95,.95],zorder=1)
    axes[4].set_ylim([0,ymax])

    # axes[4].set_yscale('log')
    # axes[4].set_ylim([1e-6,2])

    axes[4].set_ylim([0,1.05])
    axes[4].set_xlim([0,xmax])
    axes[4].set_ylabel('Loss proportion')

    plt.draw()     
    plt.show()
    plt.pause(0.0001)

    return figT, axes


def plot_training_performance(F, p, fig, axes, model_epoch):

    # print('hi')
    perf = F.get('performance',[])

    if len(perf) == 0:
        print('performance not available')
        return

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    perf                = F['performance']          #   [numfrozen]
    log_train_losses    = F['log_train_losses']
    ep_last             = np.max(np.where( log_train_losses[:,0]  != -1))
    # ep_last             = F['epoch']
    print('ep_last: %d' % (ep_last))
    xmax                = 1000*np.ceil(ep_last/1000)

    numfrozen           = len(F['models_frozen'])
    if numfrozen > 0:
        epochs_frozen       = np.arange(0,numfrozen*p.nEpochsfreeze,p.nEpochsfreeze)
    else:
        epochs_frozen   = []

    axes[5] = fig.add_subplot(7,1,6)    # Hidden neuron activity, example trial
    axes[5].cla()

    # frozen models #
    axes[5].plot(epochs_frozen,perf[0:numfrozen,0, 0 ],color='k',zorder=100)      # training perf
    axes[5].plot(epochs_frozen,perf[0:numfrozen,0, 1 ],color='r',zorder=100)      # test perf
    axes[5].scatter(epochs_frozen,perf[0:numfrozen,0, 0 ],color='k',zorder=100)      # training perf
    axes[5].scatter(epochs_frozen,perf[0:numfrozen,0, 1 ],color='r',zorder=100)      # test perf
    axes[5].vlines(x=epochs_frozen,ymin=-0.5,ymax=100+5, color=[.95,.95,.95],zorder=1)

    # earliest model #
    if 'epoch_earliest' in F.keys():
        axes[5].scatter(F['epoch_earliest'], perf[numfrozen+2,0, 0 ],s=120,  color='k',zorder=100)      # training perf
        axes[5].scatter(F['epoch_earliest'], perf[numfrozen+2,0, 1 ],s=40,  color='r',zorder=100)      # test perf

    # chosen model #
    axes[5].axvline(x=model_epoch,color = [1,.8,.8], linewidth=5, zorder=0)   # red line: chosen model

    # lowest-loss model #
    axes[5].axvline( x=F['epoch_lowest'], color = [.5,.5,1], zorder=0)

    # scale #
    axes[5].set_ylim([-0.5,105])
    axes[5].set_xlim([0,xmax])
    axes[5].set_ylabel('Task performance')

    plt.draw()     
    plt.show()
    plt.pause(0.0001)

    plt.ion()
    plt.ioff()
    plt.ion()


def plot_training_pca_variance(F,p,fig,axes):

    import matplotlib
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    import nn_analyses as nna

    ep_last             = np.max(np.where(~np.isnan(F['log_train_losses'][:,0])))
    xmax                = 1000*np.ceil(ep_last/1000)

    numfrozen           = len(F['models_frozen'])
    if numfrozen > 0:
        epochs_frozen       = np.arange(0,numfrozen*p.nEpochsfreeze,p.nEpochsfreeze)
    else:
        epochs_frozen   = []

    varexps = np.full( (numfrozen,2), np.nan )  # [modelnum,<pc1-3,pc1-6>]
    for f in range(numfrozen):
        Data, _, _, _       = run_model( p, F['models_frozen'][f], F['X_input'], TT_list = [0] , verbose = False)
        pca                 = nna.PCA_Calc(Data, Reference_TT = 0)
        varexps[f,0] = np.sum(pca.explained_variance_ratio_[0:3]) * 100
        varexps[f,1] = np.sum(pca.explained_variance_ratio_[0:6]) * 100

    axes[5] = fig.add_subplot(7,1,7)    # Hidden neuron activity, example trial
    axes[5].cla()
    axes[5].plot(epochs_frozen,varexps[0:numfrozen,0 ],color='k',zorder=100)      # 
    # axes[5].plot(epochs_frozen,varexps[0:numfrozen,1 ],color=[.5,.5,.5],zorder=100)      
    axes[5].plot(epochs_frozen,varexps[0:numfrozen,0 ],color='k',zorder=100)      # 
    # axes[5].plot(epochs_frozen,varexps[0:numfrozen,1 ],color=[.5,.5,.5],zorder=100)     
    axes[5].axvline(x=F['epoch'],color = [1,.8,.8], linewidth=5, zorder=0)   # chosen model
    axes[5].axvline(x=F['epoch_lowest'], color = [.85,.85,.85], zorder=0)
    axes[5].vlines(x=epochs_frozen,ymin=-0.5,ymax=100+5, color=[.95,.95,.95],zorder=1)

    axes[5].set_ylim([85,102])
    axes[5].set_xlim([0,xmax])
    axes[5].set_ylabel('Var Exp, PC1-3')

    plt.draw()     
    plt.show()
    plt.pause(0.0001)


##### Commmon helper function ######

def format_training_input(p,X_input,i1,i2):
    T,t1,t2,dur,jit = (p.T,p.t1,p.t2,p.dur,np.asarray(p.jit))
    if p.T > 1:     # Temporal (RNN)
        X       = np.zeros((p.T, p.U))          # X: [time, dim_input]
        if jit == 0:    # no jitter
            j1 = t1
            j2 = t2
        else:           # jitter
            prop    = 1 - jit                        # max allowed proportional reduction of d1
            dmin    = np.round(p.d1 * prop)       # min allowed delay
            if 0:  # Cueva style
                j1vals  = np.arange( t1, t2-dmin, 1)    
                j1      = int( j1vals[ np.random.randint(j1vals.size) ] )    
                j2vals  = np.arange( j1+dmin, t2+1 , 1)     
                j2      = int( j2vals[ np.random.randint(j2vals.size) ] )
            elif 1:  # only 2nd item can move
                j1      = t1
                j2vals  = np.arange( j1+dmin, t2+1 , 1)     
                j2      = int( j2vals[ np.random.randint(j2vals.size) ] )
        # print('j1: %d, j2: %d ' % (j1,j2))
        # print('delay: %d ' % (j2-j1))
        X[j1:(j1+dur),        :]  = X_input[i1]                    # install Stim 1
        X[j2:(j2+dur),        :]  = X_input[i2]                    # install Stim 2
    elif p.T <= 1:    #  Spatial (FF model)
        # print('spatial model')
        X                         = np.zeros([1,int(p.U*2)])          # Time x Stim dims
        X[t1:(t1+dur),0:p.U]      = X_input[i1]                    # install Stim 1
        X[t1:(t1+dur),-p.U:]      = X_input[i2]                    # install Stim 2  
        j1, j2                    = (0,0)
    return X, j1, j2

def format_training_output(p,j1,j2,y):
    T,dur = (p.T,p.dur)
    if p.Z == 3:    
        Y             = np.zeros([T,3])      # [R,T]  initialize   
        randval       = np.random.rand(1)
        prop1         = np.float32(p.blank_trials)
        prop2         = np.float32(p.catch_trials)
        if randval > (prop1 + prop2):
            Y[0:j2,0:2]  = 0                      # Set Choice outputs to 0 before Item 2
            Y[0:j1, 2  ]     = p.Rest_value           # Set Rest output (unit index 2)
            if y == 1:    # 1st item is closer to "A"  
                Y[j2:,0]  = 0
                Y[j2:,1]  = p.Choice_value   # set Output cell 1 to be active
            elif y == 0:  # 2nd item is closer to "A" 
                Y[j2:,0]  = p.Choice_value   # set Output cell 0 to be active 
                Y[j2:,1]  = 0
        elif randval < prop1:  # blank trial (zero inputs)
            Y[:,:]        = 0
            Y[:   , 2  ]  = p.Rest_value           # Set Rest output (unit index 2)
            Y[:   , 0:2]  = 0                      # Set Choice outputs, pre-Choice periods
        else:                   # catch trial (1-item input)
            Y[j2:(j2+dur),:]    = 0                   # Zero out Stim 2
            Y[:   , 2  ]        = p.Rest_value        # Set Rest output (unit index 2)
            Y[:   , 0:2]        = 0                   # Set Choice outputs, pre-Choice periods
    else:
        Y = y
    return Y

def format_test_input(p, X_input, xv, yv, t_ext, offsets, toggle):

    T,t1,t2,dur,jit = (p.T,p.t1,p.t2,p.dur,np.asarray(p.jit) )

    prop    = 1 - jit              # maximum allowed proportional reduction of d1
    dmin    = np.round(p.d1 * prop)       # minimum allowed delay
    if dmin == p.d1:
        dmin = p.d1 - 1  # (if there is no jitter, assume default delay) #

    if offsets[0] == -1 and offsets[1] == -1:     # random jitter 
        j1vals  = np.arange( t1, t2-dmin, 1)    
        j1      = int( j1vals[ np.random.randint(j1vals.size) ] )    
        j2vals  = np.arange( j1+dmin, t2+1 , 1)     
        j2      = int( j2vals[ np.random.randint(j2vals.size) ] )
    elif offsets[0] == -1:   # 1st item is maximally moved forward
        j1      = int( p.d1 - dmin )
        j2      = 0
    elif offsets[1] == -1:   # 2nd item is maximally moved back
        j1      = 0
        j2      = int( p.d1 - dmin )
    else:
        j1 = offsets[0]
        j2 = offsets[1]

    t1 = t1 + j1  # input 1 jitter  
    t2 = t2 - j2  # input 2 jitter

    if p.T > 1:    # Temporal
        X       = np.zeros([int(T + t_ext), p.U , p.n_items**2])    # X: [time,dim_input,conds]
        X[t1:(t1+dur), :, :]      = np.transpose(X_input[xv])         # install item 1
        X[t2:(t2+dur), :, :]      = np.transpose(X_input[yv])         # install item 2 
        if toggle[0] == 0:          # Zero-out stim1
            X[t1:(t1+dur),:]      = 0                   
        if toggle[1] == 0:          # Zero-out stim2
            X[t2:(t2+dur),:]      = 0                   # Blank Stim 2
    elif p.T == 1:  # Spatial
        X       = np.zeros([1,int(2*p.U), p.n_items**2])
        X[0, 0:p.U, :]            = np.transpose(X_input[xv])         # install item 1
        X[0, -p.U:, :]            = np.transpose(X_input[yv])         # install item 2
        if toggle[0] == 0:          # Zero-out stim1
            X[0, 0:p.U, :]        = 0                   
        if toggle[1] == 0:          # Zero-out stim2
            X[0, -p.U:, :]        = 0                   # Blank Stim 2

    return X, j1, j2















####### %% Variable delay - Performance ###############

def Evaluate_Performance_Variable_Delay(p, model, X_input):

    n_items                     = p.n_items
    _,trains,tests              = Get_trial_indices(p)

    get_test_data = Get_test_data(p)

    # get correct choices #
    _, ztarg, _, _,_,_  = get_test_data(p, X_input, 0)       # get correct choices
    ztargmat            = ztarg[:, 0].reshape(n_items, n_items)      # Target (2D)   #

    Perfs   = np.full((2,p.d1,2),np.nan)   # [ <train,test> , delay , <t1 vs. t2>]     #100*np.mean(train_acc)
    
    # check both if change time of item 1 and item 2
    for tt in [0,1]:  # 0: change t1, 1: change t2

        offsets = [0,0]

        for j in range(p.d1):  # shorten delay

            offsets[tt] = j

            # get input #
            X, _, _, _,_,_  = get_test_data(p, X_input, t_ext=0, offsets=offsets)
            u               = torch.Tensor(np.transpose( X , (0,2,1) ))

            # run model
            with torch.no_grad():     
                D           = run_model( model, p, u, hidden_output = False )
                Out_z       = D['Out_z']   # [T,B,Z]

            # parse output
            Choice, _, _          = parse_output(p, Out_z)
            Choicemat             = Choice.reshape(n_items, n_items)

            # check performance #
            train_acc      = ztargmat[ trains  ] == Choicemat[ trains  ]
            test_acc       = ztargmat[ tests   ] == Choicemat[ tests   ]

            Perfs[0,j,tt]      = 100*np.mean(train_acc)
            Perfs[1,j,tt]      = 100*np.mean(test_acc)

    return Perfs




######### Neural MSE loss function ###############################
def neural_MSEloss(output,target,p,model,rs):

    # Inputs # 
    # output: RNN output
    # target: correct answer            (requires_grad = False)
    # p:      model parameters
    # model:  RNN model

    # Outputs #
    # loss:             total loss (with gradient graph)
    # loss_values:      loss values, for logging

    # 1. Task error      #
    error           = torch.sum( torch.pow(target - output, 2) )  / target.data.nelement() # sums over [T,B,R]: time, batch, readouts
    error_value     = error.detach()

    # 2. L2 reg of Weights #
    # model.wz: [R,N]
    # model.wu: [N,U]
    weight_out      = torch.sum( torch.pow( model.wz , 2)  ) / model.wz.data.nelement()
    weight_in       = torch.sum( torch.pow( model.wu , 2)  ) / model.wu.data.nelement()
    # reg_weight      = p.reg_weight * ( weight_in + weight_out )                          # old style (absolute coefficient)
    weight_value    = weight_out.detach() + weight_in.detach()    
    reg_weight  = p.reg_weight * ( weight_in + weight_out )  # new style (proportion of (task) error)
    # if weight_value > (1/3) * error_value:
    #     reg_weight  = ( (1/3) * error_value / weight_value) * ( weight_in + weight_out )  # new style (proportion of (task) error)

    # 3. L2 reg of Activity #
    # rs:  [T,B,N]
    activities        = torch.sum( torch.pow(rs,2) ) / rs.data.nelement()  # sum over conditions, time, and hidden units
    activities_value  = activities.detach()    
    reg_metabolic   = p.reg_metabolic * activities   # old
    # if activities_value > (1/3) * error_value:
    #     reg_metabolic   = ( (1/3) * error_value / activities_value) * activities   # new

    # Full loss function #
    loss            = error + reg_weight + reg_metabolic
 
    # Log individual loss terms #
    total           = loss.detach()
    error           = error.detach()
    weight          = reg_weight.detach()
    metabolic       = reg_metabolic.detach()
    weight_raw      = weight_in.detach() + weight_out.detach()
    metabolic_raw   = activities.detach()

    # avoids plotting errors later #
    if weight    == 0:
        weight      = float('nan')     
    if metabolic == 0:    
        metabolic   = float('nan')

    #                             0      1      2         3          4            5             #
    loss_values     = torch.tensor([total, error, weight, metabolic, weight_raw, metabolic_raw ])
    loss_values     = loss_values[np.newaxis,:]

    return loss, loss_values



########## Train model #########################################################

def train_model(p):

    if p.epochShow:  # (if plotting later)
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Qt5Agg") # set the backend
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"
        matplotlib.rcParams['font.family'] = "sans-serif"
        import matplotlib.patches as patch
        from mpl_toolkits.mplot3d import Axes3D        

    X_input             = get_items(p)   # define input
    get_train_data      = Get_train_data(p)

    S            = p.Model       # Model type
    UU           = p.U

    if S > 0:       # Feedforward models

        UU          = int(2*p.U)            

        if S == 7:
            model       = nm.Model7(p)      # MLP, 1 hidden layer
        elif S == 8:
            model       = nm.Model8(p)      # Logistic Regression
        elif S == 9:
            model       = nm.Model9(p)      # MLP, 2 hidden layers

        opt             = torch.optim.Adam( model.parameters(), lr=p.eta, weight_decay=p.L2 )  
        # opt             =  torch.optim.SGD(model.parameters(), lr=p.eta)

    elif S < 0:     # ct-RNN model
        if S == -1:
            model       = nm.Model6(p) 
        opt             = torch.optim.Adam( [ model.J, model.wz, model.wu, model.b, model.bz, model.x_ic  ], lr = p.eta )
    
    loss_min        = torch.zeros(p.nEpochs)                 

    # Select Loss function #
    if p.Z == 3:                                # MSE loss with regularizations (RNNs)
        loss        = neural_MSEloss
    elif S in [7,8,9] or p.Z == 2:              # Cross-entropy
        if hasattr(p,'classweight'):
            loss    = nn.CrossEntropyLoss( weight = p.classweight )          
        else:
            loss    = nn.CrossEntropyLoss()    
    elif p.Z == 1:                              # MSE (obsolete)
        loss        = nn.MSELoss()        

    # Initialize table that keeps track of trial type #  (item1,item2)
    if type(p.n_items) is list:
        if p.n_items[0] <= 100:
            train_table     = np.zeros((p.n_items[0],p.n_items[0]))  
        else:
            print('skipping train_table! too many items')
            train_table     = []
    else:
        train_table     = np.zeros((p.n_items,p.n_items))
    
    # Preliminaries #
    print(model)
    print('max_error = %g' % (p.max_error))
    log_train_losses = torch.full((p.nEpochs,6),-1, dtype=torch.float32)
    startTime = time.time()          # Set clock

    # (optional) Plot of Training progress #
    if p.epochShow and S < 0:
        # figT = plt.figure()
        figT, (axx, axx1, axx2) = plt.subplots(3, 1)
        # matplotlib.use("Qt5Agg")
        plt.get_current_fig_manager().window.setGeometry(150,0,300,1400) 
        plt.show(block=False)

    # (optional)  Parse curriculum
    if p.Task == 1 or p.Task == 4:
        Curriculum = p.Curriculum
    elif p.Task == 2:
        Curriculum = p.Curriculum
    elif p.Task == 3:
        Curriculum = [[]]
    opt_train     = p.opt_train_args               

    # Initialize #
    F               = {}                    # output dictionary
    F['X_input']    = copy.copy(X_input)    # store input

    models_frozen   = []
    model_lowest    = None   # model w/ lowest overall loss
    model_best      = None   # model w/ lowest loss and w/ minority of loss due to regularization
    epoch_lowest    = None
    epoch_best      = None

    # Train #
    for stage in range(len(Curriculum)):   # iterate each stage of the Curriculum

        # (curricula) specify training items in curriculum #
        if len(Curriculum[stage]) == 0: 
            curric_items = []
        else:
            curric_items = Curriculum[stage] 
        opt_train["curric_items"] = curric_items

        epochs_no_improve           = 0
        early_stop                  = False     # flag variable
        perfect_performance_flag    = False     # flag variable
        noise_performance_flag      = False
        loss_min                    = np.Inf
        num_to_freeze               = np.Inf

        for train_epoch in range(p.nEpochs):

            # Initialize target output #
            if p.Z == 1 or p.Z == 2:
                y = np.zeros([p.B,1])               # batch output    (answers)
            elif p.Z == 3:
                y = np.zeros([p.T,p.B,3])               # batch output    (answers)

            ### 1. Get batch of inputs ############
            np.random.seed(p.seedB + train_epoch)          # batch seed
            if p.B > 0:
                X = np.zeros([p.T,p.B,UU])    # batch input     [time, batch, dim_input]
                for B in range(p.B):

                    if p.Task != 3:     # TI, AI
                        XX, yy, items = get_train_data(p, X_input, opt_train)      # Single stimulus
                    elif p.Task == 3:   # IRI
                        XX, yy, items, g, pair = get_train_data(p, X_input, opt_train)      # Single stimulus

                    X[:,B,:] = XX

                    if p.Z != 3:
                        y[B,:]   = yy
                    else:
                        y[:,B,:] = yy

                    if len(train_table) != 0:
                        train_table[ items[0], items[1] ] += 1   
                        
            elif p.B == 0:  # needs to be updated for p.Z == 3
                X, y, items = get_train_data(p, X_input, opt_train)      # Single stim
                X               = X[:, np.newaxis, :]   # add dim
                y               = y[np.newaxis, :]      # add dim
                if train_table != None:
                    train_table[items[0],items[1]] += 1  

            input = torch.Tensor(X)
            input.requires_grad = False
            target          = torch.Tensor(y)           # [B,1]

            ########################################



            ### For Loss f'n, convert to right format ###
            if p.Z == 1:  # regression
                target       = target.squeeze().float()
            elif p.Z == 2:
                target       = target.squeeze().long()
            elif p.Z == 3:
                target       = target.float()

            # # # ('cpu' or 'cuda') #
            # target = target.to( p.Device    )
            # input  = input.to(  p.Device    )
            # model  = model.to(  p.Device    )

            ### 2. Run model #######################
            opt.zero_grad()              
            if S > 0:       # PyTorch models (DT-RNN, etc.)
                torch.manual_seed(p.Seed + train_epoch)              # (seed the noise within pytorch model)
                out, _ = model(p, input, noisefree = False)           # out: [T,B,R]  
            elif S < 0:     # neuro RNN
                input           = torch.transpose( torch.Tensor(X) , 1, 2)           # [T,B,U]
                torch.manual_seed(p.Seed + train_epoch)
                out, hiddens    = model.forward( p, input, noisefree = False, initrand=False )                     
                _, rs, _, _    = hiddens     #  hiddens: (xs, rs, Jrs, Wus)     get rs for metabolic regularization
            ########################################

            ############################################################
            ## (optional) truncate the loss before the choice period ####
            if 0:
                if p.Z == 3:
                    _, _, t2, _, _, _          = parms_test(p,None,0)
                    target                     = target[t2:,:,:]
                    out                        = out[t2:,:,:]
            ############################################################

            ### 3. Learn ###########################
            if p.Z == 1:
                train_loss = loss(out[-1,:,:].squeeze(), target)    # calculate MSE loss     
            elif p.Z == 2:
                train_loss = loss( out[-1,:,:], target )              # calculate Cross-entropy loss
                log_train_losses[train_epoch,0:2] = train_loss.detach() # log individual loss terms
            elif p.Z == 3:
                train_loss, loss_values  = loss( out, target, p, model, rs )     # calculate neural MSE loss     
                log_train_losses[train_epoch,:] = loss_values.detach() # log individual loss terms

            train_loss.backward()                        # calculate gradients            
            opt.step()                                   # update Model

            # if loss improved, set copy of model aside #
            lossval = train_loss.data.cpu().numpy()
            if lossval < loss_min:

                model_lowest        = copy.deepcopy(model)
                epoch_lowest        = train_epoch
                loss_min            = lossval  
                eps_no_improvement  = 0

                # check for proportion of loss #
                if p.Z == 3:
                    taskprop     = loss_values[0,1]/loss_values[0,0]   # (task loss) / (total loss)
                    if taskprop > 0.5:   # if the majority is based on task
                        model_best      = copy.deepcopy(model)
                        epoch_best      = train_epoch
                else:
                    model_best      = copy.deepcopy(model)
                    epoch_best      = train_epoch
        
            else:
                eps_no_improvement  += 1

            # (optional) Early stopping (end early if not improve loss)
            if train_epoch > p.min_train_ep and eps_no_improvement == p.n_epochs_stop:
                print('Early stopping! %d epochs without improvement' % (p.n_epochs_stop))
                early_stop = True

            # (optional) Freeze model for saving
            if p.nEpochsfreeze > 0:
                if (train_epoch%p.nEpochsfreeze == 0)  :
                    print('Freezing model @', train_epoch)
                    models_frozen.append(copy.deepcopy(model)) 
                    # models_frozen.append(copy.deepcopy(model.to(p.Device) ))  # model.to guards against grad error
                    num_to_freeze = num_to_freeze - 1

            # Check Performance ######################
            if p.check_perf:    

                with torch.no_grad(): 

                    # no noise  #
                    if p.jit == -1 or p.jit == 0:
                        TT_check = [0]         # if not jitter, just check full-length delay
                    elif p.jit <= 0.5:   
                        TT_check = [0,13]   # 0: full length, and 50% advance
                    elif p.jit > 0.5:   
                        TT_check = [0,9]   # 0: full length, and 75% advance

                    Perf = np.full( (len(TT_check),2) , np.nan)   # [tt, 0: train, 1: test]   tt is local index for time-trial condition
                    for tt in range(len(TT_check)):
                        train, test = print_model_performance(p, model, F, TT_check[tt], silent = True)  
                        Perf[tt,0]  = train
                        Perf[tt,1]  = test

                    # # (TI only) with noise level of training #
                    # if p.Task == 1:
                    #     Perf_noise = np.full( (len(TT_check)) , np.nan)   # [tt, 0: train, 1: test]   tt is local index for time-trial condition
                    #     for tt in range(len(TT_check)):
                    #         perf_noise = print_TI_performance_with_noise(p, model, X_input, TT_check[tt])  
                    #         Perf_noise[tt]  = perf_noise

                # if p.Task == 1 and (perf_noise > 90):  # perfect performance
                #     if noise_performance_flag == False:
                #         F['model_earliest_noise']  = copy.deepcopy(model)
                #         F['epoch_earliest_noise']  = train_epoch            # training epoch corresponding to the best model (task loss is low + lowest loss)
                #         noise_performance_flag = True
                
                if perfect_performance_flag == False:
                    if np.all( Perf.flatten() == 100 ) or \
                        (p.Task == 1 and p.Task_version == 1 and train == 100):  # perfect performance
                        perfect_performance_flag = True
                        F['model_earliest']  = copy.deepcopy(model)
                        F['epoch_earliest']  = train_epoch            # training epoch corresponding to the best model (task loss is low + lowest loss)
                        if p.stop_perf == 0:
                            print('Perfect Performer! early stopping.')
                            early_stop = True
                        elif p.stop_perf > 0:
                            print('Perfect Performer! freezing several more models')
                            num_to_freeze = p.stop_perf


            ##########################################

            ## Check training progress, stop if warranted ##

            # print status #
            if (train_epoch%p.nEpochsplot == 0) and train_epoch > 0:
                print("epoch = ", train_epoch)                          # print training epoch
                print( torch.mean(log_train_losses[(train_epoch-p.nEpochsplot):train_epoch,1]).numpy() )     # print running avg of error (task-based) loss                   

            if True: #S < 0:        

                # (optional) plot ongoing training progress #
                if p.epochShow and (train_epoch%p.nEpochsplot == 0) and train_epoch > 0:
                    
                    plt.clf()

                    # plot total loss #
                    axx = figT.add_subplot(7,1,1)
                    axx.cla()
                    axx.plot(log_train_losses[0:train_epoch,0],color='k')    
                    numepsplot = 200 if train_epoch < 200 else train_epoch	
                    axx.set_xlim([0,numepsplot])
                    axx.set_yscale('log')
                    axx.set_ylabel('Total loss')

                    # plot error loss # 
                    axes = [[],[],[],[],[]]
                    clrs = ['','r','g','b']
                    labels = ['','Error','Weight L2','Firing L2']
                    for X in range(1,4):  # the 3 individual loss terms
                        axes[X] = figT.add_subplot(7,1,1+X)
                        values = log_train_losses[0:train_epoch,X]
                        axes[X].plot(values,color=clrs[X],linewidth=2)    
                        axes[X].set_xlim([0,numepsplot])
                        axes[X].set_ylabel(labels[X])
                        if ~np.all(np.isnan(values[1:-1])):
                            axes[X].set_yscale('log')

                    # plot proportions of loss (3 sources) #
                    axes[4] = figT.add_subplot(7,1,5)    # Hidden neuron activity, example trial
                    axes[4].cla()
                    total = log_train_losses[0:train_epoch,0]
                    proportion =  log_train_losses[0:train_epoch,1:4] / total[:,None]
                    axes[4].plot(proportion[:,0], color=clrs[1] , alpha=0.6)  # error (task-based)
                    axes[4].plot(proportion[:,1], color=clrs[2] , alpha=0.6)  # weight reg
                    axes[4].plot(proportion[:,2], color=clrs[3] , alpha=0.6)  # activity reg
                    axes[4].set_yscale('log')
                    axes[4].set_ylim([1e-6,2])
                    axes[4].set_xlim([0,numepsplot])
                    axes[4].set_ylabel('Loss proportion')

                    # axx1 = figT.add_subplot(7,1,6)    # Output neuron activity, example trial 
                    # z = out
                    # axx1.cla()
                    # axx1.plot(range(p.T),z.detach().numpy()[:,7,:])  # [T,batch_samp,R]
                    # axx1.axvline(x = t1-1,      color="gray")        # 1st input
                    # axx1.axvline(x = t1+dur-1,  color="gray")    # 1st input
                    # axx1.axvline(x = t2-1,      color="gray")        # 2nd input
                    # axx1.axvline(x = t2+dur-1,  color="gray")    # 2nd input
                    # axx1.axvline(x = p.T,       color="gray")                                # Readout time
                    # axx1.set_xlim([0,T-1])

                    # axx2 = figT.add_subplot(7,1,7)    # Hidden neuron activity, example trial
                    # axx2.cla()
                    # axx2.plot(xs.detach().numpy()[:,:,0], alpha=0.5)
                    # axx2.set_xlim([0,T-1])

                    # plt.tight_layout()
                    plt.draw()     
                    plt.show()
                    plt.pause(0.0001)

            # Stop training if Error (task-based loss) is
            # below max_error (for running mean of nEpochsplot trials)
            if (train_epoch > p.min_train_ep):
                if ( torch.mean(log_train_losses[(train_epoch-p.nEpochsplot):train_epoch,1]).numpy() < p.max_error) \
                    or early_stop or num_to_freeze == 0:
                    print('Training Done')
                    print(loss_min)        
                    break               


        
        totaltime = (time.time() - startTime)/3600
        
        F['model']              = model_best
        F['epoch']              = epoch_best            # training epoch corresponding to the best model (task loss is low + lowest loss)

        F['model_lowest']       = model_lowest   
        F['epoch_lowest']       = epoch_lowest          # training epoch corresponding to the lowest loss model

        F['models_frozen']      = copy.deepcopy(models_frozen)
        # F['X_input']            = copy.copy(X_input)
        F['out']                = copy.copy(out)
        F['opt']                = copy.copy(opt) 
        F['loss']               = copy.copy(loss)       # loss function
     
        F['log_train_losses']   = copy.copy(log_train_losses.cpu().detach().numpy())  
        F['train_table']        = copy.copy(train_table)
        F['training_duration']  = totaltime             # total time in hours
    
    print('RNN Training, Total Time (hr): %0.2f' % (totaltime)  )
    return F




########## Run Model ### (for testing, after training) ######################################################

def run_model( model, p, input, hidden_output=True, **kwargs):

    T = p.T
    B = input.shape[1]  
    S = p.Model

    Out_z   = np.zeros([T,B,p.Z])   # [T,B,R] Readout state     

    if hidden_output:
        iterate_time = range(T)
        Out_h   = np.zeros([T,B,p.N])   # [T,B,N] Hidden state 
        Out_Jr  = np.zeros([T,B,p.N]) 
        Out_Wu  = np.zeros([T,B,p.N])   
        Out_c   = None
    else:
        iterate_time = [T-1]
        Out_h   = None
        Out_Jr  = None    
        Out_Wu  = None
        Out_c   = None

    # run model in time#
    with torch.no_grad():        # controlled execution
        if S > 0:    # Feedforward (pytorch)
            out, outh     = model(p, input, noisefree = (not p.noise_test), **kwargs)    #  out:                [time,cond,num_output]                              output linear cell (entire trial)
            Out_z[-1,:,:] = out[-1,:,:].cpu().numpy()
            if hidden_output and outh != 0:
                 Out_h[-1,:,:] = outh[0][-1,:,:].cpu().numpy()
        elif S < 0:   # RNN
            ut              = torch.transpose(input,1,2)    # [T,U,B]
            # out, hiddens    = model.forward(p, ut, noisefree = False, initrand=False)             # rawout: [T,B,R]
            out, hiddens    = model.forward(p, ut, noisefree = (not p.noise_test), initrand=False)             # rawout: [T,B,R]
            Out_z           = out.data.numpy()        
            if hidden_output:
                xs, rs, h3, h4     = hiddens   # hiddens: (xs, rs, Jrs, Wus) (also, xs, rs: [T,N,B]
                Out_h              = torch.transpose(xs,1,2)    # [T,B,N]   ** activations, not activity
                if h3.shape[2] != 1:   # checks if this is the non-linear RNN model               
                    Out_Jr          = torch.transpose(h3,1,2)  # Jr inputs
                    Out_Wu          = torch.transpose(h4,1,2)  # Wu inputs

    D = {}   # output dictionary
    D = {'Out_h':Out_h, 'Out_z':Out_z, 'Out_c':Out_c, 'Out_Jr':Out_Jr, 'Out_Wu':Out_Wu}
    return D


def run_linmodel( linmodel, p, input, hidden_output=True, *xinit):

    T = p.T
    B = input.shape[1]  

    # initialize outputs #
    Out_z   = np.zeros([T,B,p.Z])   # [T,B,R] Readout state     
    if hidden_output:
        Out_x   = np.zeros([T,B,p.N])   # [T,B,N] Hidden state  
    else:
        Out_x   = None

    # run model #
    with torch.no_grad():    
        ut              = torch.transpose(  input, 1, 2  )    # [T,U,B]
        out, hiddens    = linmodel.forward( p, ut, (not p.noise_test), False, *xinit )             # rawout: [T,B,R]
        Out_z           = out.data.numpy()        
        if hidden_output:
            xs, rs, fs, fds = hiddens   # hiddens: (xs, rs, fs) (also, xs, rs: [T,N,B]
            Out_x           = torch.transpose(xs,1,2)    # [T,B,N]   ** activations, not activity
            Out_r           = torch.transpose(rs,1,2)    # [T,B,N]   ** activations, not activity            
            Out_x           = Out_x.cpu().detach().numpy()
            Out_r           = Out_r.cpu().detach().numpy()

            Out_f           = fs    #   [T,B,<FP>]
            Out_fd          = fds    #  [T,B,FP]

    D = {}   # output dictionary
    D = { 'Out_x':Out_x, 'Out_z':Out_z,   'Out_f':Out_f, 'Out_fd':Out_fd }  
    return D


#### concatenate activity ####

def concat_activity( p, Out_x, Out_z ):
    T             = Out_x.shape[0] 
    numCond       = Out_x.shape[1]
    numsamp       = T * numCond
    data          = np.zeros([numsamp  , p.N])   #   [t,N]   hidden state 
    datar         = np.zeros([numsamp  , p.Z])   #   [t,R]   readout state 
    datatag       = np.zeros([numsamp       ])   #   [t,B]   
    samp1, samp2  = [0,0]
    for s in range(numCond):         # iterate stim
        for t in range(T):              # iterate times
            data[samp1,:]       = Out_x[t,s,:]
            datar[samp1,:]      = Out_z[t,s,:]            
            datatag[samp1]      = s          # g s
            samp1 += 1
    return data, datar, datatag


def concat_activity_simple( p, data ):
    # data (tensor): [T,N,B]
    # datamat (matrix): [sample,N]
    T             = data.shape[0] 
    B             = data.shape[2]
    numsamp       = T * B
    datamat       = np.zeros([numsamp  , p.N])   #   [t,N]   hidden state 
    s             = 0
    for b in range(B):         
        for t in range(T):   
            datamat[s,:]   = data[t,:,b]
    return datamat



### Run model across multiple conditions, output data ####

def Simulate_model(p, F, model, TT_run=range(17), verbose = False):  # runs model on multiple Time/Input conditions, stores activity

    if p.T == 1:
        TT_list = range(1)
    
    if model is nm.Model11:  
        runfunction = run_linmodel
    else:
        runfunction = run_model

    Data         = {}

    data            = []   # [TT][samps,cells]  # Neural activations
    datar           = []   # [TT][samps,R]      # Readout
    datatag         = []   # [TT][samps]        # Condition tag
    Xs              = []   # [TT][T,U,B]        # Inputs

    for TT in TT_run:  # TT: Time/Input conditions

        if verbose:
            print('Running model in condition %d' % (TT))

        _, _, _, X, input, _     = parms_test(p, F, TT)

        # Run model #
        with torch.no_grad():
            D     = run_model( model, p, input, hidden_output = True )
            Out_h = D['Out_h']   # [T,B,N]
            Out_z = D['Out_z']   # [T,B,Z]
        
        # Store matrix-form data #
        dat, dat_r, dat_tag  = concat_activity( p, Out_h, Out_z )
        data.append(        dat        )
        datar.append(       dat_r       )
        datatag.append(     dat_tag     )
        Xs.append(          X           )

 
    # II. Project into PCs and also format into Tensor form
    DATA, RDATA = ([],[])
    for TT in TT_run:    
        M  = Xs[TT].shape[2]                # get # of conditions from each time/input condition
        T2 = int(data[TT].shape[0]/M)       # get time duration from each time/input condition
        DATA.append(  np.zeros([T2,p.N,M])  )
        RDATA.append( np.zeros([T2,p.Z,M])  )    # [T,R,C]
        data_raw        = data[TT]
        rdata_raw       = datar[TT]
        for mm in range(M):
            a = mm*T2
            b = (mm+1)*T2
            RDATA[TT][:,:,mm] = rdata_raw[a:b,:]
            DATA[TT][:,:,mm]  = data_raw[a:b,:]

    # matrix format #
    Data['data']         = data         # [TT][s,N]       N-D             # Original axes 
    Data['datar']        = datar
    Data['datatag']      = datatag

    # tensor format #
    Data['DATA']         = DATA         # [TT][T,N,B]     N-D, neural units 
    Data['RDATA']        = RDATA        

    # also save #
    Data['Xs']           = Xs           # inputs for each time-trial condition

    # (will do this subsequently)
    # Data['PDATA']        = PDATA        # [TT][T,N,B]     N-D, PCs
    # Data['pca']          = pca          # scikit pca transform

    return Data


def Project_Data_PCA(p, F, Data, PCA_option, TT_pca = 0, pca_external = None):

    T, t1, t2, _, _, _          = parms_test(p,None,TT_pca)

    if pca_external == None:
        if PCA_option == 0:    # All conditions, all times
            N         = p.N
            dat       = Data['data'][TT_pca]      # [sample,N]
        elif PCA_option == 1:  # Delay cross-condition mean (n_items) subtracted
            N         = p.N
            taskfunc  = Get_taskfunc(p)
            i1inds    = taskfunc.Rank1_trials(p)
            dat       =  Data['DATA'][TT_pca][t1:t2,:,i1inds]
            for t in range(t2-t1):
                dat[t,:,:] = dat[t,:,:] - np.mean( dat[t,:,:] , axis=1 )[:,np.newaxis]
            dat = concat_activity_simple(p,dat)
        elif PCA_option == 2:  # Last quarter of delay, Spatial PCA
            taskfunc  = Get_taskfunc(p)
            i1inds    = taskfunc.Rank1_trials(p)
            N         = i1inds.size  # maximum dimension
            # quarter   = int( (t2-t1) / 4)
            # dat_win   =  Data['DATA'][TT_pca][(t2-quarter):t2,:,i1inds]
            # dat_mean  =  np.mean(dat_win, axis=0)   # take mean over this Time window
            # dat       =  np.transpose(dat_mean)
            dat   =  Data['DATA'][TT_pca][t2-1,:,i1inds]
        elif PCA_option == 3:  # First item timestep
            taskfunc  = Get_taskfunc(p)
            i1inds    = taskfunc.Rank1_trials(p)
            N         = i1inds.size  # maximum dimension
            dat   =  Data['DATA'][TT_pca][t1,:,i1inds]         
        pca         = PCA()   
        pca.fit( dat )
    else:
        pca = pca_external        
        N   = pca.n_components

    # II. Project into PCs and also format into Tensor form
    PDATA = []
    for TT in range(len(Data['DATA'])):    
        T2, _, _, X, _, _     = parms_test(p, F, TT)
        M                    = X.shape[2]   # get # of conditions from each time/input condition
        data_raw             = Data['data'][TT]
        pdata_raw            = pca.transform(data_raw)  # [samp,PC]
        PDATA.append( np.zeros([T2,N,M])  )    # [T,PC,C]
        # print(TT)
        for mm in range(M):
            a = mm*T2
            b = (mm+1)*T2
            PDATA[TT][:,:,mm] = pdata_raw[a:b,:]

    # III. update projected PCA data #
    Data['PDATA']          = PDATA
    Data['pca']            = pca

    return Data


def Project_Data_Delay_PCA(p, F, Data, TT_pca = 0):

    # projects data into PCA of the last timestep of the delay

    T, t1, t2, _, _, _          = parms_test(p,None,TT_pca)

    taskfunc  = Get_taskfunc(p)
    i1inds    = taskfunc.Rank1_trials(p)
    B         = i1inds.size  # maximum dimension

    dat   =  Data['DATA'][TT_pca][t2-1,:,i1inds]
    pca         = PCA()   
    pca.fit( dat )

    PDATA                = np.zeros([T,B,B])     # [T,N,B]     # top 3 PCs
    for t in range(T):
        dat             = Data['DATA'][TT_pca][t,:,i1inds]  # dat: [B,N]         e.g. [7,100]
        pdat            = pca.transform( dat )              # pdat: [B,N]   e.g. [7,7]
        PDATA[t,:,:]    = pdat[:,:B].T                      # [N,B] 

    Data_D                   = {}
    Data_D['DATA']           = [PDATA]                      # [0][T,N,B]
    Data_D['pca']            = pca

    return Data_D

def Project_Data_Basis(p, F, Data, basis_string, data_string, basis):

    T, t1, t2, _, _, _          = parms_test(p,None,0)

    N   = p.N

    # II. Project into PCs and also format into Tensor form
    PDATA = []
    for TT in range(len(Data['DATA'])):    
        T2, _, _, X, _, _     = parms_test(p, F, TT)
        M                    = X.shape[2]   # get # of conditions from each time/input condition
        data_raw             = Data['data'][TT]
        pdata_raw            = basis.transform( data_raw, center=True )  # [samp,PC]
        PDATA.append( np.zeros([T2,N,M])  )    # [T,PC,C]
        # print(TT)
        for mm in range(M):
            a = mm*T2
            b = (mm+1)*T2
            PDATA[TT][:,:,mm] = pdata_raw[a:b,:]

    Data[data_string]             = PDATA
    Data[basis_string]            = basis

    return Data


def print_model_performance(p, model, F, TT = 0, silent = False):

  # (preliminary) get trial indices #
  _, train, test               = Get_trial_indices(p)  # train, test indices (2D matrix, [i1,i2])

  # run model #
  _, _, _, _, input, ztarg     = parms_test(p, F, TT)
  with torch.no_grad():
    D     = run_model( model, p, input, initrand = False, hidden_output = False )
    Out_z = D['Out_z']   # [T,B,Z]

  # if no target defined, just return nan
  if len(ztarg) == 0:
      train  = np.nan
      test   = np.nan
      return train, test

  # parse output #
  ztargmat              = ztarg[:, 0].reshape(p.n_items, p.n_items)      # Target (2D)   #
  Choice, _, _          = parse_output(p,Out_z)
  Choicemat             = Choice.reshape(p.n_items,p.n_items)

  # print performance #
  train_acc      = ztargmat[ train  ] == Choicemat[ train  ]   # training indices
  test_acc       = ztargmat[ test  ]  == Choicemat[ test   ]  # probe/test indices
  train      = 100*np.mean(train_acc)
  test       = 100*np.mean(test_acc)
  
#   print(trainperf)

  if not silent :
    print('Train performance')
    print(str(sum(train_acc)) + ' of ' + str(train_acc.size) + ' correct: ' + \
            '%d' % round(train) + '%' )
    print('Test performance')
    print(str(sum(test_acc)) + ' of ' + str(test_acc.size) + ' correct: ' + \
            str('%d' % round(test)  + '%' ) )

  return train, test


def print_TI_performance_with_noise(p, model, X_input, TT = 0, silent = False):

    # (preliminary) get trial indices #
    _,train,test              = Get_trial_indices(p)  # get important indices
    taskfunc                  = Get_taskfunc(p)
    inds                      = taskfunc.trial_parser_symbolic_dist(p)   # TI specific
    widest_inds               = np.concatenate( (inds[-1].flatten(),inds[-2].flatten()) ) 

    q                   = copy.deepcopy(p)   # copy so we can change noise level
    q.sig               = p.sig
    q.noise_test        = True
    M                   = p.n_items

    R                   = 100

    _, _, _, _, input, ztarg     = parms_test(p, X_input, TT)

    # Run model to get behavior #
    Correct      = np.full( (M,M,R), 0, dtype=np.float32)     # correct 
    with torch.no_grad():
        for rr in range(R):
            D                     = run_model( model, q, input, initrand = False, hidden_output = False )
            Choice, _, rt         = parse_output(p,D['Out_z'])  # choice 0 and 1
            Correct[:,:,rr]       = 1 * ( Choice.reshape(M,M) == ztarg[:,0].reshape(M,M) )       

    # (averages over all simulations)
    Cmat                = np.mean(      Correct, axis=2 )[:,:,np.newaxis].ravel()     # mean correct
    perf                = 100 * np.mean( Cmat[widest_inds] )

    print('(noise test) widest performance: %g' % perf)

    return perf

# def print_model_performance(ztargmat,Choicemat,train,test,silent=False):
#   train_acc      = ztargmat[ train  ] == Choicemat[ train  ]   # training indices
#   test_acc       = ztargmat[ test  ] == Choicemat[ test   ]  # probe/test indices
#   trainperf = 100*np.mean(train_acc)
#   testperf  = 100*np.mean(test_acc)
#   if not silent :
#     print('Train performance')
#     print(str(sum(train_acc)) + ' of ' + str(train_acc.size) + ' correct: ' + \
#             '%d' % round(trainperf) + '%' )
#     print('Test performance')
#     print(str(sum(test_acc)) + ' of ' + str(test_acc.size) + ' correct: ' + \
#             str('%d' % round(testperf)  + '%' ) )
#   else:
#       return trainperf, testperf      