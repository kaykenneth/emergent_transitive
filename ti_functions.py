from __future__ import division
import sys
import numpy as np
import torch
import scipy
import copy
from itertools import combinations
from sklearn.manifold import MDS

import nn_analyses          as nna
import nn_models            as nm
import task_train_test      as tk
import nn_plot_functions    as npf

def get_train_data(p, X_input, *opt_train):

    if p.Task_version == 0:            ### Transitive Inference ###

        M       = X_input.shape[0]                  # number of items
        i1      = np.random.randint(M - 1)          # random item to start
        i2      = i1 + 1                            # the next adjacent item
        y       = np.array([1])                     # y = 1  (i.e. 1st index < 2nd index, item 1 closer to "A")
        if np.random.rand() < 0.5:              # Randomize Temporal order
            i1, i2 = i2, i1                         # flip order
            y = np.array([0])                       # y = 0, i.e. 1st index > 2nd index  (item 2 closer to "A")
        pind    = [i1,i2]    

    elif p.Task_version == 1:       ## Transitive Inference, full-train #

        M       = X_input.shape[0]
        items   = np.random.choice( M, size=2, replace=False )
        i1      = items[0]
        i2      = items[1]
        pind    = [i1,i2]    
        if i1 < i2:
            y   = np.array([1])
        else:
            y   = np.array([0])

    else:                               ### Transverse Patterning ###
        
        M       = X_input.shape[0]
        i1  = np.random.randint(M)
        if i1 == M-1:
            i2 = 0
        else:
            i2 = i1 + 1
        y   = np.array([1])
        if np.random.rand() < 0.5:     # Randomize Temporal order
            i1, i2 = i2, i1                  # flip order
            y = np.array([0])          # 0: 2nd stim index is Lower than 1st's
        pind    = [i1,i2]    
            
    X, j1, j2   = tk.format_training_input(p,  X_input, i1, i2)  # get formatted input
    Y           = tk.format_training_output(p, j1, j2, y)        # get formatted output

    ##### (optional) Add input noise ######
    if p.input_noise != 0:
        noise = p.input_noise * np.random.normal( scale=1 , size=X.shape )
        X     = X + noise

    return X, Y, pind     # X: single input, y: correct choice, pind: [p1,p2]              


def get_test_data(p, X_input, t_ext, offsets=[0,0], toggle=[1,1]):

    M = p.n_items

    # construct output matrix (across trial types) #
    x = y   = np.arange(M)                  # x, y are index vectors for item 1 and 2, respectively
    xv, yv  = np.meshgrid(x, y, indexing='ij')             # 2D matrices       
    xv, yv  = xv.flatten(), yv.flatten()    # 1D vectors
    z       = (xv < yv) * 1.0  #  Item 1 index < Item 2 index, then z is True (then 1)  
    z       = z[:, np.newaxis]          
    z[xv == yv] = -1                        # -1 means no right answer   
    xy = np.hstack((np.transpose(xv[np.newaxis,:]),np.transpose(yv[np.newaxis,:])))
    numCond = xy.shape[0]  # number of test trial types ("conditions")

    # input #
    X, j1, j2 = tk.format_test_input(p, X_input, xv, yv, t_ext, offsets, toggle)

    if p.Task_version >= 2:   # adjust for transverse patterning
        zmat = np.reshape(z,(p.n_items,p.n_items))
        zmat[-1,0] = 1
        zmat[0,-1] = 0
        z = zmat.flatten()
        z = z[:,np.newaxis]

    return X, z, xy, numCond, j1, j2     # X: [T,U,B]


def trial_parser(p):   # Gets indices of different types of trials#

    if p.Task_version < 2:   # classic TI

        M       = p.n_items

        ### 2D Matrix format ###

        # train #
        train_diag    = np.eye(M, M, k=1) + np.eye(M, M, k=-1)  # Train (2D)        #
        train_inds    = np.array(train_diag, dtype=bool)                                 # Train (2D)        #
        
        # edges #   (does not include sames or training)
        edges_inds    = np.zeros((M,M),dtype=bool) # initialize      
        edges_inds[2:,0]    = True
        edges_inds[-1,:-2]  = True
        edges_inds[0,2:]    = True
        edges_inds[:-2,-1]  = True  

        # probe #   (internal trials)
        probe_inds    = np.zeros((M,M),dtype=bool) # initialize            # Probe (2D)        #
        it             = np.nditer(probe_inds, flags=['multi_index']) # iterate 
        for xx in it:
        #    print('%d %s' % (xx,it.multi_index))
        #    print(it.multi_index)
            if abs( it.multi_index[0] - it.multi_index[1] ) > 1:
                if not(any(np.isin( it.multi_index, [0, M-1] ))):
                    #print(it.multi_index)
                    probe_inds[it.multi_index] = 1

        # test #
        test_inds = np.logical_or( probe_inds , edges_inds )

        # triangles #
        upper_inds  = (np.triu(np.ones([M,M]),k=1) == 1)
        lower_inds  = (np.tril(np.ones([M,M]),k=-1) == 1)


        ### 1D vector format ########
        trainind  = np.reshape(train_diag,(M**2)) != 0 
        probeind  = np.reshape(probe_inds,(M**2)) != 0
        samesind  = np.reshape(np.eye(M, M, k=0) == 1,(M**2)) != 0
        edgesind  = np.reshape(edges_inds,(M**2)) != 0
        upperind  = np.reshape(upper_inds,(M**2)) != 0 
        lowerind  = np.reshape(lower_inds,(M**2)) != 0 
        # (helper) #
        dummy           = np.ones([M,M])
        dummy[:,0:3] = 0
        shareblock  = np.reshape(dummy,(M**2)) != 0

        ### List format #########
        tind = [np.where(trainind & lowerind)[0],np.where(trainind & upperind)[0]]  # train [lower,upper]
        eind = [np.where(edgesind & lowerind)[0],np.where(edgesind & upperind)[0]]  # edges [lower,upper]
        pind = [np.where(probeind & lowerind)[0],np.where(probeind & upperind)[0]]  # probe [lower,upper]
        sind = [np.where(samesind)[0],np.where(samesind)[0]]  # sames [lower,upper]
        iind = np.where(probeind & lowerind & shareblock)[0]  # edges [lower,upper]

    elif p.Task_version > 2:           # Transverse Patterning

        M                     = p.n_items
        diag_inds             = np.eye(M, M, k=1) + np.eye(M, M, k=-1)  # Train (2D)        #
        diag_inds             = np.array(diag_inds, dtype=bool)                                 # Train (2D)        #
        train_inds            = np.copy(diag_inds)
        train_inds[0,-1]      = True
        train_inds[-1,0]      = True  
        test_inds             = np.invert(train_inds)

        lowercorner_inds           = np.zeros((M,M),dtype=bool) # initialize      
        uppercorner_inds           = np.zeros((M,M),dtype=bool) # initialize      
        lowercorner_inds[-1,0]     = True
        uppercorner_inds[0,-1]     = True  

            # 1D vec #
        diagind               = np.reshape(diag_inds,(M**2))         != 0 
        lowercornerind        = np.reshape(lowercorner_inds,(M**2))      != 0 
        uppercornerind        = np.reshape(uppercorner_inds,(M**2))      != 0 

            # 2D mat #
        upper_inds            = (np.triu(np.ones([M,M]),k=1) == 1)
        lower_inds            = (np.tril(np.ones([M,M]),k=-1) == 1)
        upperind              = np.reshape(upper_inds,(M**2)) != 0 
        lowerind              = np.reshape(lower_inds,(M**2)) != 0 

            # List #
        lowers                = (diagind & lowerind) | uppercornerind
        uppers                = (diagind & upperind) | lowercornerind
        tind                  = [np.where(lowers)[0], np.where(uppers)[0]]  # train [lower,upper]
        eind = pind = sind = iind  = [np.array([]),np.array([])]

    INDS = (tind,eind,pind,sind,iind)   # 0: train, 1: edges, 2: probe, 3: sames, 4: initial

    return INDS, train_inds, test_inds      




def trial_parser_symbolic_dist(p, flatten_indices=False):   

    # Gets indices of trials of different symbolic distances #
    # inds: [dist][choice,pairnum]

    N            = p.n_items
    inds         = []
    for d in range(1,N):   # distances range from 1 to N-1
        indvecs = np.full( (2, N-d), np.nan, dtype=int)   # [ <choice> , <pairnumber across diagonal> ]
        for c in [0,1]:
            if c == 0:      # choice: 0
                sign = -1       # lower triangle  
            elif c == 1:    # choice: 1
                sign = +1       # upper triangle  
            diagonal        = sign*d
            mat             = np.eye(N, N, k=diagonal)
            indvecs[c,:]    = np.where(mat.ravel())[0]
        if flatten_indices:
            indvecs = indvecs.flatten()
        inds.append( indvecs )

    return inds



def trial_parser_terminal_item(p):   

    # Gets indices of trials for terminal item vs. not #
    # inds: [no/yes][choice,indices]   no: not-terminal item, yes: with terminal item

    N            = p.n_items
    
    inds         = []
    for tt in range(2):   # 0: no, 1: yes
        
        if tt == 0:  # non-terminal trials (also, not center diagonal trials)
            numinds = int(  ((N-2)**2-(N-2))/2 )
        else:        # terminal trials
            numinds = int( N*2-3 )

        indvecs = np.full( (2, numinds), np.nan, dtype=int)   # [ <choice> , <trial> ]

        for c in range(2):    
            if c == 0:      # choice: 0     (1st item is "larger", i.e. closer to "A", red)
                sign = -1       # lower triangle  
            elif c == 1:    # choice: 1     (2nd item is "larger", i.e. closer to "A", blue)
                sign = +1       # upper triangle
            indss = np.array([])
            for dd in range(1,N): 
                diagonal        = sign*dd
                mat             = np.eye(N, N, k=diagonal)
                if tt == 0:
                    if dd < N-1:
                        indsadd         = np.unique( np.where(mat.ravel())[0][1:-1] )
                    else:
                        continue
                elif tt == 1:
                    indsadd         = np.unique( np.where(mat.ravel())[0][np.r_[0,-1]] )
                indss            = np.concatenate((indss,indsadd))
            indvecs[c,:] = indss
        inds.append(indvecs)
    return inds


def trial_parser_lexical_marking_item(p):   

    # Gets indices of trials for lexical item vs. not #
    # inds: [first/last][choice,indices]   no: not-terminal item, yes: with terminal item

    N            = p.n_items
    XV, YV       = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')    
    rank1vec     = XV.flatten()      # rank of 1st stim
    rank2vec     = YV.flatten()      # rank of 2nd stim   

    # np.reshape(upper_inds,(M**2))

    inds         = []
    for FL in range(2):   # 0: has first item ("A"), 1: has last item ("Z")
        indvecs = np.full( (2, N-1), np.nan, dtype=int)   # [ <choice> , <trial> ]
        for cc in range(2): # choice 0: lower triangle, choice 1: upper triangle
            if FL == 0:  # first item
                if cc == 0:     # lower triangle (2nd item is closer to "A", blue)
                    ind = np.logical_and( rank1vec > 0, rank2vec == 0 )
                elif cc == 1:   # upper triangle (1st item is closer to "A", red)
                    ind = np.logical_and( rank1vec == 0, rank2vec > 0 )
            elif FL == 1: # last item
                if cc == 0:     # lower triangle
                    ind = np.logical_and( rank1vec == N-1, rank2vec < N-1 )
                elif cc == 1:   # upper triangle
                    ind = np.logical_and( rank1vec < N-1, rank2vec == N-1 )
            indvecs[cc,:] = np.where(ind)[0]
        inds.append(indvecs)
    return inds
    

def trial_parser_symmetry(p):   

    # Gets indices of trials for symmetry behavior#
    # inds: [first/second][indices]   no: not-terminal item, yes: with terminal item

    N            = p.n_items
    XV, YV       = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')    
    rank1vec     = XV.flatten()      # rank of 1st stim
    rank2vec     = YV.flatten()      # rank of 2nd stim   

    choice1st       = np.where( rank1vec < rank2vec )[0]   # (keep in mind "A" is larger)
    choice2nd       = np.where( rank1vec > rank2vec )[0]

    inds         = []
    inds.append(choice1st)
    inds.append(choice2nd)

    return inds


def trial_parser_end_order(p):   

    # Gets indices of trials for End order (i.e. if End-item is 1st vs. 2nd item)
    # inds: [end1st/end2nd][choice,indices]   no: not-terminal item, yes: with terminal item

    N            = p.n_items
    XV, YV       = np.meshgrid(np.arange(N),np.arange(N),indexing='ij')    
    rank1vec     = XV.flatten()      # rank of 1st stim
    rank2vec     = YV.flatten()      # rank of 2nd stim   

    inds         = []
    for FL in range(2):   # 0: end item comes 1st, 1: end item comes 2nd
        indvecs = np.full( (1, 2*(N-2) ), np.nan, dtype=int)   # [ <choice> , <trial> ]
        if FL == 0:  # 1st item is End item
            # upper triangle (red) + first row (A is 1st item) (also, do not include G)
            ind1 = np.logical_and( rank1vec == 0, np.logical_and( rank2vec > 0, rank2vec < N-1 ) )
            # lower triangle (2nd item is closer to "A", blue)
            ind2 = np.logical_and( rank1vec == N-1, np.logical_and( rank2vec > 0, rank2vec < N-1 ) )
        elif FL == 1: # 2nd item is End item
            ind1 = np.logical_and( np.logical_and( rank1vec < N-1, rank1vec > 0) , rank2vec == N-1 )
            ind2 = np.logical_and( np.logical_and( rank1vec < N-1, rank1vec > 0), rank2vec == 0 )
        indvecs = np.where( np.logical_or(ind1,ind2) )[0]
        inds.append(indvecs)

    return inds
        

def plot_model_summary( p, F, model, TID, model_ep, model_string='', TT = 0 ):  

  import matplotlib as mpl
  import matplotlib.pyplot as plt
  # mpl.use("Qt5Agg") # set the backend
  mpl.rcParams.update({'font.size': 14})
  mpl.rcParams['font.sans-serif'] = "Helvetica"
  mpl.rcParams['font.family'] = "sans-serif"
  import matplotlib.patches as patch
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.patches   as patch

  # preliminary #
  INDS,trains,probes        = trial_parser(p)  # get important indices (for transitive inference)
  tind,eind,pind,sind,iind  = INDS # 0: train, 1: edges, 2: probe, 3: sames, 4: initial
  n_items     = p.n_items
  X_input     = F['X_input']
  train_table = F['train_table']

  # run model #
  _, _, _, X, input, _     = tk.parms_test(p,F,TT)
  with torch.no_grad():
    D     = tk.run_model( model, p, input, hidden_output = False )
    Out_z = D['Out_z']   # [T,B,Z]

  # parse output #
  Choice, Choiceval, RT = tk.parse_output(p,Out_z)
  Choicemat             = Choice.reshape(p.n_items, p.n_items, order='C')
  Choicevalmat          = Choiceval.reshape(p.n_items, p.n_items, order='C')  
  if p.Z == 3:
      RTmat                 = RT.reshape(p.n_items,p.n_items, order='C')   
  else:
      RTmat                 = np.full((p.n_items,p.n_items),-1)
  X, _, _, _,_,_ = get_test_data(p,X_input,0)
  print('Stim (item, dim)',X_input.shape)    #  Stimulus array   <# stim> x <stim dims>
  print('Probe (time, pairs, dim)', X.shape)          #  Input array      time x stim pairs x stim dims
  
  # Plot #
  fig0 = plt.figure()
  plt.show(block = False)    # prevents busy cursor
  plt.get_current_fig_manager().window.setGeometry(80,100,1200,1750) 
  fig0.suptitle('Transitive Inference: %d items\n%d units, TID: %d, train epoch: %d\n%s' \
              % (n_items,p.N,TID,model_ep,model_string), fontsize=17, weight='bold')
  # mngr = plt.get_current_fig_manager()
  
    # Stimuli #
  ax0 = fig0.add_subplot(5,3,1)
  ax0.imshow(X_input,cmap='binary',vmin=0,vmax=1)      # Plot Stim array
  ax0.set_title('Stimulus panel')
  ax0.set_ylabel('Stimulus #')

    # Example stim sequence #
  ax1 = fig0.add_subplot(5,3,2)
  ax1.imshow(X[:, :, 3],cmap='binary',vmin=0,vmax=1)   # Plot 3rd test pair stim 
  ax1.set_title('Example input')
  ax1.set_ylabel('Time (step)')
  plt.draw()     
  plt.pause(0.0001)       

  from mpl_toolkits.axes_grid1 import make_axes_locatable

  # Plot Histogram of probe values #
  ax4 = fig0.add_subplot(5,3,3)
  if p.Z == 1:
      Hist = [-1,2]
  elif p.Z == 2 :
    #   Hist = [-20,20]
      Hist = [-5,5]
  elif p.Z == 3 :
      Hist = [-10,+10]

  bin_edges   = np.linspace(Hist[0],Hist[1],100)
  vals        = Choicevalmat[probes]
  if p.Z == 1:
      ax4.fill([Hist[0],0.5,.5,Hist[0]],[0,0,vals.size*.75,vals.size*.75],closed=True,color=tuple([.75,.75,1]))
      ax4.fill([.5,Hist[1],Hist[1],.5],[0,0,vals.size*.75,vals.size*.75],closed=True,color=tuple([1,.75,.75]))
      ax4.plot([0.5,0.5],[0,vals.size*.75],'--',color='black',zorder=-100)
      ax4.plot([1,1],[0,vals.size*.75],'-',color='red',linewidth=2)
  elif p.Z == 2:
      ax4.fill([Hist[0],0,0,Hist[0]],[0,0,vals.size*.75,vals.size*.75],closed=True,color=tuple([.75,.75,1]))
      ax4.fill([0,Hist[1],Hist[1],0],[0,0,vals.size*.75,vals.size*.75],closed=True,color=tuple([1,.75,.75]))      
  ax4.plot([0,0],[0,vals.size*.75],'-',color='black',linewidth=2)
  ax4.hist(vals,bins=bin_edges,color='black')  # arguments are passed to np.histogram  
  ax4.axis(xmin=Hist[0],xmax=Hist[1],ymin=0,ymax=vals.size*.75)
  ax4.set_title('Test trials (n = %d) ' % vals.size)
#   print('Test RNN values: ' + '%s' % vals)


  # Model choice ##
  ax5 = fig0.add_subplot(5,3,4)
  pl5 = ax5.imshow(Choicemat, vmin=0, vmax=1, cmap='bwr')        # matrix
  plot_squares((Choicemat == -1),5,ax5)  # RNN chooses neither: green squares
  plot_squares((Choicemat == -2),4,ax5)  # RNN chooses both: black squares
  plot_squares(train_table,1,ax5)   # squares
#   plot_squares(probes,2,ax5)        # circles
  plt.xticks(ticks=range(p.n_items),labels=[])
  plt.yticks(ticks=range(p.n_items),labels=[])
  plt.colorbar(pl5,ax=ax5)
  ax5.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
  ax5.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
  ax5.axis(xmin=-0.5,xmax=n_items-0.5,ymin=-0.5,ymax=n_items-0.5)
  ax5.invert_yaxis()
  ax5.set_title('RNN choice')

  # Model quantitative output (readout units, subtracted) ##   
  ax6 = fig0.add_subplot(5,3,5)
  maxreadout = np.max(np.abs(Choicevalmat))
  pl6 = ax6.imshow(Choicevalmat, vmin=-maxreadout, vmax=+maxreadout, cmap='bwr')        # matrix
  plot_squares(train_table,1,ax6)  # squares
#   plot_squares(probes,2,ax6)    # circles
  plt.xticks(ticks=range(p.n_items),labels=[])
  plt.yticks(ticks=range(p.n_items),labels=[])
  plt.colorbar(pl6,ax=ax6)
  ax6.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
  ax6.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
  ax6.axis(xmin=-0.5,xmax=n_items-0.5,ymin=-0.5,ymax=n_items-0.5)
  ax6.invert_yaxis()
  ax6.set_title('Linear readout (output diff)\nmax value: %0.2f' % (maxreadout))

  # RT  ## (at right, numerical) 
  ax7 = fig0.add_subplot(5,3,6)
#   maxRT = np.nanmax(RTmat)
  maxRT = 1
  pl7 = ax7.imshow(RTmat, vmin=0, vmax=maxRT, cmap='binary')        # matrix
  plot_squares((Choicemat == -1),5,ax7)  # RNN chooses neither: green squares
  plot_squares((Choicemat == -2),5,ax7)  # RNN chooses both: green squares
  plot_squares(train_table,1,ax7)        # squares
#   plot_squares(probes,2,ax7)             # circles
  plt.xticks(ticks=range(p.n_items),labels=[])
  plt.yticks(ticks=range(p.n_items),labels=[])
  plt.colorbar(pl7,ax=ax7)
  ax7.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
  ax7.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
  ax7.axis( xmin=-0.5, xmax=n_items-0.5, ymin=-0.5, ymax=n_items-0.5 )
  ax7.invert_yaxis()
  ax7.set_title('RT (max: %0.2f)' % (maxRT))

  # RNN output, mirror ##   
  ax8 = fig0.add_subplot(5,3,8)
  listvals = List_Mirror_Values(p, Choicevalmat, signflip=True)
  ax8.scatter( listvals[:,0], listvals[:,1], color='purple',  marker='.', s=100, alpha=0.5 )
  pearson = np.corrcoef(listvals.transpose())[0,1]  # calculate correlation
  maxval   = np.max(listvals.ravel()) +2 
  ax8.plot([0,maxval],[0,maxval],alpha=0.1,linewidth=2,color='k')  
  ax8.axis(xmin=0,xmax=maxval,ymin=0,ymax=maxval)
  ax8.set_title('RNN output, mirror \n p = %0.2g' % (pearson))
  ax8.set_aspect('equal', 'box')
  ax8.set_xlabel('Out A')
  ax8.set_ylabel('Out B')  

  # RT, mirror #   
  ax9 = fig0.add_subplot(5,3,9)
  listvals = List_Mirror_Values(p, RTmat, signflip=False)
  ax9.scatter( listvals[:,0], listvals[:,1], color='k',  marker='.', s=100, alpha=0.5 )
  pearson = np.corrcoef(listvals.transpose())[0,1]  # calculate correlation
  maxrt   = np.nanmax(listvals.ravel()) + 2
  ax9.plot([0,maxrt],[0,maxrt],alpha=0.1,linewidth=2,color='k')
  ax9.set_title('RT, mirror \n p = %0.2g' % (pearson))
  ax9.axis(xmin=0,xmax=maxrt,ymin=0,ymax=maxrt)
  ax9.set_aspect('equal', 'box')  
  ax9.set_xlabel('RT A')
  ax9.set_ylabel('RT B')  
  
  # RNN output, mirror ##   
  ax10 = fig0.add_subplot(5,3,11)
  listvals = List_Symbolic_Dist_Values(p, Choicevalmat)
  Scatter_Discrete_Jitter(ax10, p, listvals, horz_jitter=0.1, color='k')
  ax10.set_title('RNN output vs. Symbolic Distance')
  ax10.set_xlabel('Distance')
  ax10.set_ylabel('RNN output')


  # RNN output, mirror ##   
  ax11 = fig0.add_subplot(5,3,12)
  listvals = List_Symbolic_Dist_Values(p, RTmat)
  Scatter_Discrete_Jitter(ax11, p, listvals, horz_jitter=0.1, color='k')
  ax11.set_title('RT vs. Symbolic distance')
  ax11.set_xlabel('Distance')
  ax11.set_ylabel('RT')

#   # Variable delay Performance #
#   if p.T > 1:
#     ax12 = fig0.add_subplot(5,3,7)
#     if 'performance_vd' in F.keys():
#         Perfs = F['performance_vd'][model_ind,:,:,:]  
#     else:
#         Perfs = tk.Evaluate_Performance_Variable_Delay(p, model, trial_parser,  F['X_input'])
#         # t2 moved later in time #  (darker)
#     ax12.plot(range(1,p.d1+1),Perfs[0,:,1],color='k',linewidth=6,alpha=0.5)
#     ax12.plot(range(1,p.d1+1),Perfs[1,:,1],marker='o',color='r',alpha=0.5)
#         # t1 moved earlier in time # (lighter)
#     ax12.plot(range(1,p.d1+1),Perfs[0,:,0],color='k',linewidth=6,alpha=0.2)
#     ax12.plot(range(1,p.d1+1),Perfs[1,:,0],marker='o',color='r',alpha=0.2)
#     ax12.set_xlabel('Delay shortening')
#     ax12.set_ylabel('Performance')
#     ax12.set_xticks(ticks=range(0,p.d1+1,10))

  fig0.subplots_adjust(hspace=.5)

#   # Plot Correct answers ## (at left) 
#   ax5 = fig0.add_subplot(2,3,4)
#   pl5 = ax5.imshow(ztargmat, cmap='bwr')                          # matrix
#   ax5.axis(xmin=-0.5,xmax=n_items-0.5,ymin=-0.5,ymax=n_items-0.5)
#   plt.xticks(ticks=range(p.n_items),labels=[])
#   plt.yticks(ticks=range(p.n_items),labels=[])
#   plot_squares(probes,2,ax5)    # circles
#   plot_squares(train_table,1,ax5)
#   plot_squares(np.eye(n_items),3,ax5)
#   plt.colorbar(pl5, ax=ax5)
#   # plt.gca().invert_yaxis()
#   ax5.invert_yaxis()
#   ax5.set_title('Correct')







def List_Mirror_Values(p, Valuemat, signflip=False):
    # gets values from pairs of trials that are reverse-order (A then B) and (B then A)
    if signflip:
        FLIP = -1
    else:
        FLIP = 1

    listval = np.zeros((  int((p.n_items**2-p.n_items)/2) ,2)) # paired values in list

    ind = 0
    for i1 in range(p.n_items):
      for i2 in range(p.n_items):
          if i1 > i2:       
              c1 = Valuemat[i1,i2]
              c2 = Valuemat[i2,i1]
              listval[ind,0] = c1 * FLIP  # (optional) reverse sign for convention
              listval[ind,1] = c2
              ind += 1
    return listval

def List_Symbolic_Dist_Values(p,Valuemat,signflip=False):
    # converts matrix of values (trial based)
    #    into Mx2 array with 2nd column indicating symbolic distance
    listval = np.zeros((  (p.n_items**2-p.n_items) ,2)) # trials that are not on-diagonal
    ind = 0
    for i1 in range(p.n_items):
      for i2 in range(p.n_items):
          if i1 > i2:       
              distance = np.abs(i1-i2)
              c1 = Valuemat[i1,i2]
              c2 = Valuemat[i2,i1]
              listval[ind,  0] = c1
              listval[ind+1,0] = c2
              listval[ind,  1] = distance
              listval[ind+1,1] = distance
              ind += 2
    return listval

def Scatter_Discrete_Jitter(ax, p, listvals, horz_jitter=0.1, **kwargs):
    # listvals: [values, discretevalue]
    for d in range(1,p.N):  # discrete values
        vals   = listvals[listvals[:,1] == d, 0 ] # values with this symbolic distance
        xvals  = d + ( np.random.random(vals.size) - 0.5 ) * horz_jitter
        ax.scatter( xvals, vals, **kwargs)

def trial_parser_not_sames(p):
    # returns indices of trials that are not on main diagonal (same trials) #
    M         = p.n_items
    inds      = (np.eye(M, M, k=0) == 0 ).flatten()
    return inds


def Error_study( p, F, Model_select, R, noise_sigs, TT = 0, RNN_guess=False):

    # gets error / rt behavior from a single model #

    import copy

    model, _, _, _     = tk.select_model( p, F, Model_select )     # obtain model
    
    B              = []

    for sig in noise_sigs:

        q                   = copy.deepcopy(p)   # copy so we can change noise level
        q.sig               = sig
        q.noise_test        = True
        M                   = p.n_items
        INDS,trains,probes     = trial_parser(p)  # get important indices (for transitive inference)
        _, _, _, _, input, ztarg     = tk.parms_test(p, F, TT)
    
        # Run model to get behavior #
        Correct      = np.full( (M,M,R), 0, dtype=np.float32)           # correct 
        RT           = np.full( (M,M,R), np.nan, dtype=np.float32)      # reaction time
        Outcome      = np.full( (M,M,R), np.nan, dtype=np.float32)      # outcome (i.e. 0,1,-1,-2)
        
        for rr in range(R):
            with torch.no_grad():
                D                     = tk.run_model( model, q, input, hidden_output = False )
                Choice, _, rt         = tk.parse_output(p,D['Out_z'])  # choice 0 and 1
                Outcome[:,:,rr]       = Choice.reshape(M,M)
                if p.T > 1:   # rnn
                    RT[:,:,rr]            = rt.reshape(M,M)
                    if RNN_guess:
                        no_response         = np.isnan( rt )             # indices of NR trials
                        num_no_response     = np.sum(no_response)        # no. of NR trials
                        Choice[no_response] = np.float64( np.random.randint(2,size=(num_no_response)) )  # make guess

                if 1:      
                    Correct[:,:,rr]   = 1 * ( Choice.reshape(M,M) == ztarg[:,0].reshape(M,M) )                    
                elif 0:     # disregard no-response trials (Apr 2023)
                    ind_response      = Choice.reshape(M,M) >= 0
                    if np.sum(ind_response) > 0:
                        corrmat                 = np.full((M,M),np.nan)
                        corrmat[ind_response]   = Choice.reshape(M,M)[ind_response] == ztarg[:,0].reshape(M,M)[ind_response] 
                        Correct[:,:,rr]         = corrmat                 
                    

        # Averages over all simulations #
        if 1:           # standard
            Cmat          = np.mean(      Correct, axis=2 )[:,:,np.newaxis]     # mean correct
            RTmat         = np.nanmean(   RT, axis=2     )[:,:,np.newaxis]      # mean RT
        elif 0:         # disregard no-response trials  (Apr 2023)
            Cmat          = np.full( (M,M), np.nan)
            RTmat         = np.full( (M,M), np.nan)
            for i1 in range(M):
                for i2 in range(M):
                    ind_response  = Outcome[i1,i2,:] >= 0
                    if np.sum(ind_response) > 0:
                        Cmat[i1,i2]     = np.mean(   Correct[i1,i2,ind_response] )     # mean correct
                        RTmat[i1,i2]    = np.mean(   RT[i1,i2,ind_response]   )      # mean RT
            Cmat    = Cmat[:,:,np.newaxis]
            RTmat   = RTmat[:,:,np.newaxis]

        # Calculate % of all simulated trials with no response #
        inds_notsame        = trial_parser_not_sames(p)
        num_trials          = R * np.sum( inds_notsame.size )
        num_no_response     = 0
        for rr in range(R):
            num_no_response += np.sum( np.isnan( RT[:,:,rr].flatten()[inds_notsame] ) )
        prop_no_response      = num_no_response / num_trials
        print('noise: %0.2f, no_response: %0.1f' % (sig, prop_no_response*100 ) )

        # Parse each TI behavior #
        sym     = Symbolic_Distance_Behavior(p, Correct, RT, Outcome)
        sympure = Symbolic_Distance_Pure_Behavior(p, Correct, RT, Outcome, RNN_guess)
        ter     = Terminal_Item_Behavior(p, Correct, RT, Outcome, RNN_guess)
        lex     = Lexical_Marking_Behavior(p, Correct, RT, Outcome, RNN_guess)
        syt     = Symmetry_Behavior(p, Correct, RT, Outcome, RNN_guess)
        eor     = End_Order_Behavior(p, Correct, RT, Outcome, RNN_guess)

        b                   = {}
        b['noise_sig']      = sig
        b['no_response']    = prop_no_response   # proportion of trials with no response
        b['Cmat']           = Cmat 
        b['RTmat']          = RTmat 
        b['sym']            = sym 
        b['ter']            = ter 
        b['lex']            = lex    # [first,last-containing][choice,model]
        b['sympure']        = sympure
        b['syt']            = syt
        b['eor']            = eor
        B.append(b)

    F['Behavior']  = B


def Symbolic_Distance_Behavior(p, Correct, Reactiontime, Outcome):

    M       = p.n_items
    R       = Correct.shape[2]

    C       = [None] * int(M-1)                         # [dist][choice,pair,modelnum] Correct (choose correctly)
    RT      = [None] * int(M-1)                         # [dist][choice,pair,modelnum] Reaction time
    N       = [None] * int(M-1)                         # [dist][choice,pair,modelnum] No. of trials
    OC      = np.full( (M-1, 4, 1), 0, dtype=np.float32)   # [dist,outcome,modelnum]     Trial outcome: 0,1: Choice 0/1 trials, 2:neither chosen, 3: both chosen

    distinds     = trial_parser_symbolic_dist(p)  # get important indices (for transitive inference)

    for rr in range(R):

        corr        = Correct[:,:,rr].flatten()
        outcome     = Outcome[:,:,rr].flatten()
        if p.T > 1:
            rt          = Reactiontime[:,:,rr].flatten()
        
        for dd in range(M-1):     # corresponds to distance 1, 2, etc.
            for pp in range(M-dd-1):  # trial type

                # correct #
                if C[dd] is None:  # if empty, initialize
                    C[dd]   = np.full( (2, M-dd-1, 1), 0, dtype=np.float32)   # correct [choice,pair,model] 
                    RT[dd]  = np.full( (2, M-dd-1, 1), 0, dtype=np.float32)   # reaction time
                    N[dd]   = np.full( (2, M-dd-1, 1), 0, dtype=np.float32)   # number of trials with a response (an RT)

                inds_pair       = distinds[dd][:,pp]   # [dist][choice,pp]
                corr_pair       = corr[inds_pair]      # correctness for this pair  (0 or 1 or nan(?))
                C[dd][:,pp,0]  += corr_pair            # adds correctness 0, 1, or 2
                if p.T > 1:
                    rt_pair        = rt[inds_pair]
                    for c in [0,1]:
                        rtval = rt_pair[c].astype(np.float32)
                        if rtval >= 0:
                            RT[dd][c,pp,0]    += rtval 
                            N[dd][c,pp,0]     += 1
                else:
                    N[dd][:,pp,0]     += 1

                # choice 1 #
                OC[dd,0,0]   += np.sum( outcome[inds_pair[0]] == 0 )  # Choice 0 trials in which 1 was chosen
                OC[dd,1,0]   += np.sum( outcome[inds_pair[1]] == 0 )  # Choice 1 trials in which 1 was chosen
                OC[dd,2,0]   += np.sum( np.isin( outcome[inds_pair[0]] , np.array([-1,-2])) )  # Choice 0 trials in which Neither/Both chosen
                OC[dd,3,0]   += np.sum( np.isin( outcome[inds_pair[1]] , np.array([-1,-2])) )  # Choice 1 trials in which Neither/Both chosen

    # Calculate proportion across simulations #
    for dd in range(M-1):
        C[dd]       = np.divide( C[dd] ,    R )     
        # C[dd]       = np.divide( C[dd] ,    N[dd] )         # (Apr 2023) calculate correct only for response trials
        OC[dd,:]    = np.divide( OC[dd,:] , (M-dd-1)*R )
        if p.T > 1:
            RT[dd]  = np.divide( RT[dd] , N[dd] )

    # Report non-responding trials (RNN) #
    # if p.T > 1:
    #     print( '(Error) % of trials with no model response' )
    #     print( np.round(100*np.transpose(OC[:,2])) )
    #     print( np.round(100*np.transpose(OC[:,3])) )

    sym         = {}
    sym['C']    = C
    sym['RT']   = RT
    sym['N']    = N
    sym['OC']   = OC

    return sym


def Symbolic_Distance_Pure_Behavior(p, Correct, Reactiontime, Outcome, RNN_guess):

    M       = p.n_items
    R       = Correct.shape[2]

    C       = [None] * int(M-1)                         # [dist][choice,modelnum] Correct (choose correctly)
    RT      = [None] * int(M-1)                         # [dist][choice,modelnum] Reaction time
    N       = [None] * int(M-1)                         # [dist][choice,modelnum] No. of trials
    OC      = np.full( (M-1, 4, 1), 0, dtype=np.float32)   # [dist,outcome,modelnum]     Trial outcome: 0,1: Choice 0/1, 2: neither chosen, 3: both chosen

    distinds     = trial_parser_symbolic_dist(p)  # get important indices (for transitive inference)

    for rr in range(R):

        corr        = Correct[:,:,rr].flatten()
        outcome     = Outcome[:,:,rr].flatten()
        if p.T > 1:
            rt          = Reactiontime[:,:,rr].flatten()
        
        for dd in range(M-1):     # corresponds to distance 1, 2, etc.
            
            # correct #
            if C[dd] is None:  # if empty, initialize
                C[dd]   = np.full( (2, 1), 0, dtype=np.float32)   # [choice,model]
                RT[dd]  = np.full( (2, 1), 0, dtype=np.float32)
                N[dd]   = np.full( (2, 1), 0, dtype=np.float32)

            for pp in range(M-dd-1):
                inds_pair       = distinds[dd][:,pp]   # [dist][choice,pp]
                C[dd][:,0]     += corr[inds_pair]  # 0, 1, or 2
                if p.T > 1:
                    rt_pair        = rt[inds_pair]
                    for c in [0,1]:
                        rtval = rt_pair[c].astype(np.float32)
                        if rtval >= 0:
                            RT[dd][c,0]    += rtval 
                            N[dd][c,0]     += 1
                else:
                    N[dd][:,0]     += 1

                # choice 0 #
                OC[dd,0,0]   += np.sum( outcome[inds_pair[0]] == 0 )  # Choice 0 trials in which 0 was chosen
                OC[dd,1,0]   += np.sum( outcome[inds_pair[1]] == 0 )  # Choice 1 trials in which 0 was chosen
                OC[dd,2,0]   += np.sum( np.isin( outcome[inds_pair[0]] , np.array([-1,-2])) )  # Choice 0 trials in which Neither/Both chosen
                OC[dd,3,0]   += np.sum( np.isin( outcome[inds_pair[1]] , np.array([-1,-2])) )  # Choice 1 trials in which Neither/Both chosen

    # calculate proportion across simulations #
    for dd in range(M-1):
        if RNN_guess:
            C[dd]       = np.divide( C[dd] , R*(M-dd-1) )  # divide by full number of trials
        else:
            C[dd]       = np.divide( C[dd] , R*(M-dd-1) )  # divide by full number of trials
            # C[dd]       = np.divide( C[dd] , N[dd] )        # divide by only number of responded trials
        OC[dd,0,0]    = np.divide( OC[dd,0,0] , N[dd][0] )  # divide by choice 0 trials
        OC[dd,1,0]    = np.divide( OC[dd,1,0] , N[dd][1] )  # divide by choice 1 trials
        OC[dd,2,0]    = np.divide( OC[dd,2,0] , N[dd][0] )  # divide by choice 0 trials
        OC[dd,3,0]    = np.divide( OC[dd,3,0] , N[dd][1] )  # divide by choice 1 trials
        if p.T > 1:
            RT[dd]  = np.divide( RT[dd] , N[dd] )

    sympure         = {}
    sympure['C']    = C
    sympure['RT']   = RT
    sympure['N']    = N
    sympure['OC']   = OC

    return sympure

def Terminal_Item_Behavior(p, Correct, Reactiontime, Outcome, RNN_guess):

    M       = p.n_items
    R       = Correct.shape[2]  # no. of simulations

    C       = [None] * 2                         # [no/yes][choice,modelnum] Correct (choose correctly)
    RT      = [None] * 2                         # [no/yes][choice,modelnum] Reaction time
    N       = [None] * 2                         # [no/yes][choice,modelnum] No. of trials, with responses
    N_all   = [None] * 2                         # [no/yes][choice,modelnum] No. of trials, all
    OC      = np.full( (2, 4, 1), 0.0)   # [no/yes,outcome,modelnum]     Trial outcome: 0,1: Choice 0/1 trials, 2:neither chosen, 3: both chosen

    inds     = trial_parser_terminal_item(p)  # get important indices (for transitive inference)

    for rr in range(R):

        corr            = Correct[:,:,rr].flatten()
        outcome         = Outcome[:,:,rr].flatten()
        if p.T > 1:
            rt          = Reactiontime[:,:,rr].flatten()
        
        for dd in range(2):     # no/yes, if have terminal item

            if C[dd] is None:  # if empty, initialize
                C[dd]       = np.full( (2, 1), 0.0 )   # [choice,model]
                RT[dd]      = np.full( (2, 1), 0.0 )
                N[dd]       = np.full( (2, 1), 0.0 )
                N_all[dd]   = np.full( (2, 1), 0.0 )
            
            for cc in range(2): # choice 0, 1

                ind          = inds[dd][cc,:]
                C[dd][cc,0] += np.sum( corr[ind] )  # 0, 1, or 2

                if p.T > 1:
                    rtvals = rt[ind].astype(np.float32) 
                    for pp in range(rtvals.size):
                        rtval = rtvals[pp]
                        if rtval >= 0:  
                            RT[dd][cc,0]    += rtval 
                            N[dd][cc,0]     += 1
                else:
                    N[dd][cc,0]             += ind.size
                N_all[dd][cc,0]             += ind.size

            # choice 0 #
            OC[dd,0,0]   += np.sum( outcome[inds[dd][0,:]] == 0 )  # Choice 0 trials in which 0 was chosen
            OC[dd,1,0]   += np.sum( outcome[inds[dd][1,:]] == 0 )  # Choice 1 trials in which 0 was chosen
            OC[dd,2,0]   += np.sum( np.isin( outcome[inds[dd][0,:]] , np.array([-1,-2])) )  # Choice 0 trials in which Neither/Both chosen
            OC[dd,3,0]   += np.sum( np.isin( outcome[inds[dd][1,:]] , np.array([-1,-2])) )  # Choice 1 trials in which Neither/Both chosen

    # calculate proportion across simulations #
    for dd in range(2):
        if RNN_guess:
            C[dd]         = np.divide( C[dd] , N_all[dd] )
        else:
            C[dd]         = np.divide( C[dd] , N_all[dd] )
            # C[dd]         = np.divide( C[dd] , N[dd] )
        OC[dd,0,0]    = np.divide( OC[dd,0,0] , N[dd][0] )  # divide by choice 0 trials
        OC[dd,1,0]    = np.divide( OC[dd,1,0] , N[dd][1] )  # divide by choice 1 trials
        OC[dd,2,0]    = np.divide( OC[dd,2,0] , N[dd][0] )  # divide by choice 0 trials
        OC[dd,3,0]    = np.divide( OC[dd,3,0] , N[dd][1] )  # divide by choice 1 trials
        if p.T > 1:
            RT[dd]  = np.divide( RT[dd] , N[dd] )

    ter         = {}
    ter['C']    = C
    ter['RT']   = RT
    ter['N']    = N
    ter['OC']   = OC

    return ter


def Lexical_Marking_Behavior(p, Correct, Reactiontime, Outcome, RNN_guess):

    M       = p.n_items
    R       = Correct.shape[2]  # no. of simulations

    C       = [None] * 2                         # [no/yes][choice,modelnum] Correct (choose correctly)
    RT      = [None] * 2                         # [no/yes][choice,modelnum] Reaction time
    N       = [None] * 2                         # [no/yes][choice,modelnum] No. of trials, respond trials
    N_all   = [None] * 2                         # [no/yes][choice,modelnum] No. of trials, all
    OC      = np.full( (2, 4, 1), 0.0)   # [no/yes,outcome,modelnum]     Trial outcome: 0,1: Choice 0/1 trials, 2:neither chosen, 3: both chosen

    inds     = trial_parser_lexical_marking_item(p)  # get important indices (for transitive inference)

    for rr in range(R):

        corr            = Correct[:,:,rr].flatten()
        outcome         = Outcome[:,:,rr].flatten()
        if p.T > 1:
            rt          = Reactiontime[:,:,rr].flatten()
        
        for dd in range(2):     # first/last - containing trial

            if C[dd] is None:  # if empty, initialize
                C[dd]       = np.full( (2, 1), 0.0 )   # [choice,model]
                RT[dd]      = np.full( (2, 1), 0.0 )
                N[dd]       = np.full( (2, 1), 0.0 )
                N_all[dd]   = np.full( (2, 1), 0.0 )
            
            for cc in range(2): # choice 0, 1

                ind          = inds[dd][cc,:]
                C[dd][cc,0] += np.sum( corr[ind] )  # 0, 1, or 2

                if p.T > 1:
                    rtvals = rt[ind].astype(np.float32) 
                    for pp in range(rtvals.size):
                        rtval = rtvals[pp]
                        if rtval >= 0:  
                            RT[dd][cc,0]    += rtval 
                            N[dd][cc,0]     += 1
                else:
                    N[dd][cc,0]     += ind.size
                N_all[dd][cc,0]     += ind.size

            # choice 0 #
            OC[dd,0,0]   += np.sum( outcome[inds[dd][0,:]] == 0 )  # Choice 0 trials in which 0 was chosen
            OC[dd,1,0]   += np.sum( outcome[inds[dd][1,:]] == 0 )  # Choice 1 trials in which 0 was chosen
            OC[dd,2,0]   += np.sum( np.isin( outcome[inds[dd][0,:]] , np.array([-1,-2])) )  # Choice 0 trials in which Neither/Both chosen
            OC[dd,3,0]   += np.sum( np.isin( outcome[inds[dd][1,:]] , np.array([-1,-2])) )  # Choice 1 trials in which Neither/Both chosen

    # calculate proportion across simulations #
    for dd in range(2):
        if RNN_guess:
            C[dd]         = np.divide( C[dd] , N_all[dd] )
        else:
            C[dd]         = np.divide( C[dd] , N_all[dd] )
            # C[dd]         = np.divide( C[dd] , N[dd] )
        OC[dd,0,0]    = np.divide( OC[dd,0,0] , N[dd][0] )  # divide by choice 0 trials
        OC[dd,1,0]    = np.divide( OC[dd,1,0] , N[dd][1] )  # divide by choice 1 trials
        OC[dd,2,0]    = np.divide( OC[dd,2,0] , N[dd][0] )  # divide by choice 0 trials
        OC[dd,3,0]    = np.divide( OC[dd,3,0] , N[dd][1] )  # divide by choice 1 trials
        if p.T > 1:
            RT[dd]  = np.divide( RT[dd] , N[dd] )

    lex         = {}
    lex['C']    = C     # [first,last-containing][choice]
    lex['RT']   = RT    # [first,last-containing][choice]
    lex['N']    = N     # [first,last-containing][choice]
    lex['OC']   = OC    # [first,last-containing][choice]

    return lex


def Symmetry_Behavior(p, Correct, Reactiontime, Outcome, RNN_guess):

    M       = p.n_items
    R       = Correct.shape[2]  # no. of simulations

    C       = [None] * 2                         # [1st/2nd][modelnum] Correct (choose correctly)
    RT      = [None] * 2                         # [1st/2nd][modelnum] Reaction time
    N       = [None] * 2                         # [1st/2nd][modelnum] No. of trials, respond
    N_all   = [None] * 2                         # [1st/2nd][modelnum] No. of trials, all
    OC      = np.full( (4, 1), 0.0)              # [outcome,modelnum]     Trial outcome: 0,1: Choice 0/1 trials, 2:neither chosen, 3: both chosen

    inds     = trial_parser_symmetry(p)  # get important indices (for transitive inference)

    for rr in range(R):

        corr            = Correct[:,:,rr].flatten()
        outcome         = Outcome[:,:,rr].flatten()
        if p.T > 1:
            rt          = Reactiontime[:,:,rr].flatten()
        
        for cc in range(2): # choice 0, 1

            if C[cc] is None:  # if empty, initialize
                C[cc]       = np.full( (1), 0.0 )   # [choice,model]
                RT[cc]      = np.full( (1), 0.0 )
                N[cc]       = np.full( (1), 0.0 )
                N_all[cc]   = np.full( (1), 0.0 )

            ind             = inds[cc]
            C[cc][0]       += np.sum( corr[ind] )  # 0, 1, or 2

            if p.T > 1:
                rtvals = rt[ind].astype(np.float32) 
                for pp in range(rtvals.size):
                    rtval = rtvals[pp]
                    if rtval >= 0:  
                        RT[cc][0]    += rtval 
                        N[cc][0]     += 1
            else:
                N[cc][0]             += ind.size
            N_all[cc][0]                 += ind.size

        # choice 0 #   
        OC[0,0]   += np.sum( outcome[inds[0]] == 0 )  # Choice 0 trials in which 0 was chosen
        OC[1,0]   += np.sum( outcome[inds[1]] == 0 )  # Choice 1 trials in which 0 was chosen
        OC[2,0]   += np.sum( np.isin( outcome[inds[0]] , np.array([-1,-2])) )  # Choice 0 trials in which Neither/Both chosen
        OC[3,0]   += np.sum( np.isin( outcome[inds[1]] , np.array([-1,-2])) )  # Choice 1 trials in which Neither/Both chosen

    # calculate proportion across simulations #
    if RNN_guess:
        NN = N_all
    else:
        NN = N_all
        # NN = N
    C[0]         = np.divide( C[0]      , NN[0] )
    C[1]         = np.divide( C[1]      , NN[1] )
    OC[0,0]    = np.divide( OC[0,0] , NN[0] )  # divide by choice 0 trials
    OC[1,0]    = np.divide( OC[1,0] , NN[1] )  # divide by choice 1 trials
    OC[2,0]    = np.divide( OC[2,0] , NN[0] )  # divide by choice 0 trials
    OC[3,0]    = np.divide( OC[3,0] , NN[1] )  # divide by choice 1 trials
    if p.T > 1:
        RT[0]  = np.divide( RT[0] , NN[0] )
        RT[1]  = np.divide( RT[1] , NN[1] )
 
    syt         = {}
    syt['C']    = C
    syt['RT']   = RT
    syt['N']    = N
    syt['OC']   = OC

    return syt


def End_Order_Behavior(p, Correct, Reactiontime, Outcome, RNN_guess):

    M       = p.n_items
    R       = Correct.shape[2]  # no. of simulations

    C       = [None] * 2                         # [1st/2nd][choice,modelnum] Correct (choose correctly)
    RT      = [None] * 2                         # [1st/2nd][choice,modelnum] Reaction time
    N       = [None] * 2                         # [1st/2nd][choice,modelnum] No. of trials, respond trials
    N_all   = [None] * 2                         # [1st/2nd][choice,modelnum] No. of trials, all
    OC      = np.full( (2, 4, 1), 0.0)   # [1st/2nd,outcome,modelnum]     Trial outcome: 0,1: Choice 0/1 trials, 2:neither chosen, 3: both chosen

    inds     = trial_parser_end_order(p)  # get important indices (for transitive inference)

    for rr in range(R):

        corr            = Correct[:,:,rr].flatten()
        outcome         = Outcome[:,:,rr].flatten()
        if p.T > 1:
            rt          = Reactiontime[:,:,rr].flatten()
        
        for dd in range(2):     # end item is 1st vs. 2nd 

            if C[dd] is None:  # if empty, initialize
                C[dd]       = np.full( (1, 1), 0.0 )   # [1,model]
                RT[dd]      = np.full( (1, 1), 0.0 )
                N[dd]       = np.full( (1, 1), 0.0 )
                N_all[dd]   = np.full( (1, 1), 0.0 )
            
            ind          = inds[dd]
            C[dd][0,0] += np.sum( corr[ind] )  # 0, 1, or 2

            if p.T > 1:
                rtvals = rt[ind].astype(np.float32) 
                for pp in range(rtvals.size):
                    rtval = rtvals[pp]
                    if rtval >= 0:  
                        RT[dd][0,0]    += rtval 
                        N[dd][0,0]     += 1
            else:
                N[dd][0,0]     += ind.size
            N_all[dd][0,0]     += ind.size

            # choice 0 #
            OC[dd,0,0]   += np.sum( outcome[inds[dd]] == 0 )  # Trials in which 0 was chosen
            OC[dd,1,0]   += np.sum( outcome[inds[dd]] == 0 )  # (same as above) 
            OC[dd,2,0]   += np.sum( np.isin( outcome[inds[dd]] , np.array([-1,-2])) )  # Trials in which Neither/Both chosen
            OC[dd,3,0]   += np.sum( np.isin( outcome[inds[dd]] , np.array([-1,-2])) )  # (same as above)

    # calculate proportion across simulations #
    for dd in range(2):
        if RNN_guess:
            C[dd]         = np.divide( C[dd] , N_all[dd] )
        else:
            C[dd]         = np.divide( C[dd] , N_all[dd] )
            # C[dd]         = np.divide( C[dd] , N[dd] )
        OC[dd,0,0]    = np.divide( OC[dd,0,0] , N[dd] )  # divide by total # trials
        OC[dd,1,0]    = np.divide( OC[dd,1,0] , N[dd] )  # 
        OC[dd,2,0]    = np.divide( OC[dd,2,0] , N[dd])   # 
        OC[dd,3,0]    = np.divide( OC[dd,3,0] , N[dd] )  # 
        if p.T > 1:
            RT[dd]  = np.divide( RT[dd] , N[dd] )

    eor         = {}
    eor['C']    = C     # [first,last-containing][choice]
    eor['RT']   = RT    # [first,last-containing][choice]
    eor['N']    = N     # [first,last-containing][choice]
    eor['OC']   = OC    # [first,last-containing][choice]

    return eor


def Behavior_plot( p, Behavior, TIDs_plot=[], windowshift=0 , titlestr = '', omit_individuals = False, SEM = False):  

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.size': 25})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    from scipy.stats import wilcoxon
    from scipy.stats import ranksums

    SEM_MULTIPLE = 1

    M = p.n_items
    _, trains, _     = trial_parser(p)  # get important indices (for transitive inference)

    if len(TIDs_plot) == 0:
        TIDs_plot   = np.array( Behavior['TIDs'] )
        # inds        = np.array([0])  # plot the first one
        inds        = np.full( (np.asarray( Behavior['TIDs'] ).size), True)
    else:
        inds        = np.where( np.isin(np.asarray(Behavior['TIDs']),TIDs_plot) )[0]   
        TIDs_plot   = np.asarray( Behavior['TIDs'] )[inds]
    # print(TIDs_plot)
    # print(len(inds))

    numinstances =  inds.size

    if len(Behavior['RTmat']) != 0  and  np.nanmax( Behavior['RTmat'] ) > 100:
        humanflag = 1
    else:
        humanflag = 0

    NUMCOLS = 9

    if p.T > 1:
        PLOTS = 2 # performance + RT
        windowheight = 750
    else:
        PLOTS = 1  # performance
        windowheight = 300

    # dot_alpha = 0.15  
    dot_alpha = 0.1  
    sem_alpha = 1   #0.5
    sem_lw    = 1.5


    # Plot #
    fig0 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(40,100+windowshift,2500,windowheight) 
    if titlestr:
        fig0.suptitle('Transitive Inference, Human subjects\n%s' % (titlestr), fontsize=16, weight='bold')
    else:
        # fig0.suptitle('Transitive Inference, TIDs: %s' % (np.array2string(TIDs_plot)), fontsize=8, weight='bold')
        fig0.suptitle('Transitive Inference, TIDs: %s' % (np.array2string(TIDs_plot[:10])), fontsize=8, weight='bold')

    for pl in range(PLOTS):

        if pl == 0:         # Performance #
            datastr = 'C'
            YLABEL  = 'Performance'
            Y_MIN = 0.40
            Y_MAX = 1.00
            YTICKS = np.arange(.4,1.1,0.1)
            YTICKLABELS = ['','0.5','','','','','1']

        elif pl == 1:       # RT #
            datastr = 'RT'
            YLABEL  = 'RT'
            Y_MIN = 0
            if humanflag:
                Y_MAX           = 700     # in milliseconds
                Y_MAX_RT_MAT    = 700
                Y_MAX_RT_MAT    = np.ceil( np.nanmax(np.nanmean(Behavior['RTmat'][:,:,inds],axis=2).flatten()) )
                print('Human Max RT: %g milliseconds' % (Y_MAX_RT_MAT))
                YTICKS = np.arange(0,Y_MAX+100,100)
                # YTICKLABELS = ['0','','','','','','','','0.8']
                YTICKLABELS = ['0','','','','','','','0.7']
            else:
                Y_MAX           = .7     # proportion of choice period duration
                # Y_MAX_RT_MAT    = .7
                Y_MAX_RT_MAT    =  np.nanmax(np.nanmean(Behavior['RTmat'][:,:,inds],axis=2).flatten())
                print('RNN Max RT: %g milliseconds' % (Y_MAX_RT_MAT))
                YTICKLABELS = ['0','','','','','','','0.7']
                YTICKS = np.arange(0,.8,0.1)

        # (1st) Matrix format #########
        if pl == 0:
            mat = Behavior['Cmat'][:,:,inds]
            vmin, vmax = (0,1)
        else:
            mat = Behavior['RTmat'][:,:,inds] 
            vmin, vmax = (0,Y_MAX_RT_MAT)
        ax1   = fig0.add_subplot(PLOTS,NUMCOLS,1+pl*NUMCOLS)
        mat_all = np.nanmean(mat,axis=2)
        pl5 = ax1.imshow(mat_all, vmin=vmin, vmax=vmax, cmap='binary')        # matrix
        plot_squares(trains,1,ax1)   # squares
        # plot_squares(probes,2,ax1)        # circles
        plt.xticks(ticks=range(p.n_items),labels=[])
        plt.yticks(ticks=range(p.n_items),labels=[])
        ax1.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
        ax1.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
        # plt.colorbar(pl5,ax=ax1)
        ax1.axis(xmin=-0.5,xmax=M-0.5,ymin=-0.5,ymax=M-0.5)
        ax1.invert_yaxis()
        ax1.set_xticks([])
        ax1.set_yticks([])
        if pl == 1:
            ax1.set_title('max RT: %0.6f' % (Y_MAX_RT_MAT),fontsize=10)
        # ax1.set_title(YLABEL)
        # plot_squares((Choicemat == -1), 5, ax1)  # RNN chooses neither: green squares
        # plot_squares((Choicemat == -2), 4, ax1)  # RNN chooses both: black squares
        plt.rcParams['font.size'] = '20'

        # # (3rd) Symbolic Dist: Choice 0 proportion #  (does not account for RNN_guess)
        # if pl == 0:
        #     OC      = Behavior['Sym']['OC']    
        #     ax     = fig0.add_subplot(PLOTS,NUMCOLS,2+pl*NUMCOLS)
        #     ax.set_title('Symbolic Distance')
        #     for m in inds:
        #         clr       = [ [np.random.rand(),0,0] ]
        #         clr.append( [0,0,np.random.rand()] )
        #         for ch in [0,1]:
        #             ax.scatter( np.arange(M-1)+1, OC[:,ch,m], s=150, color=clr[ch] )
        #             ax.plot(    np.arange(M-1)+1, OC[:,ch,m], linewidth=2, color=clr[ch] )
        #     xlims = ax.get_xlim()
        #     ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k')
        #     ax.set_xlim(xlims)
        #     ax.set_xticks(np.arange(M-1)+1)
        #     ax.set_ylim([0,1])
        #     ax.set_ylabel('Proportion Choice 0')
        #     ax.set_xlabel('Distance')
        #     ax.tick_params('both', length=8, width=1, which='major')


        # (8th) Delay robustness   #
        if 1:
            if pl == 0 and p.T > 1 and 'Train_vd' in Behavior.keys():
                ax     = fig0.add_subplot(PLOTS,NUMCOLS,9+pl*NUMCOLS)
                xvec    = np.arange(p.d1)
                clr_train   = (.3,.1,.4)
                clr_test    = (.4,.4,.4)
                train_mean  =  np.mean(     Behavior['Train_vd'][:,inds]/100    , axis=1   )
                test_mean   =  np.mean(     Behavior['Test_vd'][:,inds]/100     , axis=1   )
                if 0:
                    for m in inds:
                        ax.plot(xvec/p.d1, Behavior['Train_vd'][:,m]/100 ,'-',zorder=0, alpha=0.3, color=[.5,.4,.5], linewidth=2) 
                        ax.plot(xvec/p.d1, Behavior['Test_vd'][:,m]/100 ,'-',zorder=0, alpha=0.3, color=[.8,.8,.8],  linewidth=2)   
                elif 1:
                    train_sem   =  np.std(      Behavior['Train_vd'][:,inds]/100    , axis=1    ) #/ np.sqrt(inds.size)
                    test_sem    =  np.std(      Behavior['Test_vd'][:,inds]/100     , axis=1    ) #/ np.sqrt(inds.size)
                    ax.fill_between(xvec/p.d1, train_mean-train_sem,  train_mean+train_sem,  color=clr_train,  edgecolor=None,      alpha=.15)
                    ax.fill_between(xvec/p.d1, test_mean-test_sem,    test_mean+test_sem,    color=clr_test,       edgecolor=None,      alpha=.15)

                ax.plot(    xvec/p.d1, train_mean ,'-',     zorder=100, alpha=0.35, color=clr_train, linewidth=4 ) 
                ax.plot(    xvec/p.d1, test_mean  ,'-',     zorder=100, alpha=1,   color=clr_test, linewidth=4) 
                ax.set_xlim([0,1])
                if pl == 0:
                    xlims = ax.get_xlim()
                ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k')
                ax.set_xticks(np.arange(0,1.25,0.25))
                ax.axes.xaxis.set_ticklabels(['0','','0.5','','1'])
                ax.set_yticks(np.arange(0,1.1,.1))
                ax.set_ylim([0,1])
                ax.axes.yaxis.set_ticklabels(['0','','','','','0.5','','','','','1'])
                # ax.set_xlabel('Delay shortening')
                ax.tick_params('both', length=5, width=1, which='major')

        # continue


        # (1.5) (all models, SEM) Distance-Pair Groups #

        DODGE = 0.15
        PAD   = 0.07

        X = Behavior['Sym'][datastr]  #xxx

        ax     = fig0.add_subplot(PLOTS,NUMCOLS,2+pl*NUMCOLS)
        for g in range(len(X)):     # groups
            P       = X[g].shape[1]
            dist    = g+1  # symbolic distance
            xdel    = np.arange(P)*DODGE
            xvec    = dist + xdel - np.mean(xdel)
            for choice in [0,1]:
                if choice == 1:  # upper triangle
                    clr1 = 'r'
                elif choice == 0:  # lower triangle           
                    clr1 = 'b'
                meanvals = np.mean( X[g][choice,:,inds], axis=0)
                semvals  = SEM_MULTIPLE * np.std( X[g][choice,:,inds], axis=0) / np.sqrt(numinstances)
                ax.plot([xvec,xvec], [meanvals,meanvals + 1*semvals], '-',zorder=0,color=clr1,alpha=0.5,linewidth=2)  # performance
                ax.plot([xvec,xvec], [meanvals,meanvals - 1*semvals], '-',zorder=0,color=clr1,alpha=0.5,linewidth=2)  # performance
                ax.scatter(xvec, meanvals, s=30, color=clr1, alpha=0.5, zorder=10)  # performance
                ax.plot(xvec,meanvals,'-',zorder=0,color=clr1, alpha=0.5, linewidth=2)   # connecting line
            # critical pairs patch #
            if dist >= 2 and dist <= M-3:
                path = [ [xvec[1]-PAD,Y_MIN], [xvec[1]-PAD,Y_MAX], [xvec[-2]+PAD,Y_MAX], [xvec[-2]+PAD,Y_MIN]   ]
                ax.add_patch(Polygon(path, facecolor=[.94,.94,0.6], fill=True, alpha=0.5,zorder=-1))
            # ticks for pairs
            for xx in range(xvec.size):
                ax.plot([xvec[xx],xvec[xx]],[Y_MIN,Y_MIN+0.01],'-',linewidth=1,color='k')
        xlims = ax.get_xlim()
        if pl == 0:
            # ax.set_title('All trials')
            ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k',zorder=100)
        ax.set_xlim(xlims)
        ax.set_xticks(np.arange(M-1)+1)
        ax.tick_params('both', length=8, width=1, which='major')
        ax.set_yticklabels(YTICKLABELS)
        ax.set_yticks(YTICKS)
        ax.set_ylim([Y_MIN,Y_MAX])
        # ax.set_ylabel(YLABEL)
        # ax.set_xlabel('Distance')
        plt.rcParams['font.size'] = '20'


        # continue


        # (2nd) Distance-Pair Groups , single networks #

        if 0:
            DODGE = 0.15
            PAD   = 0.07

            X = Behavior['Sym'][datastr]  #xxx

            ax     = fig0.add_subplot(PLOTS,NUMCOLS,3+pl*NUMCOLS)
            for m in inds:   #inds:   # inds[0] range(inds):    
                # clr = np.random.rand((4))
                for g in range(len(X)):     # groups
                    P       = X[g].shape[1]
                    dist    = g+1  # symbolic distance
                    xdel    = np.arange(P)*DODGE
                    xvec    = dist + xdel - np.mean(xdel)
                    for choice in [0,1]:
                        if choice == 1:  # upper triangle
                            clr1 = 'r'
                        elif choice == 0:  # lower triangle           
                            clr1 = 'b'
                        # clr1 = varycolor(clr)
                        ax.scatter(xvec, X[g][choice,:,m], s=30, color=clr1, alpha=0.5, zorder=10)  # performance
                        ax.plot(xvec,X[g][choice,:,m],'-',zorder=0,color=clr1, alpha=0.5, linewidth=2)   # connecting line
                    # critical pairs patch #
                    if dist >= 2 and dist <= M-3:
                        path = [ [xvec[1]-PAD,Y_MIN], [xvec[1]-PAD,Y_MAX], [xvec[-2]+PAD,Y_MAX], [xvec[-2]+PAD,Y_MIN]   ]
                        ax.add_patch(Polygon(path, facecolor=[.94,.94,0.6], fill=True, alpha=0.5,zorder=-1))
                    # ticks for pairs
                    for xx in range(xvec.size):
                        ax.plot([xvec[xx],xvec[xx]],[Y_MIN,Y_MIN+0.01],'-',linewidth=1,color='k')
            xlims = ax.get_xlim()
            if pl == 0:
                # ax.set_title('All trials')
                ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k',zorder=100)
            ax.set_xlim(xlims)
            ax.set_xticks(np.arange(M-1)+1)
            ax.tick_params('both', length=8, width=1, which='major')
            ax.set_yticks(YTICKS)
            ax.set_yticklabels([])
            ax.set_ylim([Y_MIN,Y_MAX])
            ax.axes.yaxis.set_ticklabels([])
            # ax.set_ylabel(YLABEL)
            # ax.set_xlabel('Distance')
            plt.rcParams['font.size'] = '20'



        
        # continue   # just individual plots


        # (4th) Symbolic Distance  #
        X      = Behavior['Sympure'][datastr]                   # [no,yes][choice,modelnum]
        ax     = fig0.add_subplot(PLOTS,NUMCOLS,4+pl*NUMCOLS)
        xvec    = np.arange(M-1)+1  
        if not SEM:             # plot individual instances
            for m in inds:
                for choice in [0,1]:
                    if choice == 1:       
                        clr1 = 'r'
                        RB_OFFSET = -0.2
                    elif choice == 0:             
                        clr1 = 'b'
                        RB_OFFSET = +0.2
                    vals = []
                    for gg in range(M-1):
                        vals.append( X[gg][choice,m] )
                    # ax.plot(xvec + RB_OFFSET, vals,'-',zorder=0, alpha=dot_alpha, color='k',linewidth=2)   # connecting line
                    ax.scatter(xvec + RB_OFFSET, vals, s=80, color=clr1, alpha=dot_alpha, zorder=10)  # performancec
                for ddd in range(M-1):
                    ax.plot([ddd+1-0.2,ddd+1+0.2], np.flip( X[ddd][:,m] ),'-',zorder=0, alpha=dot_alpha, color='k',linewidth=2)   # connecting line
        else:        
            means   = np.full( (2,M-1) , np.nan)
            sems    = np.full( (2,M-1) , np.nan )
            for choice in [0,1]:
                if choice == 1:       
                    clr1 = 'r'
                    RB_OFFSET = -0.2
                elif choice == 0:             
                    clr1 = 'b'
                    RB_OFFSET = +0.2
                for gg in range(M-1):
                    means[choice,gg] = np.mean( X[gg][choice,inds] ) 
                    sems[choice,gg]  = SEM_MULTIPLE * np.std(  X[gg][choice,inds] )  / np.sqrt(numinstances)  
                ax.scatter(xvec + RB_OFFSET, means[choice,:], s=80, linewidth=sem_lw, edgecolor='k', color=clr1, alpha=sem_alpha, zorder=10)  # performancec
                ax.plot(   [xvec+RB_OFFSET,xvec+RB_OFFSET], [means[choice,:]-sems[choice,:],means[choice,:]+sems[choice,:]], linewidth=sem_lw, color='k', alpha=sem_alpha, zorder=-90)  # performancec
            for ddd in range(M-1):
                ax.plot([ddd+1-0.2,ddd+1+0.2], np.flip( means[:,ddd] ),'-',zorder=0, alpha=1, color='k',linewidth=sem_lw)   # connecting line
        
        ax.set_xlim([xlims[0],xlims[1]+0.4])
        xlims = ax.get_xlim()
        if pl == 0:
            # ax.set_title('Symbolic Distance')
            ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k')
        ax.set_xticks(np.arange(M-1)+1)
        ax.set_ylim([Y_MIN,Y_MAX])
        ax.set_yticks(YTICKS)
        ax.set_yticklabels([])
        # ax.set_xlabel('Distance')
        ax.tick_params('both', length=8, width=1, which='major')



        # (5th) Terminal Item  #
        C      = Behavior['Ter'][datastr]                   # [no,yes][choice,modelnum]
        ax     = fig0.add_subplot(PLOTS,NUMCOLS,5+pl*NUMCOLS)
        if not SEM:
            for m in inds:
                for gg in [0,1]:
                    G    = gg+1   # term or not
                    for choice in [0,1]:
                        if choice == 1:         
                            clr1 = 'r'
                            ord  = 0
                        elif choice == 0:                  
                            clr1 = 'b'
                            ord  = 1
                        xvec    = G + ord*0.2 - 0.1
                        # clr1 = varycolor(clr)
                        ax.scatter(xvec, C[gg][choice,m], s=80, color=clr1, alpha=dot_alpha, zorder=10)  # performance
                    ax.plot([G - 0.1,G+0.2 - 0.1], [ C[gg][1,m], C[gg][0,m]  ],'-',zorder=0, alpha=dot_alpha, color='k',linewidth=2)
        else:
            means  = np.full((2,2),np.nan)
            sems   = np.full((2,2),np.nan)  
            for gg in [0,1]:
                G    = gg+1   # term or not
                for choice in [0,1]:
                    if choice == 1:         
                        clr1 = 'r'
                        ord  = 0
                    elif choice == 0:                  
                        clr1 = 'b'
                        ord  = 1
                    xvec    = G + ord*0.2 - 0.1
                    means[choice,gg]    = np.mean( C[gg][choice,inds] )
                    sems[choice,gg]     = SEM_MULTIPLE * np.std( C[gg][choice,inds]  ) / np.sqrt(numinstances)
                    ax.scatter(xvec, means[choice,gg], s=80, edgecolor='k', linewidth=sem_lw, color=clr1, alpha=1, zorder=10)  # performance
                    ax.plot([xvec,xvec], [means[choice,gg]-sems[choice,gg],means[choice,gg]+sems[choice,gg]] , \
                            linewidth=sem_lw, color='k', alpha=1, zorder=-30)  # performance
                ax.plot([G - 0.1,G+0.2 - 0.1], [ means[1,gg], means[0,gg]  ],'-',zorder=0, alpha=1, color='k',linewidth=sem_lw)
        ax.set_xlim([0,3])
        ax.set_ylim([Y_MIN,Y_MAX])
        ax.set_xticks([1,2])
        ax.set_xticklabels(['No','Yes'])
        if pl == 0:
            xlims = ax.get_xlim()
            ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k')
        ax.set_yticks(YTICKS)
        ax.set_yticklabels([])
        # ax.set_xlabel('End Item')
        ax.tick_params('both', length=8, width=1, which='major')


        # (7th) End-order   #
        if p.T > 1:
            C      = Behavior['Eor'][datastr]                   # [no,yes][choice,modelnum]
            ax     = fig0.add_subplot(PLOTS,NUMCOLS,6+pl*NUMCOLS)
            if not SEM:
                for m in inds:
                    for gg in [0,1]:
                        if gg == 0:                     # End item 1st
                            clr1 = 'purple'
                        elif gg == 1:                   # End item 2nd
                            clr1 = 'orange'
                        # clr1 = varycolor(clr)
                        ax.scatter(1+gg, C[gg][0,m], s=80, alpha=dot_alpha, color=clr1,zorder=10)  # performance
                    ax.plot([1,2], [ C[0][0,m], C[1][0,m]  ],'-',zorder=0, alpha=dot_alpha, color='k',linewidth=2)   # connecting line
            else:
                means = np.full((2),np.nan)
                sems  = np.full((2),np.nan)
                for gg in [0,1]:
                    if gg == 0:                     # End item 1st
                        clr1 = 'purple'
                    elif gg == 1:                   # End item 2nd
                        clr1 = 'orange'
                    means[gg] = np.mean( C[gg][0,inds] )
                    sems[gg]  = SEM_MULTIPLE * np.std( C[gg][0,inds] )  / np.sqrt(numinstances)
                    ax.scatter( 1+gg, means[gg], s=80, linewidth=sem_lw, edgecolor='k', alpha=sem_alpha, color=clr1,zorder=10)  # performance
                    ax.plot(   [1+gg,1+gg], [means[gg]-sems[gg],means[gg]+sems[gg]], linewidth=sem_lw, alpha=sem_alpha, color='k', zorder=-20)  # performance
                ax.plot([1,2], [ means[0], means[1]  ],'-',zorder=0, alpha=0.25, color='k',linewidth=sem_lw)   # connecting line
                if pl == 1: 
                    pairedvals = C[0][0,inds] - C[1][0,inds]
                    _, p_value = scipy.stats.wilcoxon(pairedvals)
                    print('(p = %g, Wilcoxon) RT, end-order effect (signed rank), across subjects' % (p_value) )

            ax.set_xlim([0,3])
            ax.set_ylim([Y_MIN,Y_MAX])
            ax.set_xticks([1,2])
            if pl == 0:
                xlims = ax.get_xlim()
                ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k')
            ax.set_xticklabels(['1st','2nd'])
            ax.set_yticks(YTICKS)
            ax.set_yticklabels([])
            # ax.set_xlabel('End-item order')
            ax.tick_params('both', length=8, width=1, which='major')

        if omit_individuals:
            continue


        # (7th) Lexical Marking   #
        C      = Behavior['Lex'][datastr]                   # [no,yes][choice,modelnum]
        ax     = fig0.add_subplot(PLOTS,NUMCOLS,7+pl*NUMCOLS)
        for m in inds:
            clr       = [ [np.random.rand(),0,0] ]
            clr.append( [0,0,np.random.rand()] )
            for gg in [0,1]:
                G    = gg+1   # term or not
                for choice in [0,1]:
                    if choice == 1:         
                        clr1 = 'r'
                        ord  = 0
                    elif choice == 0:                
                        clr1 = 'b'
                        ord  = 1
                    xvec    = G + ord*0.2 - 0.1
                    # clr1 = varycolor(clr)
                    ax.scatter(xvec, C[gg][choice,m], s=80, alpha=dot_alpha, color=clr1,zorder=10)  # performance
                ax.plot([G - 0.1,G+0.2 - 0.1], [ C[gg][1,m], C[gg][0,m]  ],'-',zorder=0, alpha=dot_alpha, color='k',linewidth=2)   # connecting line
        # xlims = ax.get_xlim()
        if pl == 0:
            xlims = ax.get_xlim()
            ax.plot([xlims[0],xlims[1]],[0.5,0.5],'--',color='k')
        ax.set_xlim([0,3])
        ax.set_ylim([Y_MIN,Y_MAX])
        ax.set_xticks([1,2])
        ax.set_xticklabels(['A','G'])
        ax.set_ylim([Y_MIN,Y_MAX])
        ax.axes.yaxis.set_ticklabels([])
        if humanflag == 0 or pl == 0:
            ax.set_yticks(YTICKS)
        # ax.spines.left.set_visible(False)
        # ax.spines.right.set_visible(False)
        # ax.spines.top.set_visible(False)
        # ax.set_xlabel('End item')
        ax.tick_params('both', length=8, width=1, which='major')





        # (8th) Symmetry   #
        C      = Behavior['Syt'][datastr]                   # [no,yes][choice,modelnum]
        ax     = fig0.add_subplot(PLOTS,NUMCOLS,8+pl*NUMCOLS)
        for m in inds:
            clr       = [ [np.random.rand(),0,0] ]
            clr.append( [0,0,np.random.rand()] )
            for gg in [0,1]:
                G    = gg+1   # term or not
                if gg == 0:             # red       (choose 1st item as closer to "A")
                    clr1 = 'r'
                else:                   # blue  (choose 1st item as closer to "A")
                    clr1 = 'b'
                xvec    = G
                # clr1 = varycolor(clr)
                ax.scatter(xvec, C[gg][m], s=80, alpha=dot_alpha, color=clr1,zorder=10)  # performance
            ax.plot([1,2], [ C[0][m], C[1][m]  ],'-',zorder=0, alpha=dot_alpha, color='k',linewidth=2)   # connecting line
        if pl == 0:
            # ax.set_title('Symmetry')
            ax.plot([xlims[0]-1,xlims[1]+1],[0.5,0.5],'--',color='k')
        ax.set_xlim([0,3])
        ax.set_ylim([Y_MIN,Y_MAX])
        ax.set_xticks([1,2])
        ax.set_xticklabels(['1st','2nd'])
        if humanflag == 0 or pl == 0:
            ax.set_yticks(YTICKS)
        ax.set_ylim([Y_MIN,Y_MAX])
        ax.axes.yaxis.set_ticklabels([])
        # ax.set_xlabel('Choice')
        ax.tick_params('both', length=8, width=1, which='major')


    fig0.subplots_adjust(hspace=1)
    plt.gcf().subplots_adjust(bottom=0.25)

    plt.draw()     
    plt.show()
    plt.pause(0.0001)

    plt.ion()
    plt.ioff()
    plt.ion()




def varycolor(baseclr):
    clr = np.array(baseclr)   + .3*(np.random.rand(3)-0.5)
    clr[clr>1] = 1
    clr[clr<0] = 0
    return clr


# def Behavior_simulations(jobname, TIDs_process, R, sig, localdir = '',  TT_test = [0], Model_select = -1, RNN_guess=True ):

#    if TIDs_process.size == 0:
#     print('no TIDs!')
#     return

#    sys.path.insert(0, localdir[:-1] + jobname)
#    from init import singleout, params, genfilename
#    p = singleout()                    # instantiate simply to access .dims conveniently
#    pdict = dict(vars(params()))

#    print('(%s) Plotting selected TIDs Error Pattern' % (localdir + p.jobname))
   
#    # ii. Get stored values from all models ####

#    TIDs           = []
#    Cmat           = []   # [M,M]  Correct matrix    (mean over simulations)
#    RTmat          = []   # [M,M]  RT matrix         (mean over simulations)
#    Sym            = {}
#    Ter            = {}
#    Lex            = {}
#    Sympure        = {}
#    Syt            = {}

#    for tt in range(len(TIDs_process)):

#       TID               = int(TIDs_process[tt])
#       TIDs.append(TID)
#       p_TID             = singleout(TID = TID)  # get parms object
#       fname             = genfilename(p_TID,localdir)
#       file              = tk.readfile(fname,".p",silent=True)
#       if len(file) == 0:
#          continue
#       F                 = file[0]  # retrieve model

#       # Run error study #
#       Error_study( p, F, Model_select, R, sig, TT = 0, RNN_guess=RNN_guess)

#       # Get delay robustness #
#       if 'performance_vd' in F.keys(): 
#         _, _, i, _     = tk.select_model( p, F, Model_select )
#         train_vd = F['performance_vd'][i,0,:,1,np.newaxis]  # [t,model]  t: time (in timesteps) shortened from full delay
#         test_vd  = F['performance_vd'][i,1,:,1,np.newaxis]  # [t,model]

#       # Concatenate across models #
#       if len(Cmat) == 0:
#         Cmat    = cmat       # [i1,i2]         >> [i1,i2,model]
#         RTmat   = rtmat
#         Sym     = sym
#         Ter     = ter
#         Lex     = lex
#         Sympure = sympure
#         Syt     = syt
#         Train_vd     = train_vd
#         Test_vd      = test_vd
#       else:
#         Concatenate_Models_Behavior(  Sym,          sym )
#         Concatenate_Models_Behavior(  Ter,          ter )
#         Concatenate_Models_Behavior(  Lex,          lex )
#         Concatenate_Models_Behavior(  Sympure,      sympure )
#         Concatenate_Models_Behavior(  Syt,      syt )
#         Cmat    = np.concatenate( (Cmat, cmat), axis = 2)
#         RTmat   = np.concatenate( (RTmat, rtmat), axis = 2)
#         Train_vd    = np.concatenate( (Train_vd, train_vd), axis = 1)
#         Test_vd    = np.concatenate( (Test_vd, test_vd), axis = 1)

#    Behaviors            = {}
#    Behaviors['TIDs']    = TIDs
#    Behaviors['Sym']     = Sym
#    Behaviors['Ter']     = Ter
#    Behaviors['Lex']     = Lex
#    Behaviors['Sympure'] = Sympure
#    Behaviors['Syt']     = Syt

#    Behaviors['Cmat']        = Cmat
#    Behaviors['RTmat']       = RTmat
#    Behaviors['Train_vd']    = Train_vd
#    Behaviors['Test_vd']     = Test_vd

#    return Behaviors

def Behavior_across_models( jobname, TIDs_process, Model_select, localdir = '', MAX_INSTANCES = np.inf ):

    if TIDs_process.size == 0:
        print('no TIDs!')
        return

    sys.path.insert(0, localdir[:-1] + jobname)
    from init import singleout, params, genfilename
    p = singleout()                    # instantiate simply to access .dims conveniently
    pdict = dict(vars(params()))

    print('(%s) Plotting selected TIDs Error Pattern' % (localdir + p.jobname))
    
    # ii. Get stored values from all models ####

    TIDs           = []
    model_variant  = []
    noise_sigs     = []
    Cmat           = []   # [M,M]  Correct matrix    (mean over simulations)
    RTmat          = []   # [M,M]  RT matrix         (mean over simulations)
    Sym            = {}
    Ter            = {}
    Lex            = {}
    Sympure        = {}
    Syt            = {}
    Eor            = {}
    Eor_nf         = {}

    _,trains,tests    = trial_parser(p)  # get important indices (for transitive inference)
    train_inds        = trains.flatten()
    test_inds         = tests.flatten()
    inds              = trial_parser_symbolic_dist(p)
    all_inds          = np.concatenate( (np.where(test_inds)[0],np.where(train_inds)[0]) )
    sinds             = trial_parser_symmetry(p)
    uppertrain        = np.intersect1d( sinds[0] , np.where(train_inds)[0])        
    lowertrain        = np.intersect1d( sinds[1] , np.where(train_inds)[0])           
    endinds           = trial_parser_terminal_item(p)   # [0:non-end,1:has end]
    wide_inds         = np.concatenate( (inds[-1].flatten(),inds[-2].flatten()) ) 
    widest_inds       = inds[-1].flatten()
    endtrain          = np.intersect1d( np.where(train_inds)[0] , endinds[1])  # training pairs w/ end items        

    # Variant
    variant_count   = np.array([0,0,0,0,0])
    variant_keepcount   = np.array([0,0,0,0,0])
    no_response_prop    = []

    for tt in range(len(TIDs_process)):

      TID               = int(TIDs_process[tt])
      p_TID             = singleout(TID = TID)  # get parms object
      fname             = genfilename(p_TID,localdir)
      file              = tk.readfile(fname,".p",silent=True)
      if len(file) == 0:
         continue
      F                 = file[0]  # retrieve model

      # (reportage) Count number of RNN variants inherited from real_model_performance #
      v = p_TID.Model_variant
      variant_count[v] += 1

      # Get delay robustness #
      if p.Model < 0 and 'performance_vd' in F.keys(): 
        _, _, i, _     = tk.select_model( p, F, Model_select )
        train_vd = F['performance_vd'][i,0,:,1,np.newaxis]  # [t,model]  t: time (in timesteps) shortened from full delay
        test_vd  = F['performance_vd'][i,1,:,1,np.newaxis]  # [t,model]
      else:
        train_vd = test_vd = None

      # Identify (titrate) naturalistic noise level based on average performance  #
      foundflag = False
      for nn in range(len(F['Behavior'])):

        noise_sig = F['Behavior'][nn]['noise_sig']
        # if p.T > 1 and noise_sig > 3:  # RNNs don't need higher noise
        #     continue

        all_perf                = F['Behavior'][nn]['Cmat'].flatten()[all_inds]
        train_perf_1_mean       = np.mean( F['Behavior'][nn]['Cmat'].flatten()[uppertrain] ) # mean of choice 1st training trials
        train_perf_2_mean       = np.mean( F['Behavior'][nn]['Cmat'].flatten()[lowertrain] ) # mean of choice 2nd training trials
        # train_ends_perf         = F['Behavior'][nn]['Cmat'].flatten()[endtrain]
        # (unused) #
        # perf                = np.mean( F['Behavior'][nn]['Cmat'][trains] )       # all training trials
        # wide_perf           = np.mean( F['Behavior'][nn]['Cmat'].flatten()[wide_inds] )
        widest_perf             = np.mean( F['Behavior'][nn]['Cmat'].flatten()[widest_inds] )
        # train_perf_1        =  F['Behavior'][nn]['Cmat'].flatten()[uppertrain]     # choice 1st training trials
        # train_perf_2        =  F['Behavior'][nn]['Cmat'].flatten()[lowertrain]     # choice 2nd training trials
        train_perf_mean            = np.mean(  F['Behavior'][nn]['Cmat'].flatten()[np.where(train_inds)[0]] )
        test_perf_mean            = np.mean(  F['Behavior'][nn]['Cmat'].flatten()[np.where(test_inds)[0]] )

        ###### ** Take first model that meets this Performance Criterion ##############
        # if train_perf_1_mean > 0.5 and train_perf_2_mean > 0.5 and np.all(all_perf < 0.95):  #.95:  # criterion
        # if train_perf_1_mean > 0.5 and train_perf_2_mean > 0.5 and train_perf_mean <= 0.95:  #.95:
        # if train_perf_1_mean > 0.5 and train_perf_2_mean > 0.5 and test_perf_mean <= 0.9:  #.95:
        if train_perf_1_mean > 0.5 and train_perf_2_mean > 0.5 and widest_perf <= .96:  #.95:   # Final criteria
        # if train_perf_1_mean > 0.5 and train_perf_2_mean > 0.5 and widest_perf < 1:  #.95:
            foundflag = True
            break
        #########################################################################

      if foundflag is False:
          print('(noise titration) no valid performance for this model!')
          continue    
    
      # (variant count) check whether we need this instance
      if variant_keepcount[v] < MAX_INSTANCES:
            variant_keepcount[v] += 1
            model_variant.append(v)
      else:
            continue

      # calculate % of no response trials #
    #   F['Behavior'][nn]['sym']['OC']    #  [dist,outcome,modelnum]     Trial outcome: 0,1: Choice 0/1, 2: neither chosen, 3: both chosen      TIDs.append(TID)
        # F['Behavior'][nn]['syt']['OC']    #  [dist,outcome,modelnum]     Trial outcome: 0,1: Choice 0/1, 2: neither chosen, 3: both chosen      TIDs.append(TID)
      no_response = np.mean( F['Behavior'][nn]['syt']['OC'][2:] )
      no_response_prop.append(no_response)
    #   print( np.round(100*np.transpose(OC[:,2])) )
    #   print( np.round(100*np.transpose(OC[:,3])) )

      # store TID and noise level #
      TIDs.append(TID)
      noise_sig = F['Behavior'][nn]['noise_sig']
      noise_sigs.append(  noise_sig  )

      # report noise analysis #
      print('TID: %d | noise: %g, test: %g, train: %g, no_response: %g' % \
                        (TID, noise_sig, 100*test_perf_mean, 100*train_perf_mean, 100*no_response))

      # Concatenate across models #
      if len(Cmat) == 0:
        Cmat    = F['Behavior'][nn]['Cmat']       # [i1,i2]         >> [i1,i2,model]
        RTmat   = F['Behavior'][nn]['RTmat']
        Sym     = F['Behavior'][nn]['sym']
        Ter     = F['Behavior'][nn]['ter']
        Lex     = F['Behavior'][nn]['lex']
        Sympure = F['Behavior'][nn]['sympure']
        Syt     = F['Behavior'][nn]['syt']
        if p.T > 1:
            Eor     = F['Behavior'][nn]['eor']
            Eor_nf  = F['Behavior'][0]['eor']  # noise-free
            Train_vd     = train_vd
            Test_vd      = test_vd
      else:
        Concatenate_Models_Behavior(  Sym,          F['Behavior'][nn]['sym'] )
        Concatenate_Models_Behavior(  Ter,          F['Behavior'][nn]['ter'] )
        Concatenate_Models_Behavior(  Lex,          F['Behavior'][nn]['lex'] )
        Concatenate_Models_Behavior(  Sympure,      F['Behavior'][nn]['sympure'] )
        Concatenate_Models_Behavior(  Syt,      F['Behavior'][nn]['syt'] )
        Cmat    = np.concatenate( (Cmat,    F['Behavior'][nn]['Cmat'] ), axis = 2)
        RTmat   = np.concatenate( (RTmat,   F['Behavior'][nn]['RTmat'] ), axis = 2)
        if p.T > 1:
            Concatenate_Models_Behavior(  Eor,      F['Behavior'][nn]['eor'] )
            Concatenate_Models_Behavior(  Eor_nf,      F['Behavior'][0]['eor'] )
            Train_vd    = np.concatenate( (Train_vd, train_vd), axis = 1)
            Test_vd    = np.concatenate( (Test_vd, test_vd), axis = 1)

    Behaviors            = {}
    Behaviors['TIDs']    = TIDs
    Behaviors['noise_sigs']  = noise_sigs
    Behaviors['model_variant'] = model_variant
    Behaviors['Sym']     = Sym
    Behaviors['Ter']     = Ter
    Behaviors['Lex']     = Lex
    Behaviors['Sympure'] = Sympure
    Behaviors['Syt']     = Syt
    Behaviors['Eor']     = Eor
    Behaviors['Eor_nf']  = Eor_nf       # noise-free

    Behaviors['Cmat']        = Cmat
    Behaviors['RTmat']       = RTmat
    
    if p.T > 1:
        Behaviors['Train_vd']    = Train_vd
        Behaviors['Test_vd']     = Test_vd
        Behaviors['no_response'] = no_response_prop

    # report model counts #
    print('Model variants, instances inherited from read_model_perf: %s' % (variant_count))
    print('Model variants, instances kept: %s'    % (variant_keepcount))
            
    return Behaviors


def Debrief_across_humans ( datadir ):

    #### Human data: load & parse #####################
    import pandas as pd
    from init import singleout, params, genfilename
    p = singleout()                    # instantiate simply to access .dims conveniently
    debriefpath         = datadir + '/debrief.csv'
    db                  = pd.read_csv(debriefpath)
    data_db             = db.to_numpy()
    # basic  #
    TIDs_all            = np.unique( data_db[:,1] )        # All TIDs (PIDs) in this dataframe
    Num_Total_Subjects  = TIDs_all.size
    print('---- %d total human subjects ----' % (Num_Total_Subjects))
    print('(%s) Behavioral plots, Human study' % (datadir + p.jobname))
    ####################################################
    return data_db

def Behavior_across_humans( datadir ):

    #### Human data: load & parse #####################
    import pandas as pd
#    sys.path.insert(0, localdir[:-1] + jobname)
    from init import singleout, params, genfilename
    p = singleout()                    # instantiate simply to access .dims conveniently
    datapath            = datadir + '/data.csv'
    debriefpath         = datadir + '/debrief.csv'
    df                  = pd.read_csv(datapath)
    db                  = pd.read_csv(debriefpath)
    data                = df.to_numpy()
    data_db             = db.to_numpy()
    # basic  #
    TIDs_all            = np.unique( data[:,1] )        # All TIDs (PIDs) in this dataframe
    Num_Total_Subjects  = TIDs_all.size
    # ind_test_phase   = data[:,5] > 3          # Test Phase trials
    # ind_train       = data[:,28] == 1        # Training trial types
    # ind_test        = data[:,28] > 0         # Testing trial types
    # ind_all_dists   = data[:,28] >= 1        # 
    Train_Blocks        = np.unique( data[ data[:,6] == 'training', 5] )
    Test_Blocks         = np.unique( data[ data[:,6] == 'test', 5] )
    TR                  = Train_Blocks.size  # get number of testing blocks
    R                   = Test_Blocks.size  # get number of testing blocks
    # print #
    print('---- %d total human subjects ----' % (Num_Total_Subjects))
    print('(%s) Behavioral plots, Human study' % (datadir + p.jobname))
    ####################################################

    #### Basic task info #####################
    M                   = p.n_items
    INDS,trains,tests   = trial_parser(p)  # get important indices (for transitive inference)
    train_inds          = trains.flatten()
    test_inds           = tests.flatten()
    sinds               = trial_parser_symmetry(p)
    train_1st           = np.intersect1d( sinds[0] , np.where(train_inds)[0])        
    train_2nd           = np.intersect1d( sinds[1] , np.where(train_inds)[0])           
    
    probe_inds          = np.array([INDS[2][0],INDS[2][1]]).flatten()
    inds_not_end        = ~ np.any( np.isin( data[:,9:11], (1,7) ), axis=1 )  # not contain end items

    # inds              = trial_parser_symbolic_dist(p)
    # all_inds          = np.concatenate( (np.where(test_inds)[0],np.where(train_inds)[0]) )
    # endinds           = trial_parser_terminal_item(p)   # [0:non-end,1:has end]

    # ii. Get stored values from all models ####
    TIDs           = []
    VER            = []
    Cmat           = []   # [M,M]  Correct matrix    (mean over simulations)
    RTmat          = []   # [M,M]  RT matrix         (mean over simulations)
    Sym            = {}
    Ter            = {}
    Lex            = {}
    Sympure        = {}
    Syt            = {}
    Eor            = {}    
    inds           = np.full((Num_Total_Subjects),False)

    for tt in range(Num_Total_Subjects):

        TID               = int(TIDs_all[tt])
        ind_subject       = data[:,1] == TID
        versions          = np.unique( data[ind_subject,33] )
        if versions.size > 1:
            print('more than one study participation? --- skipping the subject!')
            continue
        else:
            ver = versions[0]

        # Parse Training Data #
        Correct_train           = np.full((TR,2),0)  # [block,1st/2nd]
        Totaltrials_train       = np.full((TR,2),0)
        num_no_response_train   = 0
        for rr in range(TR):             # iterate over number of blocks - here, same as # of Testing blocks)
            ind_train_block = Train_Blocks[rr] == data[:,5]
            for pp in range(int(M*M)):
                r1,r2              = ranks(p,pp)
                if np.abs(r1-r2)==1:  # adjacent pairs
                    inds1 = np.logical_and( ind_subject     , ind_train_block )
                    inds2 = np.logical_and( data[:,9] == r1+1 , data[:,10] == r2+1 )
                    ind   = np.logical_and( inds1 , inds2 )
                    if np.sum(ind) == 0:
                        print('(PID: %g) no values for this trial type!' % (TID))
                        continue
                    choice             = np.int32( np.unique( data[ind,15] ))   # 0: choose 2nd, 1: choose 1st 
                    # RT[r1,r2,rr]        = data[ind,13]      # rt
                    Correct_train[rr,choice]         += np.nansum( data[ind,16] )      # whether choice was correct
                    Totaltrials_train[rr,choice]     += np.sum( ind )      # choice
                    num_no_response_train     += np.sum( pd.isnull(data[ind,32]) )
                    # if np.sum( pd.isnull( data[ind,32]) ) > 0:
                    #     print('no training response!')


        # Parse Testing Data #  
        Correct         = np.full( (M,M,R), 0, dtype=np.float32)           # correct  time
        Outcome         = np.full( (M,M,R), np.nan, dtype=np.float32)      # outcome
        RT              = np.full( (M,M,R), np.nan, dtype=np.float32)      # reaction

        for rr in range(R):             # iterate over number of blocks - here, same as # of Testing blocks)
            ind_test_block = Test_Blocks[rr] == data[:,5]
            for pp in range(int(M*M)):
                r1,r2              = ranks(p,pp)
                if r1 != r2:  # only non-same trial types
                    inds1 = np.logical_and( ind_subject     , ind_test_block )
                    inds2 = np.logical_and( data[:,9] == r1+1 , data[:,10] == r2+1 )
                    ind   = np.logical_and( inds1 , inds2 )
                    if np.sum(ind) > 1:
                        print('(PID: %g) more than one val for this trial type! i.e. bug or more than one repeat in this block' % (TID))
                        print(data[ind,14])
                        print(data[ind,13])
                        ind = np.where(ind)[0][0]  # quick fix: take first one
                        # breakpoint()
                    elif np.sum(ind) == 0:
                        print('(PID: %g) no values for this trial type!' % (TID))
                        continue
                    Outcome[r1,r2,rr]   = data[ind,14]      # choice
                    RT[r1,r2,rr]        = data[ind,13]      # rt
                    Correct[r1,r2,rr]   = data[ind,16]      # whether choice was correct
          

        # Calculate % of trials with no response #
        inds_notsame           = trial_parser_not_sames(p)
        # num_no_response_train  # determined above
        num_no_response_test   = 0
        num_train_trials       = TR * 4 * 2 * 6
        num_test_trials        = R * np.sum( inds_notsame.size )
        for rr in range(R):
            num_no_response_test   += np.sum( np.isnan( RT[:,:,rr].flatten()[inds_notsame] ) )
        # prop_no_response_test      = num_no_response_test/num_no_response_test
        # print('% of trials with no response: %0.2f' % (prop_no_response*100 ) )

        ### Performance on final Training block ####
        Perf_Train_Late  = Correct_train[-1,:] / Totaltrials_train[-1,:]   # Performance on Block 3

        ### Performance, Early Testing #######
        EARLY_TEST                   = np.arange(3)
        cmat_early                   = np.mean(   Correct[:,:,EARLY_TEST], axis=2 )[:,:,np.newaxis]     # mean correct
        rtmat_early                  = np.nanmean(   RT[:,:,EARLY_TEST],      axis=2 )[:,:,np.newaxis]      # mean RT
        Perf_Train_Early_Test        = np.array([np.nan,np.nan])           # <choice2,choice1>
        Perf_Train_Early_Test[1]     = np.mean(   cmat_early.flatten()[train_1st]  )    # mean of mean Train performance (mean across trial types mean)
        Perf_Train_Early_Test[0]     = np.mean(   cmat_early.flatten()[train_2nd]  )    # mean of mean Train performance (mean across trial types mean)
        Perf_Test_Early_Test         = np.mean(   cmat_early.flatten()[test_inds]   )    # mean of mean Test performance (mean across trial types mean)

        ### Performance, All Testing ####### (unused for now)
        ALL_TEST                   = np.arange(6)
        cmat_all                   = np.mean(   Correct[:,:,ALL_TEST], axis=2 )[:,:,np.newaxis]     # mean correct
        rtmat_all                  = np.nanmean(   RT[:,:,ALL_TEST],      axis=2 )[:,:,np.newaxis]      # mean RT
        Perf_Train_All_Test        = np.array([np.nan,np.nan])           # <choice2,choice1>
        Perf_Train_All_Test[1]     = np.mean(   cmat_all.flatten()[train_1st]  )    # mean of mean Train performance (mean across trial types mean)
        Perf_Train_All_Test[0]     = np.mean(   cmat_all.flatten()[train_2nd]  )    # mean of mean Train performance (mean across trial types mean)
        Perf_Test_All_Test         = np.mean(   cmat_all.flatten()[test_inds]   )    # mean of mean Test performance (mean across trial types mean)

        ### Performance List ###  [1,2,3,4,5,6,7]
        perflist          = np.array([Perf_Train_Late[1], Perf_Train_Late[0], Perf_Train_Early_Test[1], Perf_Train_Early_Test[0], Perf_Test_Early_Test, num_no_response_train, num_no_response_test])

        ### Performance, average by block ########################
        Perf_Blocks   = np.full((3,TR+R),np.nan)        # [<test,train,critical>,block]
        for rr in range(TR):    # training trial types, training blocks
            Perf_Blocks[0,rr]   = np.nanmean( Correct_train[rr]/Totaltrials_train[rr] )
        for rr in range(R):     # each of 3 trial types, testing blocks 
            Perf_Blocks[0,TR+rr]   = np.nanmean( Correct[:,:,rr].flatten()[train_inds] )  # train 
            Perf_Blocks[1,TR+rr]   = np.nanmean( Correct[:,:,rr].flatten()[test_inds] )   # test
            Perf_Blocks[2,TR+rr]   = np.nanmean( Correct[:,:,rr].flatten()[probe_inds] )  # critical

        ### Performance: Critical trials ########################
        #  (indices)
        # inds_not_end      indices of trials that do not have end items #
        ind_1st_test_block      = Test_Blocks[0] == data[:,5]           # 1st test block
        ind_all_test_trials     = data[:,7]  == 'non_adjacent'                  # test trials 
        inds_test               = np.logical_and.reduce( (ind_subject, ind_1st_test_block, ind_all_test_trials) )    # for this subject    
        inds_critical           = np.logical_and.reduce( (inds_test, inds_not_end) )
        # ind_earliest_test     = np.min( np.where(inds_test)[0] )           # very first Test trial for this subject
        ind_earliest_probe      = np.min( np.where(inds_critical)[0] ) 
        
        # Get first presentation(s) of critical pairs #
        inds_first_probes   = np.array([],dtype=np.int32)
        criticals           = np.where(inds_critical)[0]
        for nn in reversed( range(criticals.size) ):
            i1              = data[criticals[nn],9]
            i2              = data[criticals[nn],10]
            addflag         = True
            for nnn in range(nn):
                i_1         = data[criticals[nnn],9]
                i_2         = data[criticals[nnn],10]
                if i_1 == i2 and i_2 == i1:
                    addflag = False
                    break
            if addflag:
                inds_first_probes = np.append(inds_first_probes,criticals[nn])

        # Calculate correct #
        numcorrect_critical     = np.sum(data[inds_first_probes,32] == 1)
        numcorrect_critical_all = np.sum(data[inds_critical,32] == 1) 
        numcritical             = inds_first_probes.size
        numcritical_all         = np.sum(inds_critical)
        Perf_avg_critical     =  numcorrect_critical / numcritical
        Perf_avg_critical_all =  numcorrect_critical_all / numcritical_all
        pvalue                = scipy.stats.binom_test(numcorrect_critical, numcritical, p=0.5, alternative='two-sided')
        pvalue_all            = scipy.stats.binom_test(numcorrect_critical_all, numcritical_all, p=0.5, alternative='two-sided')
        Perf_1st_critical   = data[ind_earliest_probe,32]            # correct/inc/noresp for 1st test
        # output #
        Perf_1st            = np.array( [Perf_1st_critical, Perf_avg_critical, Perf_avg_critical_all, pvalue, pvalue_all] )
 
        # Parse each TI behavior #
        Correct_quant   = Correct[:,:,EARLY_TEST]
        RT_quant        = RT[:,:,EARLY_TEST]
        sym     = Symbolic_Distance_Behavior(p, Correct_quant, RT_quant, Outcome)
        sympure = Symbolic_Distance_Pure_Behavior(p, Correct_quant, RT_quant, Outcome, False)
        ter     = Terminal_Item_Behavior(p, Correct_quant, RT_quant, Outcome, False)
        lex     = Lexical_Marking_Behavior(p, Correct_quant, RT_quant, Outcome, False)
        syt     = Symmetry_Behavior(p, Correct_quant, RT_quant, Outcome, False)
        eor     = End_Order_Behavior(p, Correct_quant, RT_quant, Outcome, False)

        # report #
        print('%g (PID: %g) Train_final: %s Train: %s Test: %s (no response: %g of %g)' % \
              (tt, TID, str(np.round(np.mean(Perf_Train_Late)*100)), str(np.round(np.mean(Perf_Train_Early_Test)*100)), str(round(Perf_Test_Early_Test*100)), num_no_response_train+num_no_response_test, num_train_trials + num_test_trials ) )
        
        # Concatenate across models #
        TIDs.append(TID)
        VER.append(ver)

        if len(Cmat) == 0:
            Cmat    = cmat_early       # [i1,i2]         >> [i1,i2,model]
            RTmat   = rtmat_early
            Sym     = sym
            Ter     = ter
            Lex     = lex
            Sympure = sympure
            Syt     = syt
            Eor     = eor
            Perfs           = perflist[:,np.newaxis]
            Perfs_Blocks    = Perf_Blocks[:,:,np.newaxis] 
            Perfs_1st       = Perf_1st[:,np.newaxis]
        else:
            Concatenate_Models_Behavior(  Sym,          sym )
            Concatenate_Models_Behavior(  Ter,          ter )
            Concatenate_Models_Behavior(  Lex,          lex )
            Concatenate_Models_Behavior(  Sympure,      sympure )
            Concatenate_Models_Behavior(  Syt,          syt )
            Concatenate_Models_Behavior(  Eor,      eor)
            Cmat         = np.concatenate( (Cmat,    cmat_early  ), axis = 2)
            RTmat        = np.concatenate( (RTmat,   rtmat_early ), axis = 2)
            Perfs        = np.concatenate( (Perfs,  perflist[:,np.newaxis]  ), axis = 1)
            Perfs_Blocks = np.concatenate( (Perfs_Blocks, Perf_Blocks[:,:,np.newaxis] ), axis=2)
            Perfs_1st    = np.concatenate( (Perfs_1st,  Perf_1st[:,np.newaxis]  ), axis = 1)

    Behaviors            = {}
    Behaviors['TIDs']    = TIDs
    Behaviors['VER']     = VER
    Behaviors['Perfs']          = Perfs   # [<perftrain,perftest,prop_no_resp>, subject]
    Behaviors['Perfs_Blocks']   = Perfs_Blocks   # [<train,test,probe>, block, subject]
    Behaviors['Perfs_1st']      = Perfs_1st   # [<perf_1st,critical_flag>, subject]
    Behaviors['Sym']     = Sym
    Behaviors['Ter']     = Ter
    Behaviors['Lex']     = Lex
    Behaviors['Sympure'] = Sympure
    Behaviors['Syt']     = Syt
    Behaviors['Eor']     = Eor
    Behaviors['Cmat']    = Cmat
    Behaviors['RTmat']   = RTmat
            
    return Behaviors, data 


def Concatenate_Models_Behavior(X,x):
    numgroups   = len(X['C'])
    modaxis     = np.ndim(X['C'][0])-1  # axis dimension to concatinate models
    for g in range(numgroups):
        X['C'][g] = np.concatenate( (   X['C'][g],      x['C'][g]),    axis = modaxis)
        X['N'][g] = np.concatenate( (   X['N'][g],      x['N'][g]),    axis = modaxis)
        X['RT'][g] = np.concatenate( (  X['RT'][g],     x['RT'][g]),    axis = modaxis)      
    X['OC'] = np.concatenate( (X['OC'],x['OC']), axis = modaxis )


def Select_Human_Subjects(p,Behaviors, TRAIN_FINAL, TRAIN, TEST, NO_RESPONSE, COHORTS):
    # perflist          = np.array([Perf_Train_Late[1], Perf_Train_Late[0], Perf_Train_Early_Test[1], Perf_Train_Early_Test[0], Perf_Test_Early_Test, num_no_response_train, num_no_response_test])
    Perfs_train_late_1    = Behaviors['Perfs'][0,:]   # training performance on final block
    Perfs_train_late_2    = Behaviors['Perfs'][1,:]   # training performance on final block
    Perfs_train_1         = Behaviors['Perfs'][2,:]   # training performance, Early Test (blocks 4-6)
    Perfs_train_2         = Behaviors['Perfs'][3,:]   # training performance, Early Test (blocks 4-6)
    Perfs_train_late      = np.mean( Behaviors['Perfs'][:2,:] , axis=0)
    Perfs_train           = np.mean( Behaviors['Perfs'][2:4,:] , axis=0)
    Perfs_test            = Behaviors['Perfs'][4,:]   # test performance, Early Test
    Perfs_noresp          = Behaviors['Perfs'][5,:] + Behaviors['Perfs'][6,:]
    inds_cohort          = np.isin( np.array(Behaviors['VER']) , COHORTS).flatten()
    inds_resp             = Perfs_noresp < NO_RESPONSE
    if 1:           # minimal
        inds1                 = np.logical_and.reduce( (inds_cohort, inds_resp, \
                                                        Perfs_train_late > TRAIN_FINAL, 
                                                        Perfs_test <= TEST  )     )
    elif 1:         # super stringent
        inds1                 = np.logical_and.reduce( (inds_cohort, inds_resp, \
                                                        Perfs_train_late_1 > TRAIN_FINAL, \
                                                        Perfs_train_late_2 > TRAIN_FINAL, \
                                                        Perfs_train_1 > TRAIN, \
                                                        Perfs_train_2 > TRAIN, \
                                                        Perfs_test <= TEST  )     )                                                    
    inds                  = np.logical_and(inds1,inds_cohort)
    TIDs                  = np.asarray(Behaviors['TIDs'])[inds]
    num_subj_cohort = np.sum(inds_cohort)
    selection_string  = 'Cohort: %s\nTrain_Final: %s, Train: %s, Test: %s\nn=%g (out of %g subjects)' % (COHORTS,TRAIN_FINAL,TRAIN,TEST,TIDs.size,num_subj_cohort) 
    print('(%s) Selecting subjects: %g of %g (%g%%) meet performance/response criteria' % \
         (COHORTS,np.sum(inds),num_subj_cohort,np.round(100*np.sum(inds)/num_subj_cohort)))
    return TIDs, selection_string


def Plot_End_Order(p, Behaviors , Neurals):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon
    
    import nn_plot_functions    as npf

    # check TIDs #
    inds_n            = np.isin(Neurals['TIDs'],Behaviors['TIDs'])
    TIDs_plot         = np.array(Neurals['TIDs'])[inds_n]
    inds_b            = np.isin(Behaviors['TIDs'],TIDs_plot)
    assert np.all( np.array(Neurals['TIDs'])[inds_n] == np.array(Behaviors['TIDs'])[inds_b] )

    # get model Variant #
    Model_variant   = Neurals['Model_variant'][inds_n]    

    # End order behavior
    RT          = Behaviors['Eor']['RT'] #  [first_last][choice,model]
    # RT          = Behaviors['Eor_nf']['RT'] #  [first_last][choice,model]
    EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])   # RTs:  (1st - 2nd) / (1st + 2nd)
    bvals       = EO_index.flatten()[inds_b]

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(200+1350,100,500,600) 
    fig.suptitle( 'End-order', fontsize=18, weight='bold')

    # scatter #
    ax = fig.add_subplot(2,1,1)  

    # if p.d1 == 20:
    #     numbins = 32
    #     bins = np.linspace(-1,1,numbins)        
    # else:
    #     numbins = 20
    #     bins = np.linspace(-1,1,numbins)        
    #     if p.jit > 0:
    #         if p.input_train == 1:
    #             numbins = 15
    #         else:
    #             numbins = 20
    #         bins = np.linspace(-1,1,numbins)        

    numbins = 22
    bins = np.linspace(-1,1,numbins)   

    MAXY = -1
    for v in reversed(range(np.unique(Model_variant).size)):
        if 1:
            maxy = npf.area_histogram(ax,bvals[Model_variant == v],bins,4,variant_color(v),0.95,0.5,100,200,normalize=False)
            if maxy > MAXY:
                MAXY = maxy
        else:
            bins = np.linspace(-1,1,60)
            histout = plt.hist( bvals[Model_variant == v], bins, ec=variant_color(v,0.8), fc=variant_color(v,0.5), histtype='stepfilled', linewidth=3.5)

    if p.input_train == 1:
        if MAXY > 50:
            if p.jit == 0:
                MAXY = 20*np.ceil(MAXY/20)
                tick = 20
            else:
                MAXY = 75
                tick = 25
        else:
            MAXY = 40
            tick = 10
    else:
        if MAXY > 100:
            # MAXY = 50*np.ceil(MAXY/50)
            # tick = 50
            MAXY = 125
            tick = 25
            
        elif MAXY > 10:
            MAXY = 20*np.ceil(MAXY/20)
            tick = 20
        else:
            MAXY = 10*np.ceil(MAXY/10)
            tick = 5

    # MAXY = 0.5

    ax.plot([0,0],[0,MAXY],linewidth=4,color=(0.8,0.8,0.8),zorder=-1)

    ax.set_yticks(np.arange(0,MAXY+tick,tick))
    ax.set_ylim([0,MAXY])
    ax.set_xlabel('End order index')


    if p.d1 == 20:
        # ax.set_xlim([-0.75,0.75])
        # ax.set_xticks(np.arange(-.75,1,0.25))
        # ax.set_xticklabels(['-0.75','','','0','','','0.75'])
        ax.set_xlim([-1,1])
        ax.set_xticks(np.arange(-1,1+.25,0.25))
        ax.set_xticklabels(['-1','','','','0','','','','1'])
    else:
        # ax.set_xlim([-0.8,0.8])
        # ax.set_xticks(np.arange(-.8,.8+0.2,0.2))
        # ax.set_xticklabels(['-0.8','','','','0','','','','0.8'])
        ax.set_xlim([-1,1])
        ax.set_xticks(np.arange(-1,1+.25,0.25))
        ax.set_xticklabels(['-1','','','','0','','','','1'])

    # ax.set_xlim([-0.55,0.55])
    # ax.set_xticks(np.arange(-.5,.75,0.25))
    # ax.set_xticklabels(['-0.5','','0','','0.5'])

    ax.set_ylabel('Count')

    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    


def Plot_End_Order_Human(p, Behaviors , TIDs_plot='', titlestring = ''):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    import nn_plot_functions    as npf

    if len(TIDs_plot) == 0:
        TIDs_plot   = np.array(Behaviors['TIDs'])
        # inds        = np.array([0])  # plot the first one
        inds        = np.full( (np.asarray( Behavior['TIDs'] ).size), True)
    else:
        inds        = np.where( np.isin(np.asarray(Behaviors['TIDs']),TIDs_plot) )[0]   

    # End order behavior
    RT          = Behaviors['Eor']['RT'] #  [first_last][choice,model]
    EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])   # RTs:  (1st - 2nd) / (1st + 2nd)
    bvals       = EO_index.flatten()[inds]

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(200,500,500,600) 
    fig.suptitle( '%s' % (titlestring), fontsize=15, weight='bold')

    # scatter #
    ax = fig.add_subplot(2,1,1)  

    if 1:
        bins = np.linspace(-1,1,30)
        maxy = npf.area_histogram(ax,bvals,bins,4,'k',0.95,0.5,100,200)
    else:
        bins = np.linspace(-1,1,60)
        histout = plt.hist( bvals, bins, ec='k', fc='k', histtype='stepfilled', linewidth=3.5)

    if 0:
        MAXY = 10+10*np.ceil(maxy/10)
        YTICK = 10
    elif 1:
        MAXY = 75
        YTICK = 25

    ax.plot([0,0],[0,MAXY],linewidth=4,color=(0.8,0.8,0.8),zorder=-1)
    
    ax.set_yticks(np.arange(0,MAXY+YTICK,YTICK))
    ax.set_ylim([0,MAXY])
    ax.set_xlabel('End order index')

    ax.set_xlim([-1,1])
    ax.set_xticks(np.arange(-1,1+.25,0.25))
    ax.set_xticklabels(['-1','','','','0','','','','1'])
    ax.set_ylabel('# subjects')

    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    
    

def Report_End_Order_Stats(p, data, Behaviors , Blocks_toplot, TIDs_plot='', titlestring = ''):

    from scipy.stats import ranksums

    if len(TIDs_plot) == 0:
        TIDs_plot   = np.array(Behaviors['TIDs'])
        # inds        = np.array([0])  # plot the first one
        inds        = np.full( (np.asarray( Behavior['TIDs'] ).size), True)
    else:
        inds        = np.where( np.isin(np.asarray(Behaviors['TIDs']),TIDs_plot) )[0]   

    # End order behavior
    RT          = Behaviors['Eor']['RT'] #  [first_last][choice,model]
    EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])   # RTs:  (1st - 2nd) / (1st + 2nd)
    bvals       = EO_index.flatten()[inds]

    numsubjects = bvals.size

    #### I. End order vs. 0 #########
    _, pval              = scipy.stats.wilcoxon( bvals )
    print('End order index vs. 0 : p = %g' % (pval) )

    #### II. Individual subjects ######## xxx
    Versions = np.full((numsubjects),np.nan)
    Pvalues = np.full((numsubjects),np.nan)
    for s in range(numsubjects):

        PID     = TIDs_plot[s]
        EOI     = bvals[s]    

        # get subject's data #
        dat             = data[ data[:,1] == PID, :]         # one subject's data
        # get trial types #
        ind_end_1st     = np.logical_and(  np.isin( dat[:,9], [1,7] ), np.isin( dat[:,10], [2,3,4,5,6] )     ) 
        ind_end_2nd     = np.logical_and(  np.isin( dat[:,10], [1,7] ), np.isin( dat[:,9], [2,3,4,5,6] )     ) 
        # get correct blocks #
        BLOCK_INDS    = np.isin( dat[:,5], Blocks_toplot )
        
        # calculate average RT #
        RT_end_1st          = dat[np.logical_and(BLOCK_INDS,ind_end_1st),13].astype('float64')
        RT_end_2nd          = dat[np.logical_and(BLOCK_INDS,ind_end_2nd),13].astype('float64')
    
        _, pval              = ranksums( RT_end_1st, RT_end_2nd )
        Pvalues[s]           = pval
        
        if np.mean(RT_end_1st) < np.mean(RT_end_2nd):
            Versions[s]      = 1
        else:
            Versions[s]      = 2
    
        print('PID: %g, version: %g, p = %g' % (PID, Versions[s], pval) )

    P_THRESH = 0.01
    num1st = np.sum(Versions==1)
    num2nd = np.sum(Versions==2)
    numsig_1st = np.sum(np.logical_and(Versions==1,Pvalues < P_THRESH))
    numsig_2nd = np.sum(np.logical_and(Versions==2,Pvalues < P_THRESH))
    print('%g of %g subjects (%g%%) show 1st-faster' % (num1st, numsubjects, np.round( 100*num1st/numsubjects) ) )
    print('%g of %g subjects (%g%%) show 2nd-faster' % (num2nd, numsubjects, np.round( 100*num2nd/numsubjects)) )
    print('%g of %g 1st-faster subjects (%g%%) are significant at %g rank-sum' % (numsig_1st, num1st,np.round(100*numsig_1st/num1st), P_THRESH ))
    print('%g of %g 2nd-faster subjects (%g%%) are significant at %g rank-sum' % (numsig_2nd, num2nd,np.round(100*numsig_2nd/num2nd), P_THRESH ))
    

            



def Human_Performance(p, Behaviors , TIDs_plot='', titlestring = ''):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    import nn_plot_functions    as npf

    if len(TIDs_plot) == 0:
        TIDs_plot   = np.array(Behaviors['TIDs'])
        # inds        = np.array([0])  # plot the first one
        inds        = np.full( (np.asarray( Behavior['TIDs'] ).size), True)
    else:
        inds        = np.where( np.isin(np.asarray(Behaviors['TIDs']),TIDs_plot) )[0]   
        inds_subj   = np.isin(np.asarray(Behaviors['TIDs']),TIDs_plot)
        version     = Behaviors['VER'][inds[0]]

    numtotalsubjects    = inds.size

    ### I. Print Perf on Initial Test ############

    #   1st presented critical pair    #
    perfs_1st           = Behaviors['Perfs_1st']            #  [Perf_1st_critical, Perf_avg_critical, Perf_avg_critical_all, pvalue, pvalue_all] 
    ind_respond         = ~np.isnan(perfs_1st[0,:] )        #  trials where subjects chose
    ii                  = np.logical_and(ind_respond,inds_subj)  
    numrespond          = np.sum( ii ) 
    numcorrect          = np.sum( perfs_1st[0,ii] )
    pvalue              = scipy.stats.binom_test(numcorrect, numtotalsubjects, p=0.5, alternative='two-sided')

    #  1st critical pairs, initial order presented, average #
    propcorrect_criticals_1 = perfs_1st[1,inds_subj] 
    _, pvalue_1              = scipy.stats.wilcoxon(propcorrect_criticals_1 - 0.5)
    meancriticals_1           = np.mean(propcorrect_criticals_1)
    semcriticals_1            = np.std(propcorrect_criticals_1) / np.sqrt( propcorrect_criticals_1.size )

    # 1st critical pairs, both orders #
    propcorrect_criticals_2 = perfs_1st[2,inds_subj] 
    _, pvalue_2              = scipy.stats.wilcoxon(propcorrect_criticals_2 - 0.5)
    meancriticals_2           = np.mean(propcorrect_criticals_2)
    semcriticals_2            = np.std(propcorrect_criticals_2) / np.sqrt( propcorrect_criticals_2.size )

    print('(%s) 1st Critical : %g of %g subjects (%g%%) (p = %g, Binomial)' % (version,numcorrect,numtotalsubjects,np.round(100*numcorrect/numtotalsubjects),pvalue_1) )
    print('(%s) Avg Critical (6)  : %0.2f ± %f s.e.m. (p = %g, Wilcoxon) across subjects' % (version,meancriticals_1,semcriticals_1, pvalue_1) )
    print('(%s) Avg Critical (12) : %0.2f ± %f s.e.m. (p = %g, Wilcoxon) across subjects' % (version,meancriticals_2,semcriticals_2, pvalue_2) )

    # II. Across block performance #
    perfs_blocks    = Behaviors['Perfs_Blocks'][:,:,inds_subj] #  [<train,test,critical>,block,subject]

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(50,100,400,800) 
    fig.suptitle( '%s' % (titlestring), fontsize=15, weight='bold')

    ax = fig.add_subplot(2,1,1)  

    train  = np.mean( perfs_blocks[0,:,:], axis=1)
    test   = np.mean( perfs_blocks[1,:,:], axis=1)
    probe  = np.mean( perfs_blocks[2,:,:], axis=1)
    train_err  = 2 * np.std( perfs_blocks[0,:,:], axis=1) / np.sqrt(numtotalsubjects)
    test_err   = 2 * np.std( perfs_blocks[1,:,:], axis=1) / np.sqrt(numtotalsubjects)
    probe_err  = 2 * np.std( perfs_blocks[2,:,:], axis=1) / np.sqrt(numtotalsubjects)

    clr_test    = (.7,.7,.7) 
    clr_train   = (.3,.1,.4)
    YMIN     = 0
    ALPHA    = 0.30
    probeclr = 'r'

    xvec   = np.arange(1,train.size+1)

    ax.plot( xvec, train , color=clr_train, linewidth=1.5)
    ax.plot( xvec, test , color=clr_test, linewidth=1.5)
    # ax.plot( xvec, train , color='k', linewidth=1.5)
    # ax.plot( xvec, test , color='k', linewidth=1.5)
    # ax.plot( xvec, probe , color=probeclr, linewidth=3)

    ax.scatter( xvec, train , marker='o', edgecolor=None, s=40, color=clr_train, linewidth=1.5,zorder=100)
    ax.scatter( xvec, test , marker='o', edgecolor=None, s=40, color=clr_test, linewidth=1.5,zorder=100)
    # ax.scatter( xvec, probe , marker='o', s=45, color=probeclr, linewidth=3)

    ax.fill_between(xvec, train-train_err, train+train_err, color=clr_train,  edgecolor=None,      alpha=ALPHA)
    ax.fill_between(xvec, test-test_err, test+test_err, color=clr_test,       edgecolor=None,      alpha=ALPHA)
    # ax.fill_between(xvec, probe-probe_err, probe+probe_err, color=probeclr,alpha=0.2)

    ax.plot([.85,9.15],[.5,.5],'--',linewidth=2,color=(0.8,0.8,0.8),zorder=-1,alpha=1)
    ax.set_ylim([YMIN,1])
    ax.set_xlabel('Block')
    ax.set_xlim([.85,9+.15])
    ax.set_xticks(np.arange(1,10,1))
    # ax.set_xticklabels(['-0.75','','','0','','','0.75'])
    ax.set_ylabel('% correct')
    ax.set_yticks(np.arange(YMIN,1.1,.1))
    # ax.set_yticklabels(['','','0.5','','','','','1'])


    # III. Plot Testing phase performance #
    ax = fig.add_subplot(2,1,2)  
    bins            = np.arange(0,1+0.05,0.05)
    trains_final    =  perfs_blocks[0,2,:]  # average across blocks #  [3,block,model]
    trains_early    = np.nanmean( perfs_blocks[0,3:6,:], axis=0)  # average across blocks #  [3,block,model]
    tests_early     = np.nanmean( perfs_blocks[1,3:6,:], axis=0)  
    probes_early    = np.nanmean( perfs_blocks[2,3:6,:], axis=0)
    trains_late    = np.nanmean( perfs_blocks[0,6:9,:], axis=0)  # average across blocks #  [3,block,model]
    tests_late     = np.nanmean( perfs_blocks[1,6:9,:], axis=0)  
    probes_late    = np.nanmean( perfs_blocks[2,6:9,:], axis=0)
    
    clr_test    = (.7,.7,.7) 
    clr_train   = (.3,.1,.4)
    colors = [clr_train, clr_train, clr_test, clr_train, clr_test ]
    # colors = ['k','k','k','k','k','k','k']
    data =  (trains_final,trains_early,tests_early,trains_late,tests_late)
    xvec = [1,3,4,6,7]
    parts = ax.violinplot(data,xvec,\
                            bw_method = 0.25, widths = 0.6, showmedians=False,  showmeans=False, showextrema=True)
    for i,pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        # pc.set_edgecolor('k')
        pc.set_linewidth(None)
        pc.set_alpha(ALPHA)
    for partname in ('cbars','cmins','cmaxes'):
        vp = parts[partname]
        vp.set_edgecolor('k')
        vp.set_linewidth(1.5)

    quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75], axis=1)
    whiskers = np.array([
        adjacent_values(sorted_array, q1, q3)
        for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

    ax.scatter( xvec, medians,      marker='o',     color='w', s=70, edgecolor='k', linewidth=2, zorder=3)
    ax.vlines(  xvec, quartile1,    quartile3,      color='k', linestyle='-', lw=5)
    # ax.vlines(  xvec, whiskers_min, whiskers_max,   color='k', linestyle='-', lw=2)
    ax.plot(  [-0.25,8.25], [0.5,0.5], '--',  color=(0.8,0.8,0.8), lw=2, alpha=1, zorder=-20)

    ax.set_xlim( [-0.25,8.25] )
    ax.set_xticks(np.arange(0,9,1))
    ax.set_yticks(np.arange(YMIN,1.1,.1))
    ax.set_xticklabels(['','Train\nlate','','Train\nearly','Test\nearly','','Train\nlate','Test\nlate',''],fontsize=14)
    # ax.set_yticklabels(['','','','0.5','','','','','1'])
    ax.set_ylim([YMIN,1])
    ax.set_ylabel('% correct')


    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    


def Verbal_Report(p, Behaviors , Debrief, TIDs_plot=''):

    import pandas as pd

    if len(TIDs_plot) == 0:
        TIDs_plot   = np.array(Behaviors['TIDs'])
        # inds        = np.array([0])  # plot the first one
        inds        = np.full( (np.asarray( Behavior['TIDs'] ).size), True)
    else:
        inds        = np.where( np.isin(np.asarray(Behaviors['TIDs']), TIDs_plot) )[0]   
        inds_subj   = np.isin(np.asarray(Behaviors['TIDs']),TIDs_plot)
        version     = Behaviors['VER'][inds[0]]

    numtotalsubjects    = inds.size

    # I. Get performance on 1st presentation of critical trials (either order)  ############
    perfs_1st           = Behaviors['Perfs_1st']            #  [Perf_1st_critical, Perf_avg_critical, Perf_avg_critical_all, pvalue, pvalue_all] 
    ind_respond         = ~np.isnan(perfs_1st[0,:] )        #  trials where subjects chose
    propcorrect_criticals_2 = perfs_1st[2,inds_subj] 
    numcritical_correct     = 12 * propcorrect_criticals_2

    # II. Get performances from Late Testing (Blocks 7,8,9) #####
    perfs_blocks    = Behaviors['Perfs_Blocks'][:,:,inds_subj] #  [<train,test,critical>,block,subject]
    # train  = np.mean( perfs_blocks[0,:,:], axis=1)
    # test   = np.mean( perfs_blocks[1,:,:], axis=1)
    # critical  = np.mean( perfs_blocks[2,:,:], axis=1)
    # trains_late    = np.nanmean( perfs_blocks[0,6:9,:], axis=0)  # average across blocks #  [3,block,model]
    tests_late     = np.nanmean( perfs_blocks[1,6:9,:], axis=0)  
    critical_late    = np.nanmean( perfs_blocks[2,6:9,:], axis=0)    

    # ranking = np.argsort(tests_late,)[::-1]   # sort in descending order
    ranking = np.lexsort((numcritical_correct,tests_late))[::-1]   # sort in descending order

    Responses = np.full((TIDs_plot.size,6),np.nan, dtype=object)
    Responses[:,0]  = TIDs_plot[ranking]                    # TID  
    Responses[:,1]  = np.round(100*tests_late[ranking])                   # Test perf, late blocks
    Responses[:,2]  = np.round(100*critical_late[ranking])                # Critical perf, late blocks
    Responses[:,3]  = numcritical_correct[ranking]          # Critical perf, initial trials

    for s in range(TIDs_plot.size):
        TID         = Responses[s,0]
        ind         = np.where( np.isin( Debrief[:,1], TID ) )[0]   
        Responses[s,4]  = Debrief[ind,4][0]   # choice strategy
        Responses[s,5]  = Debrief[ind,9][0]   # learning strategy

    df = pd.DataFrame(Responses)
    new_column_names = ['TID', 'Test', 'Critical', 'Critical initial','Choice strategy','Learning strategy']
    df.columns = new_column_names

    return df






def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def Plot_Human_Subject_End_Order_RT(data, Behaviors, Blocks_toplot, TIDs_plot='', titlestring=''):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    from scipy.stats import wilcoxon
    from scipy.stats import ranksums

    # 1. Identify TIDs #
    if len(TIDs_plot) == 0:
        TIDs_plot   = np.array(Behaviors['TIDs'])
        inds        = np.full( (np.asarray( Behavior['TIDs'] ).size), True)
    else:
        inds        = np.where( np.isin(np.asarray(Behaviors['TIDs']),TIDs_plot) )[0]   

    # 2. End order data #
    RT          = Behaviors['Eor']['RT'] #  [first_last][choice,model]
    EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])   # RTs:  (1st - 2nd) / (1st + 2nd)
    vals        = EO_index.flatten()[inds]          # EO index values (for Early Testing phase)

    # 3. Sort #
    inds_sort        = np.argsort( vals )

    # 3.5 Inds to plot
    if 1:       # sorted
        inds_plot    = inds_sort
    else:       # random
        inds_plot    = np.random.permutation(vals.size)

    # 4. Plot #
    fig = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(50,50,2600,400) 
    # fig.suptitle('Transitive Inference\n RT vs. end-item order, examples of individual subjects\n\n', fontsize=17, weight='bold')

    numexamples = 20


    datastr = 'RT'
    YLABEL  = 'RT'
    Y_MIN = 0
    Y_MAX = 2000
    YTICKS = np.arange(0,2500,500)
    # YTICKLABELS = ['0','','','','','','','','','','1']

    for type in range(2):

        if type == 0:
            Inds = inds_plot
        else:   
            Inds = np.flip(inds_plot)
        
        for ex in range(numexamples):

            PID     = TIDs_plot[Inds[ex]]
            EOI     = vals[Inds[ex]]

            # get subject's data #
            dat             = data[ data[:,1] == PID, :]         # one subject's data

            # basic trial types #
            ind_end_1st     = np.logical_and(  np.isin( dat[:,9], [1,7] ), np.isin( dat[:,10], [2,3,4,5,6] )     ) 
            ind_end_2nd     = np.logical_and(  np.isin( dat[:,10], [1,7] ), np.isin( dat[:,9], [2,3,4,5,6] )     ) 

            # calculate average performance #
            BLOCK_INDS    = np.isin( dat[:,5], Blocks_toplot )
            Perf_1st      = np.mean( dat[np.logical_and(BLOCK_INDS,ind_end_1st),16]  )
            Perf_2nd      = np.mean( dat[np.logical_and(BLOCK_INDS,ind_end_2nd),16]  )
            # calculate average RT #
            RT_end_1st          = dat[np.logical_and(BLOCK_INDS,ind_end_1st),13].astype('float64')
            RT_end_2nd          = dat[np.logical_and(BLOCK_INDS,ind_end_2nd),13].astype('float64')
            # RT_end_1            = np.nanmean( RT_end_1st  )
            # RT_end_2            = np.nanmean( RT_end_2nd  )

            ax      = fig.add_subplot(2,numexamples,type*numexamples+ex+1)

            meanvals = np.array([np.nan,np.nan])
            for gg in [0,1]:       # 0: 1st, 1: 2nd
                if gg == 0:         
                    clr1    = 'purple'
                    values  = RT_end_1st
                elif gg == 1:                
                    clr1 = 'orange'
                    values  = RT_end_2nd
                xvals = 0.3*(np.random.rand(values.size)-0.5) + 1 + gg
                ax.scatter(xvals, values, s=50, alpha=0.3, color=clr1,zorder=10)  # performance
                # mean s.d, #
                meanval = np.nanmean( values )
                meanvals[gg] = meanval
                sdval   = np.nanstd( values )
                # sdval   = np.std( values[inds_valid,gg] ) / np.sqrt(np.sum(inds_valid))
                ax.scatter(1+gg, meanval, marker='o', s=80, edgecolors='k', facecolors='None', alpha=1, zorder=10)  # performance
                ax.plot([1+gg,1+gg],[meanval-sdval,meanval+sdval],color='k',alpha=0.5,zorder=11  )
            ax.plot([1,2],[meanvals[0],meanvals[1]],color='k',linewidth=8,alpha=0.2,zorder=11  )

            if ex == 0:
                ax.set_ylabel('RT',fontsize=18)
            ax.set_xlim([0,3])
            ax.set_ylim([Y_MIN,Y_MAX])
            ax.set_xticks([1,2])
            ax.set_xticklabels(['1st','2nd'],fontsize=14)
            # ax.set_xticklabels(['',''],fontsize=14)
            ax.set_yticks(YTICKS)
            # ax.axes.yaxis.set_ticklabels(YTICKLABELS,fontsize=14)
            ax.axes.yaxis.set_ticklabels([],fontsize=14)
            _,pval = ranksums(RT_end_1st,RT_end_2nd)
            ax.set_title('PID:%g\n%0.2f, p=%0.1g' % ((PID,EOI,pval)),fontsize=10)
            # ax.set_title('%0.2f' % ((B[ii])),fontsize=12)

        fig.subplots_adjust(hspace=0.4)

    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()







def Noise_and_NR_report(p,Behaviors,jobname):
    
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 16})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon
    
    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(120,900,600,400) 

    #### I.  Noise levels ########################
    ax = fig.add_subplot(2,1,1)  
    numBins     = 60
    Xmax        = 10.0001
    bins        = np.linspace(0,Xmax,numBins)
    xticks      = np.arange(0,Xmax,1)
    Ymax = 0
    vals = np.array(Behaviors['noise_sigs'])
    for v in reversed(range(5)):
        inds = (np.array([Behaviors['model_variant']]) == v).flatten()
        ymax = npf.area_histogram(ax, vals[inds], bins, 3, variant_color(v), 0.9, 0.2, 100, 200)
        if ymax > Ymax:
            Ymax = ymax

    ax.set_yticks(np.arange(0,Ymax+20,20))
    ax.set_ylim([0,Ymax+5.000001])
    ax.set_xlabel('Noise level')
    ax.set_ylabel('# of instances')
    ax.set_xticks(xticks)
    ax.set_title( '%s\nNoise levels' % (jobname), fontsize=20, weight='bold')
    ax.set_xlim([bins[0],bins[-1]])

    #### II.  No response proportions ########################
    ax = fig.add_subplot(2,1,2)  
    numBins     = 50
    Xmax        = 1
    bins        = np.linspace(0,Xmax,numBins)
    xticks      = np.arange(0,Xmax+0.1,0.1)
    Ymax = 0
    vals = np.array(Behaviors['no_response'])
    for v in reversed(range(5)):
        inds = (np.array([Behaviors['model_variant']]) == v).flatten()
        ymax = npf.area_histogram(ax, vals[inds], bins, 3, variant_color(v), 0.9, 0.2, 100, 200)
        if ymax > Ymax:
            Ymax = ymax

    ax.set_xticks(xticks)
    ax.set_yticks(np.arange(0,Ymax+20,20))
    ax.set_ylim([0,Ymax+5.000001])
    ax.set_xlabel('No-response proportions')
    ax.set_ylabel('# of instances')
    ax.set_xlim([bins[0],bins[-1]])

    NR_mean        = np.mean(Behaviors['no_response'])
    NR_std         = np.std(Behaviors['no_response'])
    print('no response: %0.2f ± %0.2f (s.d.)' % (NR_mean,NR_std))

    plt.pause(0.1)
    plt.ion()
    plt.ioff()
    plt.ion()



def Plot_End_vs_GI(p, Behaviors, Neurals):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 10})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    # check TIDs #
    inds_n            = np.isin(Neurals['TIDs'],Behaviors['TIDs'])
    TIDs_plot         = np.array(Neurals['TIDs'])[inds_n]
    inds_b            = np.isin(Behaviors['TIDs'],TIDs_plot)
    assert np.all( np.array(Neurals['TIDs'])[inds_n] == np.array(Behaviors['TIDs'])[inds_b] )

    # get model Variant indices #
    Model_variant   = Neurals['Model_variant'][inds_n]    

    # Calculate Lexical Marking index #
    RT          = Behaviors['Eor']['RT'] #  [first_last][choice,model] Eor
    EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])
    bvals       = EO_index.flatten()[inds_b]
    
    # Collinearity + Dimensionality #
    for Index in range(1):

        fig = plt.figure()
        plt.get_current_fig_manager().window.setGeometry(200+1350*Index,100,1500,400) 
        fig.suptitle( 'Behavior (End order) x Neural (A index) study', fontsize=18, weight='bold')

        for EA in [0,1]:  # exact, average

            if Index == 0:
                DT = 0
                N   = Neurals['GI'][DT,:,EA,inds_n]              # [version,<full,quarter>,model]
                # vals = Neurals['GI'][DT,vv,EA,inds_n]
                numversions = N.shape[0]
                numdelays = N.shape[1]
                neuralstrs = ['Collinearity, early','Collinearity, late', 'Collinearity, change','OCI, early','OCI, late','OCI, change','Mean angle change','Mean distance change']

            # elif Index == 1:
            #     N  = Neurals['Neff_win'][:,inds_n].T            # [version,model]
            #     numversions = N.shape[0]
            #     numdelays = 1
            #     neuralstrs = ['1st','2nd','3rd','4th','Last','4th/1st']

            indices_toplot = [0,1,2,3,4,5,8,9]

            for V in range(len(indices_toplot)):    # 0: opposite, 1: alignment, 2: end-align

                vv = indices_toplot[V]

                xticklabels = []
                
                if Index == 0:
                    nvals = N[:,vv]
                    if V == 0 or V == 1:
                        xlims = [0,1]            
                        xticks = np.arange(0,1+.25,0.25)            
                    elif V == 2:
                        if p.input_train == 1:
                           xlims = [-0.6,0.6]            
                           xticks = np.arange(-0.6, 0.6+0.3, 0.3)     
                        else:
                           xlims = [-1,1]            
                           xticks = np.arange(-1, 1+.25, 0.25)     
                    elif V == 3 or V == 4:
                        xlims = [-0.5,1]            
                        xticks = np.arange(-0.5, 1+.25, 0.25)   
                        xticklabels = []
                    elif V == 5:
                        if p.input_train == 1:
                            xlims = [-0.6,0.6]            
                            xticks = np.arange(-0.6, 0.6+0.3, 0.3)     
                            xticklabels = []
                        else:                            
                            xlims = [-1,1]            
                            xticks = np.arange(-1, 1+0.25, 0.25)     
                            xticklabels = []
                    elif V == 6:
                        # xlims = [-0.15,+0.15] 
                        # xticks = np.arange(-0.15,0.20,0.05)           
                        # xticklabels = ['-0.15','','','0','','','0.15']        
                        xlims = [-0.2,+0.2] 
                        xticks = np.arange(-0.2,0.25,0.05)           
                        xticklabels = []        
                    elif V == 7:
                        xlims = [-5,+15]            
                        xticks = np.arange(-5,15+5,+5)           
                elif Index == 1:
                    if vv < 5:
                        nvals = N[:,vv]
                        xlims = [0,8]
                    elif vv == 5:
                        nvals = N[:,4] / N[:,0]
                        xlims = [0,5]
                    else:
                        break

                # scatter #
                if np.all(~np.isnan(nvals)):  
                    ax = fig.add_subplot(2,8,EA*8+V+1)  

                    # guide lines #
                    # if vv == 0 and Index == 0:
                    #     ax.plot([0,1],[0,0],'-',linewidth=3,color='k',alpha=0.1)
                    #     ax.set_aspect(0.5)  
                    if  Index == 0:
                        ax.plot([xlims[0],xlims[1]],[0,0],'-',linewidth=3,color=(0.8,0.8,0.8),zorder=-1)
                        
                        if 0:                    
                            ax.set_ylim([-0.55,.55])
                            ax.set_yticks(np.arange(-0.5,0.75,.25))
                            ax.set_aspect((xlims[1]-xlims[0])/(1.1))    
                        elif 1:
                            ax.set_ylim([-.75,.75])
                            ax.set_yticks(np.arange(-.75,1,.25))
                            ax.set_aspect((xlims[1]-xlims[0])/1.5)    

                        if V >= 2:
                            ax.plot([0,0],[-.75,.75],'-',linewidth=3,color=(0.8,0.8,0.8),zorder=-1)

                    ax.set_xlim(xlims)
                    ax.set_xticks(xticks)
                    if len(xticklabels)!=0:
                        ax.set_xticklabels(xticklabels,fontsize=10)

                    # for m in reversed(range(nvals.size)):
                    for m in range(nvals.size):
                        v = Model_variant[m]
                        ax.scatter(nvals[m],bvals[m],color=variant_color(v),edgecolors=None,alpha=0.1)
                        # ax.scatter(nvals[m],bvals[m],s=30,edgecolors=variant_color(v,.35),color=variant_color(v,.35))

                    ax.set_title('%s' % (neuralstrs[V]),fontsize=12)
                    if vv == 0:
                        ax.set_ylabel('Behavior index',fontsize=9)
                    if vv > 0:
                        ax.set_yticklabels([])

        fig.subplots_adjust(hspace=0.5)
        plt.draw()     
        plt.show()
        plt.pause(0.0001)
        plt.ion()
        plt.ioff()
        plt.ion()    



def Plot_End_vs_ERI(p, Behaviors, Neurals):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 16})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    # check TIDs #
    inds_n            = np.isin(Neurals['TIDs'],Behaviors['TIDs'])
    TIDs_plot         = np.array(Neurals['TIDs'])[inds_n]
    inds_b            = np.isin(Behaviors['TIDs'],TIDs_plot)
    assert np.all( np.array(Neurals['TIDs'])[inds_n] == np.array(Behaviors['TIDs'])[inds_b] )

    # get model Variant indices #
    # Model_variant   = Neurals['Model_variant'][inds_n]    


    fig = plt.figure()

    plt.get_current_fig_manager().window.setGeometry(200+1350,100,1500,400) 
        
    fig.suptitle( 'EOE vs. ERI', fontsize=18, weight='bold')

    for EE in [0,3]:   # exact, empirical 

        # ax = fig.add_subplot(1,4,1+EE)  
        ax = fig.add_subplot(2,7,EE+1)  

        for vv in range(5):    

            #### i. End Order index ####
            if EE == 0:
                # RT          = Behaviors['Eor_nf']['RT'] #  [first_last][choice,model] Eor                
                RT          = Behaviors['Eor']['RT'] #  [first_last][choice,model] Eor
            elif EE > 0:
                RT          = Behaviors['Eor']['RT'] #  [first_last][choice,model] Eor

            if p.d1 <= 20:
                if p.input_train == 1 and EE == 0:
                    xlims       = [-1,2+0.0001]
                    xtick       = 0.5
                else :
                    xlims       = [-2,4+0.0001]
                    xtick       = 1
            else:
                if p.jit == 0:
                    xlims = [-3,6+0.0001]
                    xtick = 1
                else:
                    xlims = [-2,4+0.0001]
                    xtick = 1


            EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])
            bvals       = EO_index.flatten()[inds_b]

            ### ii. ERI  #####
            inds    = Neurals['Model_variant'][inds_n] == vv
            nvals   = np.log10( Neurals['Eri'][EE,inds] )

            ### plot ###
            if 0:
                ax.set_ylim([-0.75,0.75])
                ax.set_yticks(np.arange(-0.75,1,.25))
                # ax.set_ylim([-0.6,0.6])
                # ax.set_yticks(np.arange(-0.6,.6+.2,.2))
            else:
                ax.set_ylim([-1,1])
                ax.set_yticks(np.arange(-1,1.25,.25))
                # ax.set_yticklabels(['-1','','','','0','','','','1'])     


            ax.set_xticks(np.arange(xlims[0],xlims[1],xtick))
            ax.set_xlim(xlims)

            ax.scatter(nvals,bvals[inds],color=variant_color(vv),edgecolors=None,alpha=0.15)
            # ax.set_aspect('equal', 'box')  

        y_min, y_max = ax.get_ylim()
        ax.plot([0,0],[y_min,y_max],'-',linewidth=3,color=(0.8,0.8,0.8),zorder=-1)
        ax.plot([xlims[0],xlims[1]],[0,0],'-',linewidth=3,color=(0.8,0.8,0.8),zorder=-1)
        aspect_ratio  =  (xlims[1]-xlims[0]) / (y_max-y_min)
        ax.set_aspect(aspect_ratio)  

        # ax.tick_params('both', length=7,  which='major')
        # ax.tick_params('both', length=7, which='minor')

        ax.set_xlabel('Log10(ERI)')                
        ax.set_ylabel('End order index')  
        # if EE == 0: 
        #     # ax.set_xticklabels(['-1.5','','','0','','','1.5'])     
        #     # ax.set_xticklabels(['-3','','','0','','','3'])     
        #     # ax.set_xticklabels(['-4','','','','0','','','','4'])     
        # else:
        #     # ax.set_xticklabels(['-3','','','0','','','3'])     
        #     # ax.set_xticklabels(['-4','','','','0','','','','4'])     
        fig.subplots_adjust(hspace=0.5)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    


def Plot_End_vs_GI_Both_RNNs(p, B, N):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 9})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon


    neuralstrs = ['Collinearity, early','Collinearity, late', 'Collinearity, change','OCI, early','OCI, late','OCI, change','Mean angle change','Mean distance change']

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(200,100,1500,400) 
    fig.suptitle( 'Behavior (End order) x Neural (A index) study', fontsize=24, weight='bold')

    vv_toplot = [0,1,2,3,4,5,8,9]

    for V in range(len(vv_toplot)):    # 0: opposite, 1: alignment, 2: end-align

        xticklabels = []

        vv = vv_toplot[V]

        if V == 0 or V == 1:
            xlims = [0,1]            
            xticks = np.arange(0,1+.25,0.25)            
        elif V == 2:
            xlims = [-1,1]            
            xticks = np.arange(-1, 1+0.25, 0.25)     
            xticklabels = [] 
        elif V == 3 or V == 4:
            xlims = [-0.5,1]            
            xticks = np.arange(-0.5, 1+.25, 0.25)   
            xticklabels = []
        elif V == 5:
            if p.d1 <= 20:
                xlims = [-1,1]            
                xticks = np.arange(-1, 1+0.25, 0.25)     
                xticklabels = []
            else:
                xlims = [-1.5,1.5]            
                xticks = np.arange(-1.5, 2, 0.5)     
                xticklabels = []
        elif V == 6:
            if p.d1 <= 20:
                xlims = [-0.20,+0.20] 
                xticks = np.arange(-0.20,0.25,0.05)           
            else:
                xlims = [-0.4,+0.4] 
                xticks = np.arange(-0.4,0.6,0.2)           
            xticklabels = []        
        elif V == 7:
            if p.d1 <= 20:
                xlims = [-5,+15]            
                xticks = np.arange(-5,15+5,+5)           
            else:
                xlims = [-5,+20]            
                xticks = np.arange(-5,20+5,+5)           

            
        # scatter #
        ax = fig.add_subplot(2,8,V+1)  

        for ARCH in [0,1]:  # fRNN and rRNN

            if ARCH == 0:  # f-RNN
                clr = 'k'
                alpha = 0.075
                clr = (0,0,0,.25)
            else:          # r-RNN
                clr = 'cornflowerblue'
                vals  = mpl.colors.to_rgb(clr)
                alpha = 0.075
                clr = (vals[0],vals[1],vals[2],.17)

            # check TIDs #
            inds_n            = np.isin(N[ARCH]['TIDs'],B[ARCH]['TIDs'])
            TIDs_plot         = np.array(N[ARCH]['TIDs'])[inds_n]
            inds_b            = np.isin(B[ARCH]['TIDs'],TIDs_plot)
            assert np.all( np.array(N[ARCH]['TIDs'])[inds_n] == np.array(B[ARCH]['TIDs'])[inds_b] )

            # # get model Variant indices #
            # Model_variant   = N[ARCH]['Model_variant'][inds_n]    

            # Get behavior#
            RT          = B[ARCH]['Eor']['RT'] #  [first_last][choice,model] Eor
            EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])
            bvals       = EO_index.flatten()[inds_b]
            
            # Get neural #
            DT  = 0
            EA  = 0  # exact
            nvals = N[ARCH]['GI'][DT,vv,EA,inds_n]  #[DT,vv,EA,inds_n]

            if np.all(~np.isnan(nvals)):
                  
                # ax.set_ylim([-0.75,.75])
                # ax.set_yticks(np.arange(-0.75,1,.25))
                # ax.set_aspect((xlims[1]-xlims[0])/(1.5))    

                ax.set_ylim([-1,1])
                ax.set_yticks(np.arange(-1,1.25,.25))
                ax.set_aspect((xlims[1]-xlims[0])/(2))    

                ax.set_xlim(xlims)
                ax.set_xticks(xticks)
                if len(xticklabels)!=0:
                    ax.set_xticklabels(xticklabels)

                # for m in reversed(range(nvals.size)):
                for m in range(nvals.size):
                    if ARCH == 0:
                        ax.scatter(nvals[m],bvals[m],color=clr,edgecolors=clr,alpha=alpha)
                        # ax.scatter(nvals[m],bvals[m],marker='o',s=60,color=clr,edgecolors='None',alpha=alpha)
                    else:
                        # ax.scatter(nvals[m],bvals[m],facecolors=clr,edgecolors=None,alpha=alpha)
                        ax.scatter(nvals[m],bvals[m],facecolors=clr,edgecolors=None,alpha=alpha)
                        # ax.scatter(nvals[m],bvals[m],marker='o',s=60,facecolors='None',edgecolors=clr,alpha=.4)

                ax.set_title('%s' % (neuralstrs[V]),fontsize=9)
                if vv == 0:
                    ax.set_ylabel('Behavior index',fontsize=9)
                if vv > 0:
                    ax.set_yticklabels([])
            ax.plot([xlims[0],xlims[1]],[0,0],'-',linewidth=3,color=(0.8,0.8,0.8),zorder=-1)
            if V >= 2:
                # ax.plot([0,0],[-.75,.75],'-',linewidth=3,color=(0.8,0.8,0.8),zorder=-1)
                ax.plot([0,0],[-1,1],'-',linewidth=3,color=(0.8,0.8,0.8),zorder=-1)


    fig.subplots_adjust(hspace=0.5)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    


def Plot_Wasserstein_End_Order(p, B, N, HB, TIDs_plot):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches   as patch
    
    import nn_plot_functions    as npf
    

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(200+1350,100,500,600) 
    fig.suptitle( 'End-order', fontsize=18, weight='bold')

    # human behavior #
    inds        = np.where( np.isin(np.asarray(HB['TIDs']),TIDs_plot) )[0]   
    RT          = HB['Eor']['RT'] #  [first_last][choice,model]
    EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])   # RTs:  (1st - 2nd) / (1st + 2nd)
    bvals_human = EO_index.flatten()[inds]

    dists = np.full((6,5),np.nan)

    for ARCH in range(6):  # fRNN and rRNN
    
        ax = fig.add_subplot(6,1,ARCH+1)  

        if ARCH < 6:
            # check TIDs #
            inds_n            = np.isin(N[ARCH]['TIDs'],B[ARCH]['TIDs'])
            TIDs_plot         = np.array(N[ARCH]['TIDs'])[inds_n]
            inds_b            = np.isin(B[ARCH]['TIDs'],TIDs_plot)
            assert np.all( np.array(N[ARCH]['TIDs'])[inds_n] == np.array(B[ARCH]['TIDs'])[inds_b] )

            # # get model Variant indices #
            Model_variant   = N[ARCH]['Model_variant'][inds_n]    

            # Get behavior#
            RT          = B[ARCH]['Eor']['RT'] #  [first_last][choice,model] Eor
            EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])
            bvals       = EO_index.flatten()[inds_b]

        else:
            Model_variant = np.zeros(bvals_human.size)
            bvals         = bvals_human

        numbins = 22
        bins = np.linspace(-1,1,numbins)   

        MAXY = -1
        for v in reversed(range(np.unique(Model_variant).size)):
            if 1:
                maxy = npf.area_histogram(ax,bvals[Model_variant == v],bins,4,variant_color(v),0.95,0.5,100,200,normalize=False)
                if maxy > MAXY:
                    MAXY = maxy
            else:
                bins = np.linspace(-1,1,60)
                histout = plt.hist( bvals[Model_variant == v], bins, ec=variant_color(v,0.8), fc=variant_color(v,0.5), histtype='stepfilled', linewidth=3.5)

            # wasserstein #
            if ARCH < 6:
                dists[ARCH,v] = scipy.stats.wasserstein_distance(bvals[Model_variant==v],bvals_human)


        if p.input_train == 1:
            if MAXY > 50:
                if p.jit == 0:
                    MAXY = 20*np.ceil(MAXY/20)
                    tick = 20
                else:
                    MAXY = 75
                    tick = 25
            else:
                MAXY = 40
                tick = 10
        else:
            if MAXY > 100:
                # MAXY = 50*np.ceil(MAXY/50)
                # tick = 50
                MAXY = 125
                tick = 25
                
            elif MAXY > 10:
                MAXY = 20*np.ceil(MAXY/20)
                tick = 20
            else:
                MAXY = 10*np.ceil(MAXY/10)
                tick = 5

        # MAXY = 0.5

        ax.plot([0,0],[0,MAXY],linewidth=4,color=(0.8,0.8,0.8),zorder=-1)

        ax.set_yticks(np.arange(0,MAXY+tick,tick))
        ax.set_ylim([0,MAXY])
        ax.set_xlabel('End order index')


        if p.d1 == 20:
            # ax.set_xlim([-0.75,0.75])
            # ax.set_xticks(np.arange(-.75,1,0.25))
            # ax.set_xticklabels(['-0.75','','','0','','','0.75'])
            ax.set_xlim([-1,1])
            ax.set_xticks(np.arange(-1,1+.25,0.25))
            ax.set_xticklabels(['-1','','','','0','','','','1'])
        else:
            # ax.set_xlim([-0.8,0.8])
            # ax.set_xticks(np.arange(-.8,.8+0.2,0.2))
            # ax.set_xticklabels(['-0.8','','','','0','','','','0.8'])
            ax.set_xlim([-1,1])
            ax.set_xticks(np.arange(-1,1+.25,0.25))
            ax.set_xticklabels(['-1','','','','0','','','','1'])

        # ax.set_xlim([-0.55,0.55])
        # ax.set_xticks(np.arange(-.5,.75,0.25))
        # ax.set_xticklabels(['-0.5','','0','','0.5'])

    ax.set_ylabel('Count')

    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()   

    return dists


def Plot_Wasserstein(dists):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    
    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(200,100,1000,300) 
    fig.suptitle( 'Wasserstein distances', fontsize=18, weight='bold')

    ax = fig.add_subplot(1,1,1)  

    ax.bar(np.arange(30), dists.ravel() )
    ax.set_yticks(np.arange(0,0.7+0.1,0.1))
    ax.set_ylim([0,0.7])
    # ax.set_xlabel('End order index')
    ax.set_xlim([-0.75,29.75])
    ax.set_xticks(np.arange(30))
    # ax.set_xticklabels(['-1','','','','0','','','','1'])

    # ax.set_ylabel('Count')

    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()   

    return dists


def Quantify_GI(p, B, N, EA):

    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 16})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"

    neuralstrs = ['Collinearity, early','Collinearity, late', 'Collinearity, change','OCI, early','OCI, late','OCI, change','Mean angle, early','Mean angle, late', 'Mean angle change','Mean distance change']

    for vv in [0,1,2,3,4,5,8,9]:    # Geometric index, 0: OCI (early), 1: OCI (late), 3: OCI change, 4: Mean angle chance, 5: Euc Dist change

        Vals_arch = [[],[]]
        for ARCH in [0,1]:  # f-RNN and r-RNN

            if ARCH == 0:  # f-RNN
                archstr = 'f-RNN'
                clr = 'k'
                alpha = 0.25
                clr = (0,0,0,.25)
            else:          # r-RNN
                archstr = 'r-RNN'
                clr = 'cornflowerblue'
                vals  = mpl.colors.to_rgb(clr)
                alpha = 0.17
                clr = (vals[0],vals[1],vals[2],.17)

            # check TIDs #
            inds_n            = np.isin(N[ARCH]['TIDs'],B[ARCH]['TIDs'])
            TIDs_plot         = np.array(N[ARCH]['TIDs'])[inds_n]
            inds_b            = np.isin(B[ARCH]['TIDs'],TIDs_plot)
            assert np.all( np.array(N[ARCH]['TIDs'])[inds_n] == np.array(B[ARCH]['TIDs'])[inds_b] )

            # # get model Variant indices #
            Model_variant   = N[ARCH]['Model_variant'][inds_n]    

            # Get behavior#
            RT          = B[ARCH]['Eor']['RT'] #  [first_last][choice,model] Eor
            EO_index    = (RT[0] - RT[1]) / (RT[1] + RT[0])
            bvals       = EO_index.flatten()[inds_b]
            
            # Get neural #

            vals_acrossvariants = []
            for mv in range(6):  # variants 0-4, 5 is random vectors
                if mv < 5:
                    DT  = 0
                    nvals       = N[ARCH]['GI'][DT,vv,EA,inds_n]
                    vals        = nvals[Model_variant == mv] 
                    var_string  = 'variant: %g' % (mv)
                elif mv == 5:
                    DT  = 1
                    nvals       = N[ARCH]['GI'][DT,vv,EA,inds_n]
                    vals        = nvals[Model_variant == 0] 
                    var_string  = 'random vecs'
                meanvals    = np.mean(vals)
                stdvals     = np.std(vals)
                _,p_value = scipy.stats.wilcoxon(vals)
                print('(%s,%s) %s: %0.3f ± %0.3f (mean ± s.d., n=%g, p=%g)' \
                        % (archstr,var_string,neuralstrs[vv],meanvals,stdvals,vals.size,p_value) )
                vals_acrossvariants = np.concatenate((vals_acrossvariants,vals),axis=0)
            allvals = np.asarray(vals_acrossvariants).flatten()

            # across arch #
            _,p_value = scipy.stats.wilcoxon(allvals)
            print('(%s, all variants) %s: %0.3f ± %0.3f (mean ± s.d., n=%g, p=%g)' \
                    % (archstr,neuralstrs[vv],np.mean(allvals),np.std(allvals),allvals.size,p_value) )

    # Specific comparisons #

    EA            = 0   # exact

    # i. Collinearity, early: f-RNN complex vs. f-RNN simples #
    # ARCH        = 0   # 0: f-RNN, 1: r-RNN
    # nvals       = N[ARCH]['GI'][DT,vv,EA,inds_n]
    DT            = 0       # real data
    vv            = 0       # early collinearity
    # f-RNN complex #
    inds_n        = np.isin(N[0]['TIDs'],B[0]['TIDs'])
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals1       = N[0]['GI'][DT,vv,0,inds_n][Model_variant == 4]   # f-RNN complex
    # f-RNN simples #
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals2       = N[0]['GI'][DT,vv,0,inds_n][np.isin(Model_variant,[0,1])]   # f-RNN complex
    print('Collinearity, early: f-RNN complex vs. f-RNN simples')
    print_ranksum(vals1,vals2)

    
    # ii. Collinearity, late: f-RNN complex vs. f-RNN simples #
    # ARCH        = 0   # 0: f-RNN, 1: r-RNN
    # nvals       = N[ARCH]['GI'][DT,vv,EA,inds_n]
    DT            = 0       # real data
    vv            = 1       # late collinearity
    # f-RNN complex #
    inds_n        = np.isin(N[0]['TIDs'],B[0]['TIDs'])
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals1       = N[0]['GI'][DT,vv,0,inds_n][Model_variant == 4]   # f-RNN complex
    # f-RNN simples #
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals2       = N[0]['GI'][DT,vv,0,inds_n][np.isin(Model_variant,[0,1])]   # f-RNN complex
    print('Collinearity, late: f-RNN complex vs. f-RNN simples')
    print_ranksum(vals1,vals2)

    # iii. Collinearity, early: f-RNN complex vs. random #
    # ARCH        = 0   # 0: f-RNN, 1: r-RNN
    # nvals       = N[ARCH]['GI'][DT,vv,EA,inds_n]
    vv            = 0       # late collinearity
    # f-RNN complex #
    inds_n        = np.isin(N[0]['TIDs'],B[0]['TIDs'])
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals1       = N[0]['GI'][0,vv,0,inds_n][Model_variant == 4]   # f-RNN complex
    # f-RNN simples #
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals2       = N[0]['GI'][1,vv,0,inds_n][Model_variant == 0]   # random
    print('Collinearity, early: f-RNN complex vs. random')
    print_ranksum(vals1,vals2)    

    # iv. Collinearity, late: f-RNN complex vs. random #
    # ARCH        = 0   # 0: f-RNN, 1: r-RNN
    # nvals       = N[ARCH]['GI'][DT,vv,EA,inds_n]
    vv            = 1       # late collinearity
    # f-RNN complex #
    inds_n        = np.isin(N[0]['TIDs'],B[0]['TIDs'])
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals1       = N[0]['GI'][0,vv,0,inds_n][Model_variant == 4]   # f-RNN complex
    # f-RNN simples #
    Model_variant = N[0]['Model_variant'][inds_n] 
    vals2       = N[0]['GI'][1,vv,0,inds_n][Model_variant == 0]   # random
    print('Collinearity, late: f-RNN complex vs. random')
    print_ranksum(vals1,vals2)      

    # v. Collinearity, late: r-RNN simple vs. complex #
    # ARCH        = 0   # 0: f-RNN, 1: r-RNN
    # nvals       = N[ARCH]['GI'][DT,vv,EA,inds_n]
    vv            = 1       # late collinearity
    # r-RNN complex #
    inds_n        = np.isin(N[1]['TIDs'],B[1]['TIDs'])
    Model_variant = N[1]['Model_variant'][inds_n] 
    vals1       = N[1]['GI'][0,vv,0,inds_n][Model_variant == 0]   # r-RNN simple
    # f-RNN simples #
    Model_variant = N[1]['Model_variant'][inds_n] 
    vals2       = N[1]['GI'][0,vv,0,inds_n][Model_variant == 4]   # r-RNN complex
    print('Collinearity, late: r-RNN simple vs. complex')
    print_ranksum(vals1,vals2)      


def print_ranksum(vals1,vals2):
    from scipy.stats import ranksums
    _, pval    = ranksums( vals1, vals2 )
    print('%0.3f ± %0.3f vs. %0.3f ± %0.3f (mean ± s.d., n=%g, %g, p=%g)' \
            % (np.mean(vals1),np.std(vals1),np.mean(vals2),np.std(vals2),vals1.size,vals2.size,pval) )

def Calculate_Neurals( jobname, TIDs_process, Model_select, localdir = '' , topPCs=-1, PCA_geometry=0, topPCs_geometry=-1, MAX_INSTANCES=np.inf):

    if len(TIDs_process) == 0:
        print('no TIDs!')
        return

    sys.path.insert(0, localdir[:-1] + jobname)
    from init import singleout, params, genfilename
    p = singleout()                    # instantiate simply to access .dims conveniently
    pdict = dict(vars(params()))

    print('(%s) Calculating Neural measures across models' % (localdir + p.jobname))
    
    # ii. Get stored values from all models ####

    TIDs           = []
    Model_variant  = []
    GI             = []   # [vv,2,model]  Q measure    (mean over simulations)
    Neff_trace     = []   # [vv,model]    Dimensionality
    LD             = []   # list of LD dictionaries
    ANG             = []   # 
    CH             = []   # 
    PC             = []   # list of pca objects
    DATA_G         = []

    # model counts #
    variantcount = np.array([0,0,0,0,0])

    for tt in range(len(TIDs_process)):

        TID               = int(TIDs_process[tt])
        p_TID             = singleout(TID = TID)  # get parms object
        print('(Calculating Neurals) TID: %g' % (TID))
        fname             = genfilename(p_TID,localdir)
        file              = tk.readfile(fname,".p",silent=True)
        if len(file) == 0:
            continue
        F                 = file[0]  # retrieve model
        model, _, i, _     = tk.select_model( p, F, Model_select )

        # check model counts
        v = p_TID.Model_variant
        if variantcount[v] < MAX_INSTANCES:
            variantcount[v] += 1
        else:
            continue

        # store these #
        TIDs.append(TID)
        Model_variant.append(p_TID.Model_variant)

        #   print('TID: %d' % (TID,))

        # Simulate noise-free #
        Data        = tk.Simulate_model( p, F, model, TT_run=[0] )

        # Calculate PCA #
        Data        = tk.Project_Data_PCA(p, F, Data, 0, 0)

        # Delay PCA # 
        Data_G      = tk.Project_Data_Delay_PCA(p, F, Data, 0)

        # Select which PCA to use for geometry and correlation indices #
        if PCA_geometry == 0:
            Data_geometry = Data
            pca           = None
        elif PCA_geometry == 1:
            Data_geometry = Data_G
            pca           = Data_G['pca']      

        # # Calculate Linear dynamics during Delay #
        ld         = Linear_Dynamics_Delay_Period(p, F, Data, TT=0, topdims=topPCs )
        nna.Project_Simulated_Data( Data, ld['oscbasis'], ('data_emp','DATA_emp','basis_emp') )
        l          = ld['l']  # eigenvalues of linear dynamics (during delay period)

        # Calculate Angle between Oscillatory and XCM / Choice axis #
        ang       = Calculate_Angles( p, F, model, Data, ld,  TT=0, topdims=topPCs )

        # Calculate Projection (A-G) onto Choice Axis #
        # Calculate Choice availability #
        # Calculate End recoding index
        ch, ec, eri   = Calculate_Choice_Measures( p, F, Data, ld['pca'],  TT=0 )

        # Calculate G (geometric) indices #
        G_dict        = Calculate_G(    p, Data_geometry, topPCs_geometry )
        gi            = Calculate_GI(   p, G_dict  )[:,:,:,np.newaxis]
        g_win         = G_dict['Pair_win'][:,:,:,:,np.newaxis]    # [DT,mm,ww,pp,modelnum]   DT: datatype, mm: measure, ww: window, pp: pair
        g_dyn         = G_dict['Pair_dyn'][:,:,:,np.newaxis]    # [DT,mm,pp,modelnum]
        g_trace       = G_dict['Pair_trace'][:,:,:,:,np.newaxis]    # [DT,mm,t,pp,modelnum]   DT: datatype, mm: measure, ww: window, pp: pair

        # Calculate Correlation (Kendall) during Delay # 
        C_dict        = Calculate_C(    p, Data_geometry, topPCs_geometry, pca )
        ci            = Calculate_CI(   p, C_dict  )[:,:,:,np.newaxis]
        c_win         = C_dict['Cond_win'][:,:,:,:,np.newaxis]    # [DT,mm,ww,pp,modelnum]   DT: datatype, mm: measure, ww: window, pp: pair
        c_dyn         = C_dict['Cond_dyn'][:,:,:,np.newaxis]    # [DT,mm,pp,modelnum]
        c_trace       = C_dict['Cond_trace'][:,:,:,:,np.newaxis]    # [DT,mm,t,pp,modelnum]   DT: datatype, mm: measure, ww: window, pp: pair

        # Calculate D (dimensionality) indices #
        neff_win, neff_trace      = Calculate_Dim_Delay(p, F, model,  TT = 0)

        # concatenate across models #
        if len(GI) == 0:
            G_win       = g_win        # [v,2,model]
            G_dyn       = g_dyn        # [v,2,model]
            G_trace     = g_trace        # [v,2,model]
            GI          = gi            # [DT,vv,EA,model]
            C_win       = c_win        # [v,2,model]
            C_dyn       = c_dyn        # [v,2,model]
            C_trace     = c_trace        # [v,2,model]
            CI          = ci            # [DT,vv,EA,model]
            Neff_win    = neff_win[:,np.newaxis]
            Neff_trace  = neff_trace[:,np.newaxis]
            L           = l[:,np.newaxis]
            Ec          = ec[:,np.newaxis]
            Eri         = eri[:,np.newaxis]
        else:
            GI          = np.concatenate( (GI,gi), axis = 3 )        
            G_win       = np.concatenate( (G_win,g_win), axis = 4 )        
            G_dyn       = np.concatenate( (G_dyn,g_dyn), axis = 3 )        
            G_trace     = np.concatenate( (G_trace,g_trace), axis = 4 )        

            CI          = np.concatenate( (CI,ci), axis = 3 )        
            C_win       = np.concatenate( (C_win,c_win), axis = 4 )        
            C_dyn       = np.concatenate( (C_dyn,c_dyn), axis = 3 )        
            C_trace     = np.concatenate( (C_trace,c_trace), axis = 4 )        

            Neff_win    = np.concatenate( (Neff_win, neff_win[:,np.newaxis]), axis = -1)
            Neff_trace  = np.concatenate( (Neff_trace, neff_trace[:,np.newaxis]), axis = -1)
            L           = np.concatenate( (L, l[:,np.newaxis]), axis = -1)       
            Ec          = np.concatenate( (Ec, ec[:,np.newaxis]), axis = -1)       
            Eri         = np.concatenate( (Eri, eri[:,np.newaxis]), axis = -1)      
        

        LD.append(ld)
        ANG.append(ang)
        CH.append(ch)
        PC.append(Data['pca'])
        DATA_G.append(Data_G)

    # report counts
    print('Model variant counts: %s' % (variantcount))

    # output
    Neurals                      = {}
    Neurals['TIDs']              = TIDs
    Neurals['Model_variant']     = np.array( Model_variant )
    Neurals['Pairs']             = G_dict['Pairs']
    Neurals['Pairs_symdists']             = G_dict['Pairs_symdists']
    if p.T > 1:
        Neurals['G_win']         = G_win     
        Neurals['G_dyn']         = G_dyn      
        Neurals['G_trace']       = G_trace    
        Neurals['GI']            = GI          

        Neurals['C_win']         = C_win     
        Neurals['C_dyn']         = C_dyn      
        Neurals['C_trace']       = C_trace    
        Neurals['CI']            = CI          

        Neurals['Neff_win']      = Neff_win         
        Neurals['Neff_trace']    = Neff_trace

        Neurals['Ec']            = Ec.flatten()     # [model]
        Neurals['Eri']           = Eri              # [<exact,empirical>,model]

        Neurals['l']             = L         
        Neurals['LD']            = LD
        Neurals['ANG']            = ANG
        Neurals['CH']            = CH
        Neurals['PC']            = PC

        Neurals['Data_G']          = DATA_G

        Neurals['topPCs']           = topPCs
        Neurals['topPCs_geometry']  = topPCs_geometry

    return Neurals

   



def Plot_GI( p, Neurals, Behaviors, normalize = False, random_data=False ):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.size': 9})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import nn_plot_functions    as npf

    inds_n            = np.isin(Neurals['TIDs'],Behaviors['TIDs'])
    TIDs_plot         = np.array(Neurals['TIDs'])[inds_n]
    inds_b            = np.isin(Behaviors['TIDs'],TIDs_plot)
    assert np.all( np.array(Neurals['TIDs'])[inds_n] == np.array(Behaviors['TIDs'])[inds_b] )

    if random_data:
        DT_toplot = [0,1]
    else:
        DT_toplot = [0]

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(50,600,2000,500) 
    fig.suptitle( '\n\n', fontsize=20, weight='bold')

    
    for EA in [0,1]:        # 0: exact, 1: averaged (time)
        if EA == 0:
            str1 = 'Exact time'
        elif EA == 1:
            str1 = 'Averaged'
        for vv in range(Neurals['GI'].shape[1]):    # iterate over each geometrix index 
            xlims = []
            if np.isin(vv,[0,1]):
                if vv == 0: 
                    str = 'Col, early'
                elif vv == 1:
                    str = 'Col, late'
                numBins = 20
                bins = np.linspace(0,1,numBins)
                xticks = np.arange(0,1.25,0.25)
                xticklabels = ['0','','','','1']
            elif np.isin(vv,[3,4]):
                if vv == 3:
                    str = 'OCI, early'
                elif vv == 4:
                    str = 'OCI, late'
                numBins = 20
                bins = np.linspace(-0.5,1,numBins)
                xticks = np.arange(-0.5,1.25,0.25)
                xticklabels = ['-0.5','','0','','','','1']
            elif np.isin(vv,[2,5]):
                if vv == 2:
                    str = 'Col, change'
                elif vv == 5:
                    str = 'OCI, change'
                numBins = 22
                if np.any( Neurals['GI'][0,vv,:] > 0.6):
                    bins = np.linspace(-1,1,numBins)
                    xticks = np.arange(-1,1.25,0.25)  
                    xticklabels = ['-1','','','','0','','','','1']
                else:        
                    # bins = np.linspace(-0.75,0.75,numBins)
                    # xticks = np.arange(-0.75,1,0.25)  
                    # xticklabels = ['-0.75','','','0','','','0.75']        
                    bins = np.linspace(-0.6,0.6,numBins)
                    xticks = np.arange(-0.6,0.6+0.3,0.3)  
                    xticklabels = ['-0.6','','0','','0.6']        
            elif np.isin(vv,[6,7]):
                if vv == 6:
                    str = 'Mean angle, early'
                elif vv == 7:
                    str = 'Mean angle, late'
                numBins = 20
                bins = np.linspace(-0.3,0.3,numBins)
                xticks = np.arange(-.3,.4,0.1)
                xticklabels = ['-0.3','','','0','','','0.3']
            elif vv == 8:
                str = 'Mean angle change'
                if p.input_train == 1:
                    numBins = 24
                    bins = np.linspace(-0.15,0.15,numBins)
                    xticks = np.arange(-.15,.2,0.05)
                    xticklabels = ['-0.15','','','0','','','0.15']
                else:
                    numBins = 21
                    bins = np.linspace(-0.2,0.2,numBins)
                    xticks = np.arange(-.2,.3,0.1)
                    xticklabels = []                
                # else:
                #     bins = np.linspace(-0.3,0.3,numBins)
                #     xticks = np.arange(-.3,.4,0.1)
                #     xticklabels = ['-0.3','','','0','','','0.3']
            elif vv == 9:
                numBins = 20
                str = 'Mean distance change'
                bins = np.linspace(-5,15,numBins)
                xticks = np.arange(-5,20,5)
                # xticklabels = ['-2','0','2','4','6','8','10','12']
                xticklabels = ['-5','0','5','10','15']

            ax = fig.add_subplot(2,10,EA*10+vv+1)  
            Ymax = 0
            for DT in DT_toplot:  #[0,1]:   # 0: data, 1: random vectors
                if DT == 0:
                    variants_toplot = np.arange(4,-1,-1)
                elif DT == 1:
                    variants_toplot = [0]
                vals = Neurals['GI'][DT,vv,EA,inds_n]
                if np.all(~np.isnan(vals)):  
                    for v in variants_toplot:
                        inds = Neurals['Model_variant'][inds_n] == v
                        if 0:
                            histout = plt.hist( vals[inds], bins, fc=variant_color(v,0.4), histtype='stepfilled', ec=variant_color(v,0.8), linewidth=3)
                            Ymax = np.max(histout[0])
                        elif 1:
                            if DT == 0:
                                ymax = npf.area_histogram(ax, vals[inds], bins, 3, variant_color(v), 0.9, 0.2, 100, 200, normalize)
                            else:
                                ymax = npf.area_histogram(ax, vals[inds], bins, 3, 'k', .25, 0, 100, 200, normalize)
                            if ymax > Ymax:
                                Ymax = ymax
                        else:
                            ax.hist( vals[inds], bins, density=True, color=variant_color(v), alpha=0.8 )
                if normalize:
                    if vv < 3:          # Collinearity
                        Ymax = 0.75
                        ytick = 0.25
                    elif vv < 6:        # OCI
                        Ymax = 1
                        ytick = 0.25
                    elif vv < 9:        # Mean angle
                        Ymax = 1
                        Ymax = .75
                        ytick = 0.25
                        # Ymax = 0.5
                        # ytick = 0.1
                    elif vv == 9:       # Mean distance
                        Ymax = 1
                        ytick = 0.25
                    ax.set_yticks(np.arange(0,Ymax+ytick,ytick))

                    # elif vv < 9:
                    #     Ymax = 1
                    #     ax.set_yticks(np.arange(0,Ymax+.25,.25))
                    # elif Ymax > 0.5:
                    #     Ymax = 0.5
                    #     ax.set_yticks(np.arange(0,Ymax+.1,.1))
                    ax.set_ylim([0,Ymax])
                    # if vv > 0:
                        # ax.set_yticklabels([])
                else:
                    ax.set_yticks(np.arange(0,Ymax+150,50))
                    ax.set_ylim([0,Ymax+50.000001])
                if vv > 1:
                    ax.plot([0,0],[0,Ymax],'-',color=(0.7,0.7,0.7),linewidth=1.5)
                if vv == 0:
                    ax.set_ylabel(str1 + '\n\n',fontsize=16)
                # ax.set_xlabel('Index')
                ax.set_xticks(xticks)
                ax.set_title(str,fontsize=16)
                if len(xticklabels) != 0:
                    ax.set_xticklabels(xticklabels,fontsize=16)
                ax.set_xlim([bins[0],bins[-1]])
    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    



def Plot_CI( Neurals, Behaviors , jobname):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import nn_plot_functions    as npf

    inds_n            = np.isin(Neurals['TIDs'],Behaviors['TIDs'])
    TIDs_plot         = np.array(Neurals['TIDs'])[inds_n]
    inds_b            = np.isin(Behaviors['TIDs'],TIDs_plot)
    assert np.all( np.array(Neurals['TIDs'])[inds_n] == np.array(Behaviors['TIDs'])[inds_b] )

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(50,600,1000,500) 
    fig.suptitle( 'Correlation (%s) \n\n' % (jobname), fontsize=16, weight='bold')
    
    numBins = 20
    for EA in [0,1]:        # 0: exact, 1: averaged (time)
        if EA == 0:
            str1 = 'Exact time'
        elif EA == 1:
            str1 = 'Averaged'
        strs = ['Sensory','SymDists (4,5,6), early','SymDists (4,5,6), late','','','','']
        for vv in range(3):    # iterate over each geometrix index 
            if vv == 0 :
                bins = np.linspace(-1,1,numBins)
                xticks = np.arange(-1,1.25,0.25)  
                xticklabels = ['-1','','','','0','','','','1']
            else:    
                bins = np.linspace(-1,1,numBins)
                xticks = np.arange(-1,1.25,0.25)  
                xticklabels = ['-1','','','','0','','','','1']

            ax = fig.add_subplot(2,3,EA*3+vv+1)  
            Ymax = 0
            for DT in [0]:  #[0,1]:   # 1: data, 2: random vectors
                vals = Neurals['CI'][DT,vv,EA,inds_n]
                if np.all(~np.isnan(vals)):  
                    for v in reversed(range(5)):
                        inds = Neurals['Model_variant'][inds_n] == v
                        if 0:
                            histout = plt.hist( vals[inds], bins, fc=variant_color(v,0.4), histtype='stepfilled', ec=variant_color(v,0.8), linewidth=3)
                            Ymax = np.max(histout[0])
                        elif 1:
                            ymax = npf.area_histogram(ax, vals[inds], bins, 3, variant_color(v), 0.9, 0.2, 100, 200)
                            if ymax > Ymax:
                                Ymax = ymax
                        else:
                            ax.hist( vals[inds], bins, density=True, color=variant_color(v), alpha=0.8 )
                    ax.plot([0,0],[0,Ymax+50],'-',color=(0.7,0.7,0.7),linewidth=3)
                ax.set_yticks(np.arange(0,Ymax+20,50))
                ax.set_ylim([0,Ymax+20+5.000001])
                if vv == 0:
                    ax.set_ylabel(str1 + '\n\n#',fontsize=16)
                ax.set_xlabel('Index')
                ax.set_xticks(xticks)
                ax.set_title(strs[vv],fontsize=16)
                if len(xticklabels) != 0:
                    ax.set_xticklabels(xticklabels,fontsize=18)
                ax.set_xlim([bins[0],bins[-1]])
    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()        


def Plot_LD_R2(p, Neurals ):

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(300,100,280,450) 
    plt.suptitle('LSQ linear dynamics\nRsquareds', fontsize=24, fontweight='bold')
    ax = plt.subplot(1,1,1)

    width = 0.3

    # collect values #
    vals_variants = []

    for vv in range(5):

        inds = np.where( Neurals['Model_variant'] == vv )[0]
        vals = []
        for mm in list(inds):
            vals.append(Neurals['LD'][mm]['R2'])
        vals = np.array(vals)

        ### violin ###
        if 1:
            parts   = ax.violinplot(vals,[1+vv],\
                                    bw_method = 0.3, widths = 0.6, showmedians=False,  showmeans=False, showextrema=False)

            for i,pc in enumerate(parts['bodies']):
                pc.set_facecolor(variant_color(vv))
                # pc.set_edgecolor('k')
                # pc.set_linewidth(2)
                pc.set_alpha(.45)
            ### box ###
            clr = variant_color(vv)
            q1, median, q3 = np.percentile( vals, [25, 50, 75] )
            whiskers_max = q3 + (q3 - q1) * 1.5
            whiskers_min = q1 - (q3 - q1) * 1.5
            # ax.scatter( [1+vv], median,      marker='o',     color='w', s=15, zorder=21)
            # ax.vlines(  [1+vv], q1,    q3,      color=clr, linestyle='-', lw=6,zorder=20)
            ax.scatter( [1+vv], median,      marker='o',     color='w', s=50, edgecolor='k', linewidth=1.5, zorder=21)
            ax.plot(  [1+vv,1+vv], [q1,q3],      color=clr, linestyle='-', lw=4, zorder=20, solid_capstyle='round')
       
        # scatter #
        else:
            xjit = 1 + vv + width * np.random.rand(len(vals)) - width/2
            ax.scatter( xjit, np.asarray(vals), color=variant_color(vv,1), s=55, alpha=0.6)           
        
        ##########

        meanvals = np.mean( vals )
        stdvals = np.std( vals )
        print('R2, variant %g: mean ± s.d.: %0.2f ± %0.2f' % (vv,meanvals,stdvals))

        vals_variants.append(np.asarray(vals))

    ax.set_xlim([0,6])
    ax.set_yticks(np.arange(0,1.1,0.1))
    ax.set_ylim([0,1.0001])
    ax.set_xticks(np.arange(1,6,1))
    ax.tick_params('both', length=6,  which='major')
    ax.tick_params('both', length=6, which='minor')
    fig.tight_layout()    
    plt.show()

    if 0: #p.input_train == 1:
        vals_simple = np.array([vals_variants[0],vals_variants[1]])
        meanvals    = np.mean( vals_simple )
        stdvals     = np.std(  vals_simple )
        print('R2 of simple-regimers: mean ± s.d.: %0.2f ± %0.2f' % (meanvals,stdvals))



def Plot_Angles( p, Neurals ):

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    width = 0.3

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(150,600+100,1000,350) 

    # collect values #
    for Ang in range(4):

        ax = plt.subplot(1,4,Ang+1)

        if Ang == 0:
            str = 'xcm_ang'
        elif Ang == 1:
            str = 'choice_ang'
        elif Ang == 2:
            str = 'xcm_choice_ang'
        elif Ang == 3:
            str = 'choice_readout_ang'

        plt.suptitle('Oscillation (OLS)\n', fontsize=24, fontweight='bold')

        for vv in range(6):

            if vv <= 4:
                vvv = vv
                ii = np.array([0])
                clr = variant_color(vv,1)
                alpha = 0.4
                inds = np.where( Neurals['Model_variant'] == vvv )[0]
                vals = np.array([])
                for mm in list(inds):
                    vals = np.concatenate((vals, Neurals['ANG'][mm][str][ii]),axis=0)
            else:
                vvv = 0 
                ii = np.arange(1,10)
                clr = (.7,.7,.7)
                alpha = 0.15
                vals =  Neurals['ANG'][0][str][1:300]

            vals = np.abs( np.array(vals) )

            plt.title(str)

            # # scatter jitter #
            # xjit = 1 + vv + width * np.random.rand(len(vals)) - width/2
            # ax.scatter( xjit, np.asarray(vals), color=clr, s=55, alpha=alpha)    

            ## violin ###
            parts   = ax.violinplot(vals,[1+vv],\
                                    bw_method = 0.25, widths = 0.6, showmedians=False,  showmeans=False, showextrema=False)
            for i,pc in enumerate(parts['bodies']):
                pc.set_facecolor(clr)
                # pc.set_edgecolor(clr)
                # pc.set_linewidth(2)
                pc.set_alpha(.45)

            ## box ###
            q1, median, q3 = np.percentile( vals, [25, 50, 75] )
            # whiskers_max = q3 + (q3 - q1) * 1.5
            # whiskers_min = q1 - (q3 - q1) * 1.5
            ax.scatter( [1+vv], median,      marker='o',     color='w', s=50, edgecolor='k', linewidth=1.5, zorder=21)
            ax.plot(  [1+vv,1+vv], [q1,q3],      color=clr, linestyle='-', lw=4, zorder=20, solid_capstyle='round')
            # ax.vlines(  [1+vv], whiskers_min, whiskers_max,   color=clr, linestyle='-', lw=1.5,zorder=15)
        

            # ax.set_ylim([-np.pi,np.pi])
            ax.set_ylim([0,1])
            ax.set_xlim([0.25,6.75])

            # ax.set_yticks(np.arange(0,1.1,0.1))
            ax.set_xticks(np.arange(1,7,1))
            ax.set_xticklabels(['1','2','3','4','5','Rand'])
            ax.set_yticks(np.arange(0,1.25,0.25))
            if Ang == 0:
                ax.set_ylabel('Cos angle')
            else:
                ax.set_yticklabels([])
            ax.tick_params('both', length=6,  which='major')
            ax.tick_params('both', length=6, which='minor')
            fig.tight_layout()    
            plt.show()

            fig.subplots_adjust(hspace=.5)

    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    

def Plot_PCA_varexp(p, Neurals, topPCs = 3 ):

    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 18})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(300,100,400,500) 
    plt.suptitle('PCA variance explained (top %g PCs)' % (topPCs), fontsize=24, fontweight='bold')
    ax = plt.subplot(1,1,1)

    width = 0.3

    # collect values #
    for vv in range(5):

        inds = np.where( Neurals['Model_variant'] == vv )[0]
        vals = []
        for mm in list(inds):
            pca     = Neurals['PC'][mm]  # pca scikit object
            val      =  np.sum(pca.explained_variance_ratio_[:topPCs])            
            vals.append(val)
        xjit = 1 + vv + width * np.random.rand(len(vals)) - width/2

        ax.scatter( xjit, np.asarray(vals), color=variant_color(vv,1), s=55, alpha=0.6)    
        # ax.violinplot(vals,xvec,bw_method = 0.25, widths = 0.6, showmedians=False,  showmeans=False, showextrema=True)

        ax.set_yticks(np.arange(0,1.1,0.1))
        ax.set_xticks(np.arange(1,6,1))
        ax.set_ylim([0.5,1])
        ax.set_xlim([0,6])
        ax.tick_params('both', length=6,  which='major')
        ax.tick_params('both', length=6, which='minor')
        fig.tight_layout()    
        plt.show()

        meanvals = 100*np.mean(vals)
        sdvals = 100*np.std(vals)
        print('Variant %g, varexp, mean ± s.d.: %0.1f ± %0.1f' % (vv,meanvals,sdvals) )

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(50,50,300,350) 
    plt.suptitle('PCA variance explained', fontsize=24, fontweight='bold')
    ax = plt.subplot(1,1,1)

    # collect values #
    for vv in reversed(range(5)):

        inds = np.where( Neurals['Model_variant'] == vv )[0]
        vals = np.zeros( (inds.size,6) )
        i    = 0
        for mm in list(inds):
            pca        = Neurals['PC'][mm]  # pca scikit object
            vals[i,:] = np.cumsum( pca.explained_variance_ratio_[:6] )       
            i = i+1    
        meanvals = np.mean(vals[:,:6],axis=0)
        sdvals   = np.std(vals[:,:6],axis=0)
        # ax.plot( np.arange(1,6), np.mean(vals[:,:5],axis=0), color=variant_color(vv,1))    
        ax.fill_between(np.arange(6), meanvals-sdvals,  meanvals+sdvals,  color=variant_color(vv,1),  edgecolor=None,      alpha=.65)

        ax.set_yticks(np.arange(0,1.1,0.1))
        ax.set_xticks(np.arange(6))
        ax.set_ylim([0,1])
        ax.set_xlim([0,5])
        ax.tick_params('both', length=6,  which='major')
        ax.tick_params('both', length=6, which='minor')
        fig.tight_layout()    
        plt.show()


def Plot_G(p, Neurals):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 25})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"

    MAXVAL = 1

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(770,100,1500,1000) 
    fig.suptitle( 'RDMs, averages across model instances', fontsize=24, weight='bold')

    DT = 0
    ww = 0

    for mm in range(6):

        if mm == 0:
            minval,maxval = (-1,1)
            cmap = 'bwr'
            str = 'Cosine similarity\n'
        elif mm == 1:
            minval,maxval = (0,10)
            cmap = 'binary'
            str = 'Euclidean distance\n'
        elif mm == 2:
            minval,maxval = (0,20)
            cmap = 'binary'
            str = 'Angular speed\n'
        elif mm == 3:
            minval,maxval = (0,3)
            cmap = 'binary'
            str = 'Speed\n'
        elif mm == 4:
            minval,maxval = (-0.5,0.5)
            cmap = 'binary'
            str = 'Ang mean\n'
        elif mm == 5:
            minval,maxval = (0,15)
            cmap = 'binary'
            str = 'Dist mean\n'

        for v in range(5):

            ax      = fig.add_subplot(6,5,mm*5+v+1)   # plot by regime variant
            inds    = Neurals['Model_variant'] == v

            if mm < 2:
                matval = Convert_Pairs_to_Matrix(p, Neurals['G_win'][DT,mm,ww,:,:], Neurals['Pairs'])
                avgmat  = np.mean(matval[:,:,inds], axis=2)
                plt.imshow( avgmat , vmin=minval, vmax=maxval, cmap=cmap)
                ax.set_xticks(ticks=range(p.n_items))
                ax.set_yticks(ticks=range(p.n_items))
                ax.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
                ax.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
            else:
                values = Neurals['G_win'][DT,mm,ww,:7,:]
                avgvals  = np.mean(values[:,inds], axis=1)[:,np.newaxis]
                plt.plot( avgvals , color='k', linewidth=2)
                ax.set_ylim([minval,maxval])
            if mm == 0:
                ax.set_title('%s\n' % (variant_string(v)),fontsize=20, color=variant_color(v))
            if v == 0:
                ax.set_ylabel(str,fontsize=20)

    fig.subplots_adjust(hspace=.5)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    



def Plot_G_diff(p, Neurals):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 25})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    MAXVAL = 1

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(770,100,1500,600) 
    fig.suptitle( 'RDMs, averages across model instances', fontsize=24, weight='bold')

    DT = 0
    ww = [0,3]  # initial, later

    for mm in range(2):

        if mm == 0:
            minval,maxval = (-1,1)
            cmap = 'bwr'
            str = 'Cosine similarity\n'
        elif mm == 1:
            minval,maxval = (-10,10)
            cmap = 'bwr'
            str = 'Euclidean distance\n'

        for v in range(5):

            ax      = fig.add_subplot(2,5,mm*5+v+1)   # plot by regime variant
            inds    = Neurals['Model_variant'] == v

            if mm < 2:
                matval_i = Convert_Pairs_to_Matrix(p, Neurals['G_win'][DT,mm,ww[0],:,:], Neurals['Pairs'])
                matval_f = Convert_Pairs_to_Matrix(p, Neurals['G_win'][DT,mm,ww[1],:,:], Neurals['Pairs'])
                diffmat  = matval_f - matval_i
                avgmat  = np.mean(diffmat[:,:,inds], axis=2)
                plt.imshow( avgmat , vmin=minval, vmax=maxval, cmap=cmap)
                ax.set_xticks(ticks=range(p.n_items))
                ax.set_yticks(ticks=range(p.n_items))
                ax.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
                ax.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
            # else:
            #     values = Neurals['G_win'][DT,mm,ww,:7,:]
            #     avgvals  = np.mean(values[:,inds], axis=1)[:,np.newaxis]
            #     plt.plot( avgvals , color='k', linewidth=2)
            #     ax.set_ylim([minval,maxval])
            if mm == 0:
                ax.set_title('%s\n' % (variant_string(v)),fontsize=20, color=variant_color(v))
            if v == 0:
                ax.set_ylabel(str,fontsize=20)

    fig.subplots_adjust(hspace=.5)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    

def Plot_G_single_model(p, Neurals):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 25})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    I           = np.argsort(Neurals['GI'][0,2,:])      # lowest cosine mean to highest
    if 0:
        ind     = I[-35]    # most positive
    elif 0:
        ind     = I[2]     # most negative
    else:
        ind     = 2
    TID        = np.array([Neurals['TIDs']]).flatten()[ind]  
    cosinedelta = Neurals['GI'][0,2,0,ind]    # [DT,vv,EA,inds_n]

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(770,100,1500,600) 
    fig.suptitle( 'RDMs\nTID: %g, Cosine delta: %0.3f' % (TID,cosinedelta), fontsize=24, weight='bold')

    DT          = 0
    windows     = [0,1,2,3]  # 1st quarter
    measures    = [0]

    MAXVAL = 1

    for m in range(len(measures)):

        mm = measures[m]

        if mm == 0:
            minval,maxval = (-1,1)
            cmap = 'bwr'
            str = 'Cosine similarity\n'
        elif mm == 1:
            minval,maxval = (0,10)
            cmap = 'binary'
            str = 'Euclidean distance\n'
        elif mm == 2:
            minval,maxval = (-1,1)
            cmap = 'bwr'
            str = 'Pearson r\n'

        for w in range(len(windows)):
            ax      = fig.add_subplot(len(measures),len(windows)+1,m*len(windows)+w+1)   # plot by regime variant
            ww      = windows[w]
            matvals = Convert_Pairs_to_Matrix(p, Neurals['G_win'][DT,mm,ww,:,:], Neurals['Pairs'])
            plt.imshow( matvals[:,:,ind] , vmin=minval, vmax=maxval, cmap=cmap)
            ax.set_xticks(ticks=range(p.n_items))
            ax.set_yticks(ticks=range(p.n_items))
            ax.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
            ax.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
            if mm == 0:
                ax.set_title('%s\n' % (window_string(ww)),fontsize=20, color='k')
            if w == 0:
                ax.set_ylabel(str,fontsize=20)

        ax      = fig.add_subplot(len(measures),len(windows)+1,5)   # plot by regime variant
        matval0 = Convert_Pairs_to_Matrix(p, Neurals['G_win'][DT,mm,0,:,:], Neurals['Pairs'])
        matval3 = Convert_Pairs_to_Matrix(p, Neurals['G_win'][DT,mm,3,:,:], Neurals['Pairs'])
        diffmat = matval3 - matval0
        plt.imshow( diffmat[:,:,ind] , vmin=minval, vmax=maxval, cmap=cmap)
        ax.set_xticks(ticks=range(p.n_items))
        ax.set_yticks(ticks=range(p.n_items))
        ax.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
        ax.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
        if mm == 0:
            ax.set_title('diff (4th-1st)' % (window_string(ww)),fontsize=20, color='k')
        if w == 0:
            ax.set_ylabel(str,fontsize=20)
                

    fig.subplots_adjust(hspace=.4)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    



def Plot_Neural_Traces_Single_RNN(p, Neurals, TID):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 20})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    T, t1, t2, _, _, _     = tk.parms_test(p,None,0)    

    # I           = np.argsort(Neurals['GI'][0,2,:])      # lowest cosine mean to highest
    # if 0:
    #     ind     = I[-2]    # most positive
    # elif 0:
    #     ind     = I[0]     # most negative
    # elif 1:
    #     ind     = 1
    # else:
    #     # TID     = 149  # f-RNN simple example
    #     # TID     = 458  # f-RNN complex example
    #     # TID     = 440  # r-RNN complex example
    #     TID = 0
    #     ind     = int(np.where(np.asarray(Neurals['TIDs'])==TID)[0])
    # # TID        = np.array([Neurals['TIDs']]).flatten()[ind]  
    # # cosinedelta = Neurals['GI'][0,2,ind]

    ind             = int(np.where(np.asarray(Neurals['TIDs'])==TID)[0])
    Pairs          = Neurals['Pairs']
    symdists       = Neurals['Pairs_symdists']

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(100,600,1000,400) 
    fig.suptitle( 'TID: %g' % (TID), fontsize=24, weight='bold')

    DT          = 0

    for m in range(7):

        if m >= 3:
            skip = 1
        else:
            skip = 0

        ax      = fig.add_subplot(2, 4, m+1)   # plot by regime variant

        if m == 0:              # mean cos angle
            str = 'Cosine (XCM)'
            mm = 0  # 0: cosine similarity, 1: distance
            traces     = Neurals['G_trace'][DT,mm,:,:,ind]  # [DT,mm,T,B,model]
            colors     = symdists[:,np.newaxis]/p.n_items * np.ones((traces.shape[1],3))
            meantrace     = np.mean(traces,axis=1)
            ylims           = (-1,1)
            yticks          = np.arange(-1,1.25,.5)
        elif m == 1:            # mean euc dist
            str = 'Euclidean distance'
            mm = 1  # 0: cosine similarity, 1: distance
            traces = Neurals['G_trace'][DT,mm,:,:,ind]
            meantrace     = np.mean(traces,axis=1)
            ymax = 5*np.ceil( np.nanmax(traces.flatten())/5 )
            ylims = (0,ymax)
            yticks = np.arange(0, ymax, 5)
        elif m == 2:            # OCI (must re-create correct sign (for ordering))
            str = 'OCI'
            ylims = (-1,1)
            yticks = np.arange(-1,1.5,.5)
            mm = 0  # 0: cosine similarity, 1: distance
            trace_all = Neurals['G_trace'][DT,mm,:,:,ind]
            Pairs       = list(combinations(range(p.n_items), 2))
            # conds       = np.array([[0,6],[0,5],[0,4],[1,6],[1,5],[1,4],[2,6],[2,5],[2,4],[0,1],[0,2],[1,2],[6,5],[6,4],[4,5]])  # all opposites
            conds       = [(0,6),(0,5),(0,4),(1,6),(1,5),(1,4),(2,6),(2,5),(2,4),(0,1),(0,2),(1,2),(6,5),(6,4),(4,5)]  # all opposites
            traces      = np.full((0,t2),np.nan)
            for pp in range(len(Pairs)):
                if Pairs[pp] in conds or reversed( Pairs[pp] ) in conds:
                    sign1    = np.sign( Pairs[pp][0] - 3 )
                    sign2    = np.sign( Pairs[pp][1] - 3 )
                    if sign1 == sign2:
                        # print('pos')
                        SIGN = +1         
                    else:
                        # print('neg')
                        SIGN = -1   
                    onetrace  = trace_all[:,pp] * SIGN
                    traces = np.concatenate((traces,onetrace[np.newaxis,:]),axis=0)
            traces    = traces.T
            meantrace     = np.mean( traces, axis=1 )
        elif m == 3:
            str = 'Collinearity'
            ylims = (0,1)
            yticks = np.arange(-1,1.5,.5)
            mm = 0  # 0: cosine similarity, 1: distance
            trace_all = Neurals['G_trace'][DT,mm,:,:,ind]
            Pairs       = list(combinations(range(p.n_items), 2))
            # conds       = np.array([[0,6],[0,5],[0,4],[1,6],[1,5],[1,4],[2,6],[2,5],[2,4],[0,1],[0,2],[1,2],[6,5],[6,4],[4,5]])  # all opposites
            # conds       = [(0,6),(0,5),(0,4),(1,6),(1,5),(1,4),(2,6),(2,5),(2,4),(0,1),(0,2),(1,2),(6,5),(6,4),(4,5)]  # all opposites
            conds       = list(combinations([0,1,2,4,5,6], 2))     
            traces      = np.full((0,t2),np.nan)
            for pp in range(len(Pairs)):
                if Pairs[pp] in conds or reversed( Pairs[pp] ) in conds:
                    onetrace  = trace_all[:,pp]
                    traces = np.concatenate((traces,onetrace[np.newaxis,:]),axis=0)
            traces    = np.abs( traces.T )
            meantrace     = np.mean( traces, axis=1 )                   
        elif m == 4:     # sensory
            str = 'Correlation, sensory\n(ref: early delay)'
            mm          = 0  # 0: sensory corr, 1: memory corr, 2: symdist corr
            traces   = Neurals['C_trace'][DT,mm,:,:,ind]
            meantrace     = np.mean( traces, axis=1 )
            yticks = np.arange(-1,1.5,.5)
        elif m == 5:     # memory
            str = 'Correlation, memory\n(ref: late delay)'
            mm          = 1  # 0: sensory corr, 1: memory corr, 2: symdist corr
            traces   = Neurals['C_trace'][DT,mm,:,:,ind]        
            meantrace     = np.mean( traces, axis=1 )
            yticks = np.arange(-1,1.5,.5)
        elif m == 6:     # rank (Barak 2010)
            str = 'Correlation (origin)'
            mm              = 2  # 0: sensory corr, 1: memory corr, 2: symdist corr
            traces          = Neurals['C_trace'][DT,mm,:,:,ind]
            meantrace       = np.nanmean( traces, axis=1 )
            yticks = np.arange(-1,1.5,.5)
        elif m == 7:     # overall across-pairs
            str = 'Correlation\nacross-pairs'
            mm              = 3  # 0: sensory corr, 1: memory corr, 2: symdist corr, 3: across-pair corr
            traces          = Neurals['C_trace'][DT,mm,:,:,ind]
            meantrace       = np.nanmean( traces, axis=1 )
            yticks = np.arange(-1,1.5,.5)

        d1 = t2-t1

        ax.plot( [0,1], [0, 0], color='k', alpha=0.5, linewidth=.5 )
        if m == 0 or m == 1:
            if 0:
                for cc in range(traces[t1:t2,:].shape[1]):   
                    ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,cc],color=colors[cc,:], alpha=1, linewidth=2.5)     # individual pairs
            elif 1:
                for sd in range(M-1):  
                    # inds_sd    = np.where(  (symdists-1)==sd  )[0]
                    # inds_inner = []      
                    # for pp in range(traces.shape[1]):
                    #     if not np.any(np.isin(Pairs[pp],[0,M-1])):
                    #         inds_inner.append(pp)
                    # inds = np.intersect1d(inds_sd,inds_inner)
                    clr = np.array([sd/(M-1),sd/(M-1),sd/(M-1)])
                    # meantrace_sd = np.mean( traces[t1:t2, inds], axis=1 ) 
                    meantrace_sd = np.mean( traces[t1:t2, (symdists-1)==sd], axis=1 ) 
                    ax.plot(np.arange(d1)/(d1-1), meantrace_sd, color=clr, alpha=1, linewidth=2.5)     # individual pairs
            # ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,:],color='k', alpha=0.15, linewidth=2.5)     # individual pairs
            # ax.plot(np.arange(d1)/(d1-1), meantrace[t1:t2],'--', color='k', linewidth=2.5)           # average  (across pairs)
        elif m == 2:
            ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,:],color='k', alpha=0.15, linewidth=2.5)     # individual pairs
            ax.plot(np.arange(d1)/(d1-1), meantrace[t1:t2],'--', color='k', linewidth=2.5)           # average  (across pairs)
        elif m == 3:
            ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,:],color='k', alpha=0.075, linewidth=2.5)     # individual pairs
            ax.plot(np.arange(d1)/(d1-1), meantrace[t1:t2],'-', color='k', alpha=0.75, linewidth=2.5)           # average  (across pairs)
        elif m == 4:
            ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,:],color='k', alpha=0.15, linewidth=2.5)     # individual pairs
            ax.plot(np.arange(d1)/(d1-1), meantrace[t1:t2],color='green', linewidth=4)           # average  (across pairs)        
        elif m == 5:
            ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,:],color='k', alpha=0.15, linewidth=2)     # individual pairs
            ax.plot(np.arange(d1)/(d1-1), meantrace[t1:t2],color='green', alpha=0.25, linewidth=4)           # average  (across pairs)
        elif m == 6:
            for sd in range(M-1):        
                clr = np.array([sd/(M-1),sd/(M-1),sd/(M-1)]) 
                ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,sd], color=clr, alpha=1, linewidth=3)     # individual pairs
            ax.plot(np.arange(d1)/(d1-1), meantrace[t1:t2], '--', color='k', alpha=0.2, linewidth=3)           # average  (across pairs)
            ax.set_ylim( [-1,1] )
        elif m == 7:
            ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,:],color='k', alpha=1, linewidth=4)     # individual pairs
    
        if m != 1:
            ax.set_xticks(np.arange(0,1.25,.25))
            ax.set_xticklabels(['0','','','','1'])
            ax.set_yticklabels(['-1','','0','','1'])
    
        ax.set_xlim([0,1])
        ax.set_ylim(ylims)
        ax.set_xticks(np.arange(0,1.25,.25))
        ax.set_yticks(yticks)
        if m == 3:
            ax.set_ylim([0,1])
            ax.set_yticks(np.arange(0,1.25,.25))
            ax.set_yticklabels(['0','','','','1'])
        ax.set_xlabel('Time', fontsize=18,  fontweight='bold')
        ax.set_title(str,fontsize=18)
        ax.tick_params('both', length=6, width=1, which='major')
        ax.tick_params('both', length=6, width=1, which='minor')
        plt.tight_layout()


    fig.subplots_adjust(hspace=1)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    


def Plot_Readout_Example(p, Neurals, jobname, TID = np.nan):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 20})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    T, t1, t2, _, _, _     = tk.parms_test(p,None,0)    

    if np.isnan(TID):
        I           = np.argsort(Neurals['GI'][0,2,:])      # lowest cosine mean to highest
        if 0:
            ind     = I[-2]    # most positive
        elif 0:
            ind     = I[0]     # most negative
    else:
        if np.sum( np.asarray(Neurals['TIDs'])==TID   ) == 0:
            print('(Plot choice study) TID not found!')
            return
        else:
            ind     = int(np.where( np.asarray(Neurals['TIDs'])==TID )[0])
        # TID        = np.array([Neurals['TIDs']]).flatten()[ind]  
        # cosinedelta = Neurals['GI'][0,2,ind]

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(50,500,1600,700) 
    # fig.suptitle( 'RDMs\nTID: %g, Cosine delta: %0.3f' % (TID,cosinedelta), fontsize=24, weight='bold')

    DT          = 0
    windows     = [0,1,2,3]  # 1st quarter
    measures    = [0,1,0]

    MAXVAL = 1
    XPAD     = 1

    for TYPE in [0,1]:   # 0: readout axis, 1: choice axis

        if TYPE == 0:
            traces = Neurals['CH'][ind]['rop']
            traces = traces - np.mean(traces,axis=1)[:,np.newaxis]
            ystr    = 'Readout unit'
        elif TYPE == 1:        
            traces = Neurals['CH'][ind]['chp']
            ystr    = 'Choice axis'

        maxamp = np.max( np.abs( traces[[t1,t2-1],:].flatten() ) ) 

        if maxamp > 5:
            maxval = 15
            ytick = 5
        elif maxamp > 3:
            maxval = 6
            ytick = 2   
        elif maxamp > 2:             
            maxval = 3
            ytick = 1   
        elif maxamp > 1:
            maxval = 2
            ytick  = 1            
        else:
            maxval = 1
            ytick  = .25            

        ### I.  Plot "Twisting" across delay ##########
        ax      = fig.add_subplot(2, 3, 1+TYPE*3)   # plot by regime variant 
        meantrace         = np.mean( traces, axis=1 )
        # greens = np.array( [[0,135,20],[43,155,59],[85,175,98],[128,195,138],[170,215,177],[213,235,216],[255,255,255] ] ) / 255
        greens = np.array( [[0,135,20],[43,155,59],[85,175,98],[128,195,138],[170,215,177],[213,235,216],[240,240,240] ] ) / 255

        # maxval = np.ceil( np.nanmax(np.abs(traces.ravel())) )
        ylims = (-maxval-0.00001,maxval+.00001)
        str = 'Output study (%s) (TID: %g)\n' % (jobname,TID)

        d1 = t2-t1
        for mm in range(p.n_items):
            # clr = embedcolor(mm,p.n_items,3)
            ax.plot(np.arange(d1)/(d1-1), traces[t1:t2,mm],color=greens[mm], alpha=1, linewidth=3.5)     # individual pairs
        # ax.plot(np.arange(d1)/(d1-1), meantrace[t1:t2],color='k', linewidth=1, alpha=0.2, zorder=-10)           # average  (across pairs)
        ax.plot([0,1],[0,0],'--',linewidth=2,color='k',alpha=0.15,zorder=-5)
        ax.set_xlim([0,1])
        ax.set_ylim(ylims)
        ax.set_xticks(np.arange(0,1.25,.25))
        ax.set_yticks(np.arange(-maxval, maxval+ytick, ytick))
        ax.set_xlabel('Time', fontsize=18,  fontweight='bold')
        ax.set_ylabel(ystr, fontsize=18,  fontweight='bold')
        if TYPE == 0:
            ax.set_title(str,fontsize=18)
        ax.tick_params('both', length=6, width=1, which='major')
        ax.tick_params('both', length=6, width=1, which='minor')
        plt.tight_layout()

        ### II.  Plot Rank at Beginning and End #############
        ax      = fig.add_subplot(2, 6, 3+TYPE*6)   # plot by regime variant 
        d1      = t2-t1
        lw      = 1.5
        for mm in range(p.n_items):
            # clr = embedcolor(mm,p.n_items,3)
            ax.scatter(mm, traces[t1,mm], color=greens[mm], marker='X', alpha=1, linewidth=lw, s=125, edgecolor='k',zorder=10)                        # item 2 (FF)
            ax.scatter(mm, traces[t2-1,mm], color=greens[mm], alpha=1, linewidth=lw, s=75, edgecolor='k',zorder=5)     # item 1 (REC)
        ax.plot(np.arange(p.n_items), traces[t1,:],color='k', linewidth=3, alpha=0.2, zorder=8)           # average  (across pairs)
        ax.plot(np.arange(p.n_items), traces[t2-1,:],color='k', linewidth=3, alpha=0.2, zorder=0)           # average  (across pairs)
        ax.plot([-XPAD,p.n_items-1+XPAD],[0,0],'--',linewidth=2,color='k',alpha=0.15,zorder=-5)
        ax.set_xlim([-XPAD,p.n_items-1+XPAD])
        ax.set_ylim(ylims)
        ax.set_yticks(np.arange(-maxval, maxval+.5, .5))
        plt.yticks(np.arange(-maxval,maxval+ytick,ytick))
        plt.xticks(ticks=range(p.n_items),labels=[])
        ax.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
        ax.set_xlabel('Rank', fontsize=18,  fontweight='bold')
        ax.tick_params('both', length=6, width=1, which='major')
        ax.tick_params('both', length=6, width=1, which='minor')
        plt.tight_layout()
        # ax.set_aspect(.5)  

        ### III.  Plot Matrix of Outputs #############
        ax      = fig.add_subplot(2, 6, 4+TYPE*6)   # plot by regime variant 

        Choicemat = Subtraction_Matrix( p, traces[t2-1,:], traces[t1,:] )
        maxval      = np.max(np.abs(Choicemat.flatten()))
        print(Choicemat)
        pl5 = ax.imshow(Choicemat, vmin=-maxval, vmax=maxval, cmap='bwr')        # matrix
        plt.xticks(ticks=range(p.n_items),labels=[])
        plt.yticks(ticks=range(p.n_items),labels=[])
        # plt.colorbar(pl5,ax=ax)
        # ax.set_xticklabels(['A','B','C','D','E','F','G'],fontsize=18)
        # ax.set_yticklabels(['A','B','C','D','E','F','G'],fontsize=18)
        # ax.set_xlabel('Item 2',fontsize=18)
        # ax.set_ylabel('Item 1',fontsize=18)
        ax.axis(xmin=-0.5,xmax=p.n_items-0.5,ymin=-0.5,ymax=p.n_items-0.5)
        ax.invert_yaxis()

        # ax.set_title('RNN choice')    

    fig.subplots_adjust(hspace=.5)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    


def Subtraction_Matrix(p,i1,i2):
    
    M       = p.n_items
    output  = np.full((M,M),np.nan)
    for xx in range(M):
        for yy in range(M):
            output[xx,yy] = i1[xx] + i2[yy]

    return output


def Plot_Readout_Study(p, Neurals, Behaviors):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 20})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    T, t1, t2, _, _, _     = tk.parms_test(p,None,0)    

    inds_n            = np.isin(Neurals['TIDs'],Behaviors['TIDs'])
    TIDs_plot         = np.array(Neurals['TIDs'])[inds_n]
    inds_b            = np.isin(Behaviors['TIDs'],TIDs_plot)
    assert np.all( np.array(Neurals['TIDs'])[inds_n] == np.array(Behaviors['TIDs'])[inds_b] )

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(50,50,1200,300) 
    fig.suptitle( '\n\n', fontsize=20, weight='bold')
    
    xlims = []

    #### (1st) Choice availability ##########
    width = 0.5
    ax          = fig.add_subplot(1,5,1)
    vals_all    = [] 
    for vv in range(5):
        ii = np.array([0])
        clr = variant_color(vv,1)
        alpha = 0.4
        inds = np.where( Neurals['Model_variant'] == vv )[0]
        vals = Neurals['Ec'].flatten()[inds] / (t2-t1)
        
        xjit = 1 + vv + width * np.random.rand(len(vals)) - width/2

        # plt.title(str)
        ax.scatter( xjit, np.asarray(vals), color=clr, s=55, alpha=alpha)    

        ax.set_ylim([-0.1,1])
        ax.set_xlim([0.25,5.75])
        ax.set_ylabel('Time to Choice Availability',fontsize=18)
    
        # ax.set_yticks(np.arange(0,1.1,0.1))
        ax.set_xticks(np.arange(1,6,1))
        ax.set_xticklabels(['1','2','3','4','5'])
        ax.set_ylabel('ERI')
        ax.tick_params('both', length=6,  which='major')
        ax.tick_params('both', length=6, which='minor')
        fig.tight_layout()    
        plt.show()
        
        numearly = np.sum(vals <= 0.25)
        numtotal = vals.size
        proportion_earlier = numearly / numtotal
        print( '(readout) %g of %g (%g%%) models below choice 1st quarter (variant: %s)' % \
                (numearly,numtotal,100*proportion_earlier,vv) )
        vals_all = np.append(vals_all,vals)

    numearly_all = np.sum(vals_all <= 0.25)
    numtotal_all = vals_all.size
    meanvals    = np.mean( np.array(vals_all) )
    sdvals      = np.std( np.array(vals_all) )
    print('(readout) %g of %g models have choice avail in 1st quarter' % (numearly_all,numtotal_all) )        
    print('(readout) %0.3f ± %0.3f (s.d.) to choice avail' % (meanvals,sdvals) )        



    #### (2nd) ERI (recoding index) histogram ##########
    for EE in [0,3]:
        if EE == 0:
            titlestr = 'readout exact time'
            ax      = fig.add_subplot(1,5,3)
        elif EE == 3:
            titlestr = 'choice avg time'            
            ax      = fig.add_subplot(1,5,5)
        if p.input_train == 1:
            if p.d1 == 20:
                if EE == 0:
                    numBins = 38
                    Xmin    = -1
                    Xmax    = 2
                    xtick   = 0.5
                    ytick   = 25
                    Ymax    = 100
                elif EE == 3:
                    numBins = 33
                    Xmin    = -2
                    Xmax    = 4
                    xtick   = 1
                    ytick   = 50
                    Ymax    = 150
            else:
                numBins = 40
                Xmin    = -2
                Xmax    = 4
                xtick   = 2
                Ymax    = 100
        elif p.input_train == 0:
            numBins = 28
            Xmax    = 4
            Xmin    = -2
            xtick   = 1
            if EE == 0:
                Ymax    = 75
                ytick   = 25
            else:
                Ymax    = 150
                ytick   = 50
        bins = np.linspace(Xmin,Xmax,numBins)
        for v in range(5):
            inds = Neurals['Model_variant'][inds_n] == v
            vals = np.log10( Neurals['Eri'][EE,inds] )
            xmax = np.ceil( np.max(vals.flatten()) ) + 1
            # print(np.sum(np.isnan(vals)))
            # histout = plt.hist( vals, bins, ec=variant_color(v,0.8), fc=variant_color(v,0.5), histtype='stepfilled', linewidth=3.5)
            # ymax = np.max( histout[0].flatten() )
            ymax = npf.area_histogram(ax, vals, bins, 3, variant_color(v), 0.9, 0.2, 100, 200)
            # if ymax > Ymax:
            #     Ymax = ymax
            ax.plot([0,0],[0,Ymax],'-',color=(0.7,0.7,0.7),linewidth=2)
        ax.set_yticks(np.arange(0,Ymax+ytick,ytick))
        ax.set_ylim([0,Ymax])
        ax.set_ylabel('# of instances',fontsize=18)
        ax.set_xlabel('Encoding ratio (log10)',fontsize=18)
        # if len(xticklabels) != 0:
        #     ax.set_xticklabels(xticklabels,fontsize=18)
        # ax.set_xlim([bins[0],bins[-1]])
        ax.set_xticks(np.arange(Xmin,Xmax+xtick,xtick))
        ax.set_xlim([Xmin,Xmax+0.001])
        ax.set_title('%s\n' % (titlestr),fontsize=17)



    fig.subplots_adjust(hspace=.5)
    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()    


def Convert_Pairs_to_Matrix(p, Data, Pairs):
    # Data: [pair,modelnum]
    # Pairs: [pair][item1,item2]
    nummodels   = Data.shape[1]
    Data_mat    = np.full((p.n_items,p.n_items,nummodels),np.nan)
    for xx in range(p.n_items):         # rows
        for yy in range(p.n_items):     # columns
            if yy >= xx:  # top triangle of matrix
                foundflag = False
                for pp in range(len(Pairs)):
                    if (xx == Pairs[pp][0] and yy == Pairs[pp][1]) or \
                        (xx == Pairs[pp][1] and yy == Pairs[pp][0]):
                        foundflag = True
                        break
                if foundflag:
                    Data_mat[xx,yy,:] = Data[pp,:]
    return Data_mat

def Convert_Pairs_to_Matrix_full(p, Data, Pairs):
    # Data: [pair,modelnum]
    # Pairs: [pair][item1,item2]
    nummodels   = Data.shape[1]
    Data_mat    = np.full((p.n_items,p.n_items,nummodels),np.nan)
    for xx in range(p.n_items):         # rows
        for yy in range(p.n_items):     # columns
            if yy >= xx:  # top triangle of matrix
                foundflag = False
                for pp in range(len(Pairs)):
                    if (xx == Pairs[pp][0] and yy == Pairs[pp][1]) or \
                        (xx == Pairs[pp][1] and yy == Pairs[pp][0]):
                        foundflag = True
                        break
                if foundflag:
                    Data_mat[xx,yy,:] = Data[pp,:]
                    Data_mat[yy,xx,:] = Data[pp,:]
    return Data_mat    

def window_string(tt):
    if tt == 0:
        str = '1st quarter'
    elif tt == 1:
        str = '2nd quarter'
    elif tt == 2:
        str = '3rd quarter'
    elif tt == 3:
        str = '4th quarter'
    elif tt == 4:
        str = 'Last timestep'
    return str

def Plot_Neff(Neurals):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    bins = np.linspace(0,8,60)

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(1200,100,800,1000) 
    fig.suptitle( 'Dimensionality of RNNs during delay', fontsize=24, weight='bold')
    for tt in range(6):   
        if tt < 5:
            vals = Neurals['Neff_win'][tt,:] 
        else:
            vals = Neurals['Neff_win'][3,:]/Neurals['Neff_win'][0,:] 

        ax = fig.add_subplot(6,1,tt+1)  

        for v in range(5):
            inds = Neurals['Model_variant'] == v
            ax.hist(vals[inds],bins,color=variant_color(v),alpha=0.5,density=True)

        if tt == 4:
            ax.set_xlabel('N_eff',fontsize=20)
        if tt < 5:
            ax.set_ylabel('%s\n' % (window_string(tt)),fontsize=20)

    fig.subplots_adjust(hspace=.4)

    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()


def variant_color(v,alpha=1):
    import matplotlib
    if v == 0:
        color = 'k'                        
    elif v == 1:
        color = 'b'
    elif v == 2:
        color = 'green'
    elif v == 3:
        color = 'orange'
    elif v == 4:
        color = 'red'
    vals  = matplotlib.colors.to_rgb(color)
    clr   = (vals[0],vals[1],vals[2],alpha)
    return clr

def variant_string(v):
    if v == 0:
        str = 'Simple (high)'
    elif v == 1:
        str = 'Simple (low)'
    elif v == 2:
        str = 'Intermediate'
    elif v == 3:
        str = 'Complex (low)'
    elif v == 4:
        str = 'Complex (high)'
    return str


def Plot_MDS_stimuli_single_model(p,Neurals,measures_to_plot=[0,1,2]):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.size': 16})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    MAXVAL  = 6
    mm      = 0

    I           = np.argsort(Neurals['GI'][0,2,:])      # lowest cosine mean to highest
    if 0:
        ind     = I[-1]    # most positive
    elif 0:
        ind     = I[0]     # most negative
    else:
        ind     = 12     # most negative
    TID        = np.array([Neurals['TIDs']]).flatten()[ind]  
    cosinedelta = Neurals['GI'][0,2,ind]

    DT          = 0
    windows     = [0,1,2,3]  # 1st quarter

    # Define MDS #
    embedding = MDS( n_components=2, random_state=1, metric=True, dissimilarity='precomputed' )

    fig = plt.figure()
    plt.get_current_fig_manager().window.setGeometry(200+mm*100,100,1500,500) 
    fig.suptitle( 'Stimuli-based MDS\nTID: %g' % (TID), fontsize=24, weight='bold')
    for w in range(len(windows)):
        ax          = fig.add_subplot(1,4,1+w)  
        ww = windows[w]
        data        = Neurals['G_win'][DT,mm,ww,:,:]  # data: [pair,model]
        if mm == 0 or mm == 2:
            datasim = 1-data
        else:
            datasim = data
        datamat     = Convert_Pairs_to_Matrix_full( p, datasim, Neurals['Pairs'] )
        data_mds    = embedding.fit_transform( datamat[:,:,ind] )
        ax.scatter( data_mds[:,0], data_mds[:,1], color=rankcolors(p), s=120 )
        ax.set_xlim([-MAXVAL,MAXVAL])
        ax.set_ylim([-MAXVAL,MAXVAL])
        ax.set_title( '%s\n' % (window_string(ww)), fontsize=22 )
        ax.set_aspect('equal', 'box')  

    fig.subplots_adjust(hspace=1)

    plt.draw()     
    plt.show()
    plt.pause(0.0001)
    plt.ion()
    plt.ioff()
    plt.ion()



def Plot_MDS_stimuli(p,Neurals,measures_to_plot=[0]):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.size': 16})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    clrs    = rankcolors(p)
    MAXVAL  = 30

    DT = 0      # 0: RNN data, 1: random vectors
    ww = 4      # 3: last quarter, 4: final time step

    # Define MDS #
    embedding = MDS( n_components=2, random_state=1, metric=True, dissimilarity='precomputed' )

    for mm in measures_to_plot:
        if mm == 0:
            mstr = 'Cosine similarity'
        elif mm == 1:
            mstr = 'Euclidean'
        elif mm == 2:
            mstr = 'Pearson r'

        fig = plt.figure()
        plt.get_current_fig_manager().window.setGeometry(200+mm*100,100,1500,1000) 
        fig.suptitle( 'Stimuli-based MDS (example RNNs)\n Distance: %s' % (mstr), fontsize=24, weight='bold')
        for v in range(5):
            inds        = np.where(Neurals['Model_variant'] == v)[0]
            data        = Neurals['G_win'][DT,mm,ww,:,inds].T  # data: [pair,model]
            if mm == 0 or mm == 2:
                datasim = 1-data
            else:
                datasim = data
            datamat     = Convert_Pairs_to_Matrix_full(p, datasim, Neurals['Pairs'])
            nummodels   = datamat.shape[2]
            if 0:
                inds_plot   = np.random.permutation(nummodels)[:10]
            elif 1:
                inds_plot   = np.arange(10)  # first 10 models
            for ex in range(inds_plot.size):
                ax          = fig.add_subplot(5,10,v*10+1+ex)  
                data_mds    = embedding.fit_transform( datamat[:,:,inds_plot[ex]] )
                ax.scatter( data_mds[:,0], data_mds[:,1], color=clrs, s=60 )
                ax.set_xlim([-MAXVAL,MAXVAL])
                ax.set_ylim([-MAXVAL,MAXVAL])
                ax.set_title( '%g' % (Neurals['TIDs'][inds[inds_plot[ex]]]), fontsize=14 )
                if ex == 0:
                    ax.set_ylabel(variant_string(v), fontsize=20, color = variant_color(v))
                    ax.set_xticklabels([])            
                elif ex > 0:
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

        fig.subplots_adjust(hspace=.4)

        plt.draw()     
        plt.show()
        plt.pause(0.0001)
        plt.ion()
        plt.ioff()
        plt.ion()


def Spatial_PCA(p, F, Data, TT = 0, Option = 0, ThreeD = 0):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 2})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon
    import copy

    from sklearn.decomposition import PCA
    
    # OPTION = 0:   # 0: fit PCA at each timestep, 1: last timestep

    pca         = PCA()   
    T, t1, t2, X, _, _   = tk.parms_test(p, F, TT)
    i1inds    = Rank1_trials(p)
    N         = i1inds.size  # maximum dimension
    M         = X.shape[2]   # get # of conditions from each time/input condition

    # last timestep pca #
    if Option == 1:
        pca.fit( Data['DATA'][TT][t2-1,:,i1inds] )

    PDATA                = np.zeros([T,N,3])     # [T,N,PC]
    for t in range(T):
        # Fit PCA
        dat             = Data['DATA'][TT][t,:,i1inds]
        if Option == 0:
            pdat            = pca.fit_transform( dat )
        elif Option == 1:
            pdat            = pca.transform( dat )
        PDATA[t,:,:]    = pdat[:,:3]
    
    # plot settings #
    MAXVAL      = 15

    greens = np.array( [[0,135,20],[43,155,59],[85,175,98],[128,195,138],[170,215,177],[213,235,216],[240,240,240] ] ) / 255
    times = [p.t1,int(np.mean((p.t1,p.t2))),p.t2-1]


    if ThreeD == 0:

        fig = plt.figure(constrained_layout=True,figsize=(14,5))
        plt.get_current_fig_manager().window.setGeometry(50,50,1000,300) 
        for T in range(3):
            ax = fig.add_subplot(1,3,T+1)
            t = times[T]
            ax.scatter(PDATA[t,:,0],PDATA[t,:,1],marker='o',s=160,edgecolor='k',linewidth=1.5,color=greens)  # x: PC0, y: PC1
            ax.set_aspect('equal', 'box')         
            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim([-MAXVAL,MAXVAL])
            ax.set_ylim([-MAXVAL,MAXVAL])     
                 
    elif ThreeD == 1:

        fig = plt.figure(constrained_layout=True,figsize=(14,5))
        plt.get_current_fig_manager().window.setGeometry(100,300,1200,300) 

        times = [p.t1,int(np.mean((p.t1,p.t2))),p.t2-1]
        for T in range(3):
            ax = fig.add_subplot(1,3,T+1,projection='3d')
            ax.scatter3D(PDATA[times[T],:,0],PDATA[times[T],:,1],PDATA[times[T],:,2], \
                         marker='o',s=100,color=greens,alpha=1,linewidth=1.5,edgecolor='k')  # x: PC0, y: PC1
            # ax.scatter3D(np.mean(PDATA[times[T],:,0]),np.mean(PDATA[times[T],:,1]),np.mean(PDATA[times[T],:,2]), \
            #              marker='o',s=100,color='k')  # x: PC0, y: PC1
            ax.plot3D(PDATA[times[T],0:(p.n_items),0],PDATA[times[T],0:(p.n_items),1],PDATA[times[T],0:(p.n_items),2], \
                            '-',linewidth=3,color='k',alpha=0.2)  # x: PC0, y: PC1

            # ax.set_aspect('equal','box')
            # ax.set_box_aspect((1, 1, 1))
            # ax.set_aspect('equal', 'box')         
            # ax.set_xticks([])
            # ax.set_yticks([])
            MAXVAL = 6
            ax.set_xlim([-MAXVAL,MAXVAL])
            ax.set_ylim([-MAXVAL,MAXVAL])   
            ax.set_zlim([-MAXVAL,MAXVAL])   
            # ax.set_xlim([-MAXVAL,MAXVAL])
            # ax.set_ylim([-MAXVAL,MAXVAL])  
    
    plt.draw()     
    plt.show()
    plt.pause(0.0001)

    plt.ion()
    plt.ioff()
    plt.ion()
    
    return PDATA

def Plot_Delay_PCA(p, Neurals, Behaviors, jobname, TIDs_plot = []):

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({'font.size': 2})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.patches import Polygon

    inds_n            = np.isin(Neurals['TIDs'],Behaviors['TIDs'])
    TIDs              = np.array(Neurals['TIDs'])[inds_n]
    inds_b            = np.isin(Behaviors['TIDs'],TIDs)
    assert np.all( np.array(Neurals['TIDs'])[inds_n] == np.array(Behaviors['TIDs'])[inds_b] )

    if TIDs_plot is None:
        inds_select = np.arange(0,6)
        # for v in [0,1,2,3,4]:  #reversed(range(numVariants)):
            # inds     = np.where(Model_variant == v)[0]
    else:
        inds_select = np.where( np.isin( np.asarray(Neurals['TIDs']),TIDs_plot) )[0]


    Model_variant   = Neurals['Model_variant'][inds_n]    
    numVariants     = np.unique(Model_variant).size

    t1,t2           = (p.t1,p.t2)

    MAXVAL      = 15
    COM         = 1
    greens = np.array( [[0,135,20],[43,155,59],[85,175,98],[128,195,138],[170,215,177],[213,235,216],[240,240,240] ] ) / 255
    # times = [p.t1,int(np.mean((p.t1,p.t2))),p.t2-1]
    times = [p.t1,p.t2-1]

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(100,100,1500,800) 

    if 1:           #   Three snapshots + trajectories  # 
        for ii in range(inds_select.size):   # instances
            ind     = inds_select[ii]
            PDATA   = Neurals['Data_G'][ind]['DATA'][0]
            TID     = TIDs[ind]
            coll_e      = Neurals['GI'][0,0,0,ind]   # collinearity, early delay, exact time, 
            coll_l      = Neurals['GI'][0,1,0,ind]   # collinearity, late delay, exact time, 
            coll_del    = Neurals['GI'][0,2,0,ind]   # collinearity, change, exact time, 
            mac         = Neurals['GI'][0,8,0,ind]   # collinearity, change, exact time, 
            for T in range(3):
                ax = fig.add_subplot(6,15,1+T+ii*3)
                if T < 2:      # snapshot
                    t = times[T]
                    if T == 0:
                        mk = 35
                    elif T == 1:
                        mk = 60
                    if COM: # COM and scaled
                        com     = np.mean(PDATA[t,0:2,:],axis=1)[:,np.newaxis]
                        data    = PDATA[t,0:2,:] - com
                        maxval  = np.max(np.abs(data.flatten()))
                        maxval  = maxval + 0.4 * maxval  
                        ax.scatter(data[0,:],data[1,:],marker='o',s=mk,edgecolor='k',linewidth=1.5,color=greens)  # x: PC0, y: PC1
                        ax.plot(data[0,:],data[1,:],'-',color='k',linewidth=1.5,alpha=1,zorder=0)  # x: PC0, y: PC1
                        ax.set_xlim([-maxval,maxval])
                        ax.set_ylim([-maxval,maxval])     
                    else:  # original 
                        ax.scatter(PDATA[t,0,:],PDATA[t,1,:],marker='o',s=50,edgecolor='k',linewidth=1.5,color=greens)  # x: PC0, y: PC1
                        ax.set_xlim([-MAXVAL,MAXVAL])
                        ax.set_ylim([-MAXVAL,MAXVAL])     
                elif T == 2:    # continuous
                    if COM:                 # inherits com from T=2
                        ax.plot(PDATA[t1:t2,0,:]-com[0],PDATA[t1:t2,1,:]-com[1],'-',color='k',linewidth=1.5,alpha=0.2,zorder=5)  # x: PC0, y: PC1
                        ax.scatter(PDATA[t1,0,:]-com[0],PDATA[t1,1,:]-com[1],marker='o',s=35,edgecolor='k',linewidth=1.5,color=greens,zorder=10)  # x: PC0, y: PC1
                        ax.scatter(PDATA[t2-1,0,:]-com[0],PDATA[t2-1,1,:]-com[1],marker='o',s=60,edgecolor='k',linewidth=1.5,color=greens,zorder=0)  # x: PC0, y: PC1
                        ax.set_xlim([-maxval,maxval])
                        ax.set_ylim([-maxval,maxval])     
                    else:
                        ax.plot(PDATA[t1:t2,0,:],PDATA[t1:t2,1,:],'-',color='k',linewidth=1.5,alpha=0.2)  # x: PC0, y: PC1
                        for tt in [0,1]:
                            ax.scatter(PDATA[times[tt],0,:],PDATA[times[tt],1,:],marker='o',s=50,edgecolor='k',linewidth=1.5,color=greens)  # x: PC0, y: PC1
                        ax.set_xlim([-MAXVAL,MAXVAL])
                        ax.set_ylim([-MAXVAL,MAXVAL])     

                ax.set_aspect('equal', 'box')         
                ax.set_xticks([])
                ax.set_yticks([])
                
                if T == 0:
                    ax.set_title('TID: %g\ncol: (%0.2f,%0.2f) mac: %0.2f' % (TID,coll_l,coll_del,mac),fontsize=11)


    else:           #    Trajectories  #
        count = 0
        for v in [0,1,2,3,4]:  #reversed(range(numVariants)):
            inds        = np.where(Model_variant == v)[0]
            inds_select = np.arange(0,10)
            for ii in range(inds_select.size):   # instances
                count += 1
               
                PDATA       = Neurals['Data_G'][inds[inds_select[ii]]]['DATA'][0]
                TID         = TIDs[inds[inds_select[ii]]]
               
                ax = fig.add_subplot(5,10,count)

                ax.plot(PDATA[t1:t2,0,:],PDATA[t1:t2,1,:],'-',color='k',linewidth=1.5,alpha=0.2,zorder=-10)  # x: PC0, y: PC1
                ax.scatter(PDATA[times[0],0,:],PDATA[times[0],1,:],marker='o',s=40,edgecolor='k',linewidth=1.25,color=greens,alpha=1)  # x: PC0, y: PC1
                ax.scatter(PDATA[times[2],0,:],PDATA[times[2],1,:],marker='o',s=80,edgecolor='k',linewidth=1.25,color=greens)  # x: PC0, y: PC1

                ax.set_title('TID: %g' % (TID),fontsize=9)

                ax.set_aspect('equal', 'box')         
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim([-MAXVAL,MAXVAL])
                ax.set_ylim([-MAXVAL,MAXVAL])     




    # elif ThreeD == 1:
    #     for T in range(3):
    #         ax = fig.add_subplot(1,3,T+1,projection='3d')
    #         ax.scatter3D(PDATA[times[T],:,0],PDATA[times[T],:,1],PDATA[times[T],:,2], \
    #                      marker='o',s=100,color=greens,alpha=1,linewidth=1.5,edgecolor='k')  # x: PC0, y: PC1
    #         # ax.scatter3D(np.mean(PDATA[times[T],:,0]),np.mean(PDATA[times[T],:,1]),np.mean(PDATA[times[T],:,2]), \
    #         #              marker='o',s=100,color='k')  # x: PC0, y: PC1
    #         ax.plot3D(PDATA[times[T],0:(p.n_items),0],PDATA[times[T],0:(p.n_items),1],PDATA[times[T],0:(p.n_items),2], \
    #                         '-',linewidth=3,color='k',alpha=0.2)  # x: PC0, y: PC1
    #         # ax.set_aspect('equal','box')
    #         # ax.set_box_aspect((1, 1, 1))
    #         # ax.set_aspect('equal', 'box')         
    #         # ax.set_xticks([])
    #         # ax.set_yticks([])
    #         ax.set_xlim([-MAXVAL,MAXVAL])
    #         ax.set_ylim([-MAXVAL,MAXVAL])   
    #         ax.set_zlim([-MAXVAL,MAXVAL])   
    
    plt.draw()     
    plt.show()
    plt.pause(0.0001)

    plt.ion()
    plt.ioff()
    plt.ion()
    
    return PDATA



def Dataframe(C):

    import pandas as pd
    dist    = np.array([])
    pair    = np.array([])
    vals    = np.array([])

    for dd in range(len(C)):
        numpair = C[dd].size
        vals  = np.concatenate( (vals, C[dd]), axis=0 )
        pair  = np.concatenate( (pair, np.arange(numpair)), axis=0 )
        dist  = np.concatenate( (dist, np.full((numpair),dd+1) ), axis=0 )

    C_df = pd.DataFrame( vals, columns=['propcorr'])
    C_df.insert(0, "pair", pair)
    C_df.insert(0, "dist", dist)

    return C_df


def Mechanism_1_output(p, model_lin, Out_x):

    # initialize outputs #
    B           = Out_x.shape[1]   # no. of trial types
    Choice      = np.zeros([B])
    RT          = np.zeros([B])
    Choiceval   = np.zeros([B])

    # project onto saddle #
    x       = Out_x[p.t2+1,:,:]               # [T,B,N], activation right after second stimulus
    xstar   = model_lin.xstars[1,:].detach().numpy()         # activation state of saddle fixed-point
    jac     = model_lin.Jacs[:,:,1].detach().numpy()

    eigval, eigvec  = np.linalg.eig(jac)   # calculate 
    xc              = x - xstar            # center data on fixed point
    xproj           = np.dot( xc , np.real(eigvec[:,0]) ) # project

    Choiceval           = -xproj                     # 
    Choice              = np.sign(Choiceval)
    Choice[Choice<0]    = 0 
    RT                  = 1 - np.abs(Choiceval) / np.max(np.abs(Choiceval)) 

    return Choice, Choiceval, RT



def plot_Mechanism1_summary(fig0, p, F, model_lin, TT = 0):  

  import matplotlib as mpl
  import matplotlib.pyplot as plt
  # mpl.use("Qt5Agg") # set the backend
  mpl.rcParams.update({'font.size': 14})
  mpl.rcParams['font.sans-serif'] = "Helvetica"
  mpl.rcParams['font.family'] = "sans-serif"
  import matplotlib.patches as patch
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.patches   as patch

  # preliminary #
  INDS,trains,probes        = trial_parser(p)  # get important indices (for transitive inference)
  tind,eind,pind,sind,iind  = INDS # 0: train, 1: edges, 2: probe, 3: sames, 4: initial
  N                         = p.n_items
  n_items                   = N
  X_input                   = F['X_input']
  train_table               = F['train_table']

  # run model #
  _, _, _, X, input, ztarg     = tk.parms_test(p,X_input,TT)
  with torch.no_grad():
      D     = tk.run_linmodel( model_lin, p, input, hidden_output=True )

  # parse output #
  Choice, Choiceval, RT = Mechanism_1_output(p, model_lin, D['Out_x'] )
  Choicemat             = Choice.reshape(N,N)
  Choicevalmat          = Choiceval.reshape(N, N)  
  RTmat                 = RT.reshape(N,N)

  from mpl_toolkits.axes_grid1 import make_axes_locatable

  # Plot RNN choice ##
  ax5 = fig0.add_subplot(3,3,7)
  pl5 = ax5.imshow(Choicemat, vmin=0, vmax=1, cmap='bwr')        # matrix
#   plot_squares((Choicemat == -1),5,ax5)  # RNN chooses neither: green squares
#   plot_squares((Choicemat == -2),4,ax5)  # RNN chooses both: black squares
  plot_squares(train_table,1,ax5)   # squares
  plot_squares(probes,2,ax5)        # circles
  plt.xticks(ticks=range(p.n_items),labels=[])
  plt.yticks(ticks=range(p.n_items),labels=[])
  plt.colorbar(pl5,ax=ax5)
  ax5.axis(xmin=-0.5,xmax=n_items-0.5,ymin=-0.5,ymax=n_items-0.5)
  ax5.invert_yaxis()
  ax5.set_title('RNN choice')

  # Plot RNN output value ## (at right, numerical) 
  ax6 = fig0.add_subplot(3,3,8)
  maxval = np.max(np.abs(Choicevalmat[:]))
  pl6 = ax6.imshow(Choicevalmat, vmin=-maxval, vmax=+maxval, cmap='bwr')        # matrix
  plot_squares(train_table,1,ax6)  # squares
  plot_squares(probes,2,ax6)    # circles
  plt.xticks(ticks=range(p.n_items),labels=[])
  plt.yticks(ticks=range(p.n_items),labels=[])
  plt.colorbar(pl6,ax=ax6)
  ax6.axis(xmin=-0.5,xmax=n_items-0.5,ymin=-0.5,ymax=n_items-0.5)
  ax6.invert_yaxis()
  ax6.set_title('RNN output values')

  # Plot RT  ## (at right, numerical) 
  ax7 = fig0.add_subplot(3,3,9)
  pl7 = ax7.imshow(RTmat, vmin=0, vmax=np.max(RT)+.05*np.max(RT), cmap='binary')        # matrix
#   plot_squares((Choicemat == -1),5,ax7)  # RNN chooses neither: green squares
#   plot_squares((Choicemat == -2),5,ax7)  # RNN chooses both: green squares
  plot_squares(train_table,1,ax7)        # squares
  plot_squares(probes,2,ax7)             # circles
  plt.xticks(ticks=range(p.n_items),labels=[])
  plt.yticks(ticks=range(p.n_items),labels=[])
  plt.colorbar(pl7,ax=ax7)
  ax7.axis(xmin=-0.5,xmax=n_items-0.5,ymin=-0.5,ymax=n_items-0.5)
  ax7.invert_yaxis()
  ax7.set_title('RNN reaction times')

  fig0.subplots_adjust(hspace=.4)





###### TI plotting functions #####################################################

def groupplot(gg,INDS):
  tind,eind,pind,sind,iind = INDS # 0: train, 1: edges, 2: probe, 3: sames, 4: initial
  mk = 'v'       
  if gg == 0:             # Train   (dark RB)
      ii = tind 
      clr = [tuple([0,0,.6]),tuple([.6,0,0])]
      mksize = 140
      lw = 2
      mk = 'D'       
  elif gg == 1:           # Edges   (light RB)
      ii = eind
      lw = 1.5
      mksize = 300
      clr = [tuple([.5,.5,1]),tuple([1,.5,.5])]
  elif gg == 2:           # Probe   (vivid RB)
      mksize = 300
      lw = 1.5
      ii = pind
      clr = [tuple([0,0,1]),tuple([1,0,0])]
  elif gg == 3:           # Sames   (grey)
      mksize = 50
      ii = sind
      clr = [tuple([.7,.7,.7]),tuple([.7,.7,.7])]
      lw = 1       
  elif gg == 4:            # Initial (black)
      mksize = 300
      lw = 1.5
      ii = iind
      clr = [tuple([0,0,0]),tuple([0,0,0])] 
  return ii, clr, mksize, lw, mk


def ranks(p,pp):  # returns item 1, item 2 ranks for given trial type
    XV, YV  = np.meshgrid(np.arange(p.n_items),np.arange(p.n_items),indexing='ij')    
    XV, YV  = XV.flatten(), YV.flatten()   
    rank1   = XV[pp]      # 1st stim
    rank2   = YV[pp]      # 2nd stim   
    return rank1, rank2

def Rankvector(p):
    XV, YV  = np.meshgrid(np.arange(p.n_items),np.arange(p.n_items),indexing='ij')    
    XV, YV  = XV.flatten(), YV.flatten()   
    rank1vec   = XV      # 1st stim
    rank2vec   = YV      # 2nd stim   
    return rank1vec, rank2vec 

def Rank1_trials(p):
    mat         = np.zeros( (p.n_items,p.n_items) )
    mat[:,0]    = 1         # 1st column
    inds        = np.where( mat.flatten() )[0]   
    return inds


def plot_Rank1(p, ax, x, y, pp, **kwargs):
    rank1, _    = ranks(p,pp)
    str         = '%d' % (rank1)
    ax.text(x+0.05, y+0.05, str, **kwargs )

    

def embedcolor(pp, n_items, OPT):

  import matplotlib as mpl
  clr = []
  XV, YV  = np.meshgrid(np.arange(n_items),np.arange(n_items),indexing='ij')    
  XV, YV  = XV.flatten(), YV.flatten()  
  symdist   = YV[pp] - XV[pp] 
  jointrank = XV[pp] + YV[pp]
#   print(jointrank)

  for LR in [0,1]:
      clrr = [0,0,0]
      if OPT == 0:
          if LR == 0:
              clrr = [1,0,0]
          elif LR == 1:
              clrr = [0,0,1]
      elif np.isin(OPT,[1,2]):        # Red-Blue (i.e. keeps the output color)
          clr0 = [tuple([1,0,0]),tuple([0,0,1])] 
          if OPT == 1:
              indval    = XV[pp]      # 1st stim
          elif OPT == 2:
              indval    = YV[pp]      # 2nd stim
          maxind  = np.argmax(clr0[LR])
          maxprop = np.max(clr0[LR]) 
          prop    = (maxprop-.1)*(indval+1)/(n_items)    # proportion of full color intensity                
          for rgb in [0,1,2]:
              if rgb != maxind:
                  clrr[rgb] = prop
              else:
                  clrr[rgb] = clr0[LR][rgb]
      elif np.isin(OPT,[3,4]):       # Green
          greens = np.array( [[0,135,20],[43,155,59],[85,175,98],[128,195,138],[170,215,177],[213,235,216],[240,240,240] ] ) / 255
          if OPT == 3:
              indval    = XV[pp]      # 1st stim
          elif OPT == 4:
              indval    = YV[pp]      # 2nd stim   
        #   prop    = .3*(indval+1)/(n_items)    # proportion of full color intensity                
        #   clrr = (prop,.3,prop) 
          clrr = greens[indval,:]
      elif OPT == 5:    # Symbolic Distance, greyscale (unsigned)
          indval = np.abs(XV[pp]-YV[pp])
          prop  = indval/(n_items-1)    # proportion of full color intensity                
          clrr = (prop,prop,prop)  
      elif OPT == -1:   # 1st stim rank
          indval    = XV[pp]      
          prop      = (indval+1)/(n_items)    # proportion of full color intensity                
          cmap      = mpl.cm.get_cmap('twilight')
          clrr       = cmap(prop)[:3]
      elif OPT == 6:       # Symbolic Distance, Signed (red white blue)
        #   reds  = np.array( [[255,135,135],[255,108,108],[255,81,81],[255,54,54],[255,27,27],[255,0,0] ] ) / 255
        #   blues = np.array( [[135,135,255],[108,108,255],[81,81,255],[54,54,255],[27,27,255],[0,0,255] ] ) / 255
        #   reds  = np.array( [[255,200,200],[255,160,160],[255,120,120],[255,80,80],[255,40,40],[255,0,0] ] ) / 255
        #   blues = np.array( [[200,200,255],[160,160,255],[120,120,255],[80,80,255],[40,40,255],[0,0,255] ] ) / 255
          reds  = np.array( [[255,210,210],[255,170,170],[255,130,130],[255,90,90],[255,45,45],[255,0,0] ] ) / 255
          blues = np.array( [[210,210,255],[170,170,255],[130,130,255],[90,90,255],[45,45,255],[0,0,255] ] ) / 255
          prop      = np.abs( symdist ) / (n_items-1)
          if symdist > 0:
            # clrr       = [1*prop,0,0]
            clrr       = reds[np.abs(symdist)-1]
          elif symdist < 0:
            # clrr       = [0,0,1*prop]
            clrr       = blues[np.abs(symdist)-1]
          else:
            clrr       = [0.5,0.5,0.5]
      elif OPT == 7:       # End item (black vs. grey)
          enditem      = np.any( np.isin(np.array([XV[pp],YV[pp]]),np.array([0,n_items])) )
          if enditem:
            clrr       = [0,0,0]        # has an End item
          else:
            clrr       = [.6,.6,.6]     # does not have End item
      elif OPT == 8:        # End order, 1st (purple) vs. 2nd (orange)
          i1 = XV[pp]
          i2 = YV[pp]
          M  = n_items-1
          if i1 == 0 and ~np.isin(i2,[0,M]):
              clrr = [.6,0,.8]      # purple        # A first
              if i1 == 0 and i2 == 1:               # AB
                  clrr = [1,.3,1]
          elif i2 == 0 and ~np.isin(i1,[0,M]):
              clrr = [.6,.4,0]      # orange        # A second
              if i2 == 0 and i1 == 1:               # BA
                  clrr = [.95,.89,0]
          elif i1 == M and ~np.isin(i2,[0,M]):
              clrr = [.6,0,.8]      # purple        # G first
            #   if i1 == M and i2 == M-1:               # GF
            #       clrr = [1,0,1]
          elif i2 == M and ~np.isin(i1,[0,M]):
              clrr = [.6,.4,0]      # orange        # G second
            #   if i2 == M and i1 == M-1:               # GF
            #       clrr = [.9,.7,0]
          else:
              clrr = [.6,.6,.6]
      elif OPT == 9:       # joint rank
        #   reds  = np.array( [[255,200,200],[255,160,160],[255,120,120],[255,80,80],[255,40,40],[255,0,0] ] ) / 255
        #   blues = np.array( [[200,200,255],[160,160,255],[120,120,255],[80,80,255],[40,40,255],[0,0,255] ] ) / 255
          prop      = np.abs( jointrank ) / (n_items*2)
          clrr = [prop,prop,prop]
        #   if symdist > 0:
        #     clrr       = [1*prop,0,0]
        #     # clrr       = reds[symdist-1]
        #   elif symdist < 0:
        #     clrr       = [0,0,1*prop]
        #     # clrr       = blues[np.abs(symdist)-1]
        #   else:
        #     clrr       = [0.5,0.5,0.5]
      elif OPT == 10:        # End order, 1st (purple) vs. 2nd (orange)
          i1 = XV[pp]
          i2 = YV[pp]
          M  = n_items-1
          if i1 == 0 and ~np.isin(i2,[0,M]):
              clrr = [.6,0,.8]      # purple        # A first
          elif i2 == 0 and ~np.isin(i1,[0,M]):
              clrr = [.6,.4,0]      # orange        # A second
          elif i1 == M and ~np.isin(i2,[0,M]):
              clrr = [.6,0,.8]      # purple        # G first
          elif i2 == M and ~np.isin(i1,[0,M]):
              clrr = [.6,.4,0]      # orange        # G second
          else:
              clrr = [.6,.6,.6]      
      clr.append(tuple(clrr))



  return clr

def rankcolors(p):
    # color of items #
    clrr = []
    N = p.n_items
    for m in range(N):
        prop    = .75*(m+1)/(N)    # proportion of full color intensity                
        clrr.append([prop,.75,prop])
    return clrr

def plot_squares(draw_mat,polytype,ax):

    import matplotlib
    import matplotlib as mpl
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    matplotlib.use("Qt5Agg") # set the backend
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rcParams['font.sans-serif'] = "Helvetica"
    matplotlib.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D

    indmat = draw_mat > 0
    it = np.nditer(indmat, flags=['multi_index']) # iterate 
    for xx in it:
        if indmat[it.multi_index[0],it.multi_index[1]]:
            xc = it.multi_index[1]
            yc = it.multi_index[0]
            if polytype == 1:
                l  = 0.47
                xl = [xc-l,xc+l,xc+l,xc-l,xc-l]
                yl = [yc-l,yc-l,yc+l,yc+l,yc-l]
                ax.plot(xl,yl,'black',linewidth=2)
            elif polytype == 2:
                ax.add_patch(patch.Circle((xc, yc), .3, color='k',fill=False,linewidth=2))
            elif polytype == 3:
                l  = 0.46
                xl = [xc-l,xc+l,xc+l,xc-l,xc-l]
                yl = [yc-l,yc-l,yc+l,yc+l,yc-l]
                ax.fill(xl,yl,color='white')
            elif polytype == 4:
                l  = 0.46
                xl = [xc-l,xc+l,xc+l,xc-l,xc-l]
                yl = [yc-l,yc-l,yc+l,yc+l,yc-l]
                ax.fill(xl,yl,color='black')
            elif polytype == 5:
                l  = 0.46
                xl = [xc-l,xc+l,xc+l,xc-l,xc-l]
                yl = [yc-l,yc-l,yc+l,yc+l,yc-l]
                ax.fill(xl,yl,color='#e4e4a1')


def Readout_plot(p, Data, TT_plot, YSCALE, color_embed, title=''):

  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  mpl.use("Qt5Agg") # set the backend
  mpl.rcParams.update({'font.size': 14})
  mpl.rcParams['font.sans-serif'] = "Helvetica"
  mpl.rcParams['font.family'] = "sans-serif"
  import matplotlib.patches as patch
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.patches   as patch

  pdata = Data['PDATA']
  rdata = Data['RDATA']

  MAXPCPLOT = 2
  Plot_Initial_Share = 1
  _,t1,t2,dur,jit      = (p.T,p.t1,p.t2,p.dur,np.asarray(p.jit))
  INDS,_,_                    = trial_parser(p)

  fig = plt.figure()
  plt.get_current_fig_manager().window.setGeometry(770,600,400,400) 
  gs   = fig.add_gridspec(MAXPCPLOT,1)
  fig.suptitle( title, fontsize=17, weight='bold')  
  for TT in TT_plot:
      T = pdata[TT].shape[0]
      for pc in range(MAXPCPLOT):
          p1   = fig.add_subplot(gs[pc])
          if p.T > 1:
            if pc > 0:
              p1.plot([t1,t1],[-YSCALE,YSCALE],linewidth=3,color=(.86,.86,.86),zorder=0)
              p1.plot([t2,t2],[-YSCALE,YSCALE],linewidth=3,color=(.86,.86,.86),zorder=0)
            elif pc == 0:
              p1.plot([t1,t1],[0,1.05],linewidth=3,color=(.86,.86,.86),zorder=0)
              p1.plot([t2,t2],[0,1.05],linewidth=3,color=(.86,.86,.86),zorder=0)
          if TT == 2:
              if pc == 0:
                  p1.plot(rdata[TT][:,pc,0], linewidth=1.5,color=(.5,.5,.5),zorder=1)  # trajectory
              else:
                  p1.plot(pdata[TT][:,pc,0], linewidth=1.5,color=(.5,.5,.5),zorder=1)  # trajectory
              p1.set_aspect('auto')
              continue
          else:  # trajectory            
              alpha = 1
              for gg in [0,1,2]:     # Stim pair types  0: train, 1: edges, 2: probes
                  ii, clr, mksize, _, _ = groupplot(gg,INDS)
                  for LR in [0,1]:
                      if ii[LR].size == 0:
                         continue
                      for pp in ii[LR]:
                        clr = embedcolor(pp, p.n_items, color_embed) # 6: sym-dist, signed, 7: end-item (yes vs no), 8: end-order 1st vs. 2nd
                        clrchoice = embedcolor(pp,p.n_items,6)
                        if pc == 0:   # readout
                            p1.plot(np.arange(0,t1),nm.Tanh_readout(p,rdata[TT][0:t1,1,pp]), linewidth=2,color=(0.6,0.6,0.6), alpha=alpha)             
                            p1.plot(np.arange(t1-1,t2),nm.Tanh_readout(p,rdata[TT][t1-1:t2,1,pp]), linewidth=2,color=clr[LR], alpha=alpha)             
                            p1.plot(np.arange(t2-1,T),nm.Tanh_readout(p,rdata[TT][(t2-1):T,1,pp]), linewidth=2,color=clrchoice[LR], alpha=alpha)             
                        elif pc == 1:
                            p1.plot(np.arange(0,t1),rdata[TT][0:t1,1,pp], linewidth=2, color=(0.6,0.6,0.6), alpha=alpha)             
                            p1.plot(np.arange(t1-1,t2),rdata[TT][t1-1:t2,1,pp], linewidth=2,color=clr[LR], alpha=alpha)             
                            p1.plot(np.arange(t2-1,T),rdata[TT][(t2-1):T,1,pp], linewidth=2,color=clrchoice[LR], alpha=alpha)             
          p1.set_aspect('auto')
          if pc > 1:
              p1.text(1,YSCALE/2,'PC %d' % (pc-1),fontsize=15)
              p1.set_ylim([-YSCALE, YSCALE])
          elif pc == 1:
              p1.set_yticks(np.arange(-6,7,1))      
              p1.set_yticklabels(['-6','','','','','','0','','','','','','6'])     
              p1.set_ylim([-6, 6])
          elif pc == 0:
              if p.Z == 1:
                  p1.text(1,1.5/2,'Out',fontsize=15)
                  p1.set_ylim([-2.5, 3.5])
              else:
                #   p1.set_ylim([-2,6])            
                  p1.set_ylim([-0.01,1.05])      
                  p1.set_yticks(np.arange(0,1.1,.1))      
                  p1.set_yticklabels(['0','','','','','0.5','','','','','1'])      
                  p1.text(1,3,'Out',fontsize=15)
                  p1.plot([0,T],[p.Choice_thresh,p.Choice_thresh],'--',linewidth=2,color='k',alpha=0.4,zorder=0)
          if pc == MAXPCPLOT-1:
              p1.set_xticks(np.arange(0,T+5,5))      
              p1.set_xticklabels([])
              p1.set_xlabel('Time step',fontsize=12)
          else:
              p1.set_xticks(np.arange(0,T+5,5))      
              p1.set_xticklabels([])
        #   p1.set_xticks([])      
          p1.set_xlim([0,T-1])         
  plt.show()




def PCA_1D_plot(p, Data, TT_plot, YSCALE, title=''):

  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import matplotlib as mpl
  mpl.use("Qt5Agg") # set the backend
  mpl.rcParams.update({'font.size': 14})
  mpl.rcParams['font.sans-serif'] = "Helvetica"
  mpl.rcParams['font.family'] = "sans-serif"
  import matplotlib.patches as patch
  from mpl_toolkits.mplot3d import Axes3D
  import matplotlib.patches   as patch

  pdata = Data['PDATA']
  rdata = Data['RDATA']

  if p.N > 9:
      MAXPCPLOT = 9
  else:
      MAXPCPLOT = p.N+2
  Plot_Initial_Share = 1
  _,t1,t2,dur,jit      = (p.T,p.t1,p.t2,p.dur,np.asarray(p.jit))
  INDS,_,_                    = trial_parser(p)

  fig = plt.figure()
  plt.get_current_fig_manager().window.setGeometry(770,100,400,1600) 
  gs   = fig.add_gridspec(MAXPCPLOT,1)
  fig.suptitle( title, fontsize=17, weight='bold')  
  for TT in TT_plot:
      T = pdata[TT].shape[0]
      for pc in range(MAXPCPLOT):
          p1   = fig.add_subplot(gs[pc])
          if p.T > 1:
            if pc > 0:
                p1.plot([t1,t1],[-YSCALE,YSCALE],linewidth=3,color=(.86,.86,.86),zorder=0)
                p1.plot([t2,t2],[-YSCALE,YSCALE],linewidth=3,color=(.86,.86,.86),zorder=0)
            else:
                p1.plot([t1,t1],[0,1],linewidth=3,color=(.86,.86,.86),zorder=0)
                p1.plot([t2,t2],[0,1],linewidth=3,color=(.86,.86,.86),zorder=0)
                
          if TT == 2:
              if pc == 0:
                  p1.plot(rdata[TT][:,pc,0], linewidth=1.5,color=(.5,.5,.5),zorder=1)  # trajectory
              else:
                  p1.plot(pdata[TT][:,pc,0], linewidth=1.5,color=(.5,.5,.5),zorder=1)  # trajectory
              p1.set_aspect('auto')
              continue
          else:  # trajectory            
              for gg in [0,1,2]:     # Stim pair types
                  ii, _, mksize, _, _ = groupplot(gg,INDS)
                  for LR in [1,0]:
                      if ii[LR].size == 0:
                         continue
                      for pp in ii[LR]:
                        clr = embedcolor(pp,p.n_items,3)
                        clrchoice = embedcolor(pp,p.n_items,6)
                        if pc == 0:   # readout
                            if p.Z == 1:
                                p1.plot(rdata[TT][:,pc,pp], linewidth=1.5,color=clrchoice[LR])  # trajectory
                            elif p.Z == 2:
                                p1.plot(rdata[TT][:,0,pp], linewidth=1.5,color=clrchoice[LR], alpha=1)             
                                p1.plot(rdata[TT][:,1,pp], linewidth=1.5,color=clrchoice[LR], alpha=0.05)  
                            elif p.Z == 3:
                                p1.plot(np.arange(t1,t2),nm.Tanh_readout(p,rdata[TT][t1:t2,1,pp]), linewidth=1.5,color=clr[LR], alpha=1)             
                                p1.plot(np.arange(t2-1,T),nm.Tanh_readout(p,rdata[TT][(t2-1):,1,pp]), linewidth=1.5,color=clrchoice[LR], alpha=1)             
                        elif pc == 1:
                                p1.plot(np.arange(t1,t2),rdata[TT][t1:t2,1,pp], linewidth=1.5,color=clr[LR], alpha=0.5)     # choice neuron
                                p1.plot(np.arange(t2-1,T),rdata[TT][(t2-1):,1,pp], linewidth=1.5,color=clrchoice[LR], alpha=0.5)     # choice neuron
                                # p1.plot(rdata[TT][:,2,ii[LR]], linewidth=1.5,color=(.5,.5,0), alpha=0.5) # rest neuron
                        else:
                            p1.plot(np.arange(t2-1,T), pdata[TT][(t2-1):,pc-2,pp], linewidth=1.5,color=clrchoice[LR])  # trajectory
                            p1.plot(np.arange(t1,t2), pdata[TT][t1:t2,pc-2,pp], linewidth=1.5,color=clr[LR])  # trajectory
                            #   if Plot_Initial_Share == 1:
                        #       if p.T > 1:
                        #           p1.plot(pdata[TT][0:int(T),pc-1,INDS[4]], linewidth=1.5,color='black')
          p1.set_aspect('auto')
          if pc > 1:
              p1.text(1,YSCALE/2,'PC %d' % (pc-1),fontsize=15)
              p1.set_ylim([-YSCALE, YSCALE])
          elif pc == 1:
              p1.set_ylim([-5, 5])
          elif pc == 0:
              if p.Z == 1:
                  p1.text(1,1.5/2,'Out',fontsize=15)
                  p1.set_ylim([-2.5, 3.5])
              else:
                #   p1.set_ylim([-2,6])            
                  p1.set_ylim([-0.01,1.01])      
                  p1.set_yticks(np.arange(0,1.1,.1))      
                  p1.set_yticklabels(['0','','','','','0.5','','','','','1'])      
                  p1.text(1,3,'Out',fontsize=15)
                  p1.plot([0,T],[p.Choice_thresh,p.Choice_thresh],'--',linewidth=2,color='k',alpha=0.4,zorder=0)
          if pc == MAXPCPLOT-1:
              p1.set_xlabel('Time step',fontsize=12)
          else:
              p1.set_xticklabels([])
          p1.set_xlim([0,T-1])         
  plt.show()






#%% Unit PSTH ##########################################

def PSTH( p, Data, Datatag, X_input, TT = 0, tfactor = 1 ):
    
    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch    
    from scipy.stats import spearmanr

    COLOR_EMBED_1st = 3
    COLOR_EMBED_2nd = 2   # 1,2 (red, blue): 1st, 2nd // 3, 4 (green): 1st, 2nd 
    INDS, trains, probes     = trial_parser(p)

    T, t1, t2, _, _, _       = tk.parms_test(p,X_input,TT)    
    dt = p.dt / tfactor
    t1 = t1 * tfactor
    t2 = t2 * tfactor
    T  = T  * tfactor

    figg = plt.figure(constrained_layout=True,figsize=(3,10))
    
    plt.get_current_fig_manager().window.setGeometry(1100,100,1600,3000) 
    plotcount = 1
    
    for cnum in range(100):  #range(p.N): #[98,38,12]:  #range(p.N):   range(90,100)
        
        p1   = figg.add_subplot(10,10,plotcount)
        plotcount += 1
        minval, maxval = 0, 0

        plt.plot([t1-1,t1-1],[-10,10],linewidth=3,color=(.9,.9,.9))  # stim1
        plt.plot([t2-1,t2-1],[-10,10],linewidth=3,color=(.9,.9,.9))  # stim2

        rs = np.full( (p.n_items), np.nan )

        for gg in [0,1,2]:      # 0: Train, 1: Edges, 2: Probes, 3: Sames
            ii, clr, mksize, _, _ = groupplot(gg,INDS)
            for LR in [0,1]:          
                bclr = ('r','b')  
                for pp in ii[LR]: 
                    
                    inds        = Datatag[TT] == pp
                    onetrace    = Data[TT][inds,cnum]
                    
                    if COLOR_EMBED_1st > 0:
                        clr1 = embedcolor(pp,p.n_items,COLOR_EMBED_1st)
                    else:
                        clr1 = clr
                    if COLOR_EMBED_2nd > 0:
                        clr2 = embedcolor(pp,p.n_items,COLOR_EMBED_2nd)
                    else:
                        clr2 = clr
                    
                    rank1, _    = ranks(p,pp)
                    rs[rank1] = onetrace[t1]
                    
                    # the PSTH plot #
                    tvec1 = np.arange( 0,              t2*dt,       dt)
                    tvec2 = np.arange( t2*dt,      (T-1)*dt,    dt)

                    # p1.plot(tvec1,        onetrace[0:t2],   linewidth=1.5,color=clr1[LR]) # 1st part
                    # p1.plot(tvec2,        onetrace[t2:], linewidth=1.5,color=clr2[LR]) # 2nd part

                    p1.scatter(tvec1[t1],        onetrace[t1],   marker='.' , s=150, color=clr1[LR]) # 1st part

                    # p1.plot(range(0,t2),        onetrace[0:(t2)],   linewidth=1.5,color=clr1[LR]) # 1st part
                    # p1.plot(range(t2-1,T),      onetrace[(t2-1):T], linewidth=1.5,color=bclr[LR]) # 2nd part

                    # if maxval < np.max(onetrace):
                    #     maxval = np.max(onetrace)
                    # if minval > np.min(onetrace):
                    #     minval = np.min(onetrace)

                    if maxval < np.max(onetrace[t1]):
                        maxval = np.max(onetrace[t1])
                    if minval > np.min(onetrace[t1]):
                        minval = np.min(onetrace[t1])                        

        rho = spearmanr( rs, np.arange(p.n_items))[0]  # correlate with monotonic ranking

        p1.set_title('%0.2f' % (rho))
        p1.set_aspect('auto')

        if np.abs( rho ) == 1:
            p1.set_facecolor('grey')

        # p1.set_xlim([0,dt*(T-1)])
        p1.set_ylim([minval-0.05,maxval+0.05])

        p1.set_xlim([dt*(t1-1),dt*(t1+1)])
        # p1.set_ylim([-0.2,0.2])

        p1.set_ylabel('Cell %d' % (cnum))    
        if cnum == 9:
            # p1.set_xlabel('Time steps')
            p1.set_xlabel('Time')
            # p1.set_xlabel('Time steps')
        plt.tight_layout()
        # p1.set_xticks([])
        p1.set_yticks([0,1])
        p1.tick_params('both', length=5, width=1, which='major')
        

############################################################


def Unit_Rank_Tuning_Evolution( p, Data, TT ):
    
    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch    
    
    from scipy.stats import spearmanr

    INDS,trains,probes     = trial_parser(p)
    T, t1, t2, _, _, _     = tk.parms_test(p,None,TT)    
    
    data    = Data['DATA']

    traces  = data[TT]  # [T,N,B]

    # Get trial data for Ranks 1 through N #
    rank1vec,_  = Rankvector(p)
    data_rank1  = np.full( ( t2, p.N, p.n_items ), np.nan )   # [T,N,B]
    for r in range(p.n_items):
        matchtraces         = traces[:t2,:,rank1vec==r]    # all traces with this rank1 -- will only be equal up to t1
        data_rank1[:,:,r]   = matchtraces[:,:,0]          # since the same (if noiseless) through delay just take the first matched trace
        
    # (unit analysis) Calculate Spearman's rho #
    rho = np.full( (t2, p.N), np.nan )
    for nn in range(p.N):       # unfortunately slow
        for tt in range(t2):
            ranking    = data_rank1[tt,nn,:]        # observed ranking
            if np.unique(ranking).size == 1:
                rho[tt,nn] = 0
            else:
                # ranking    = np.argsort(ranking)
                rho[tt,nn] = spearmanr( ranking, np.arange(p.n_items))[0]  # correlate with monotonic ranking

    # Calculate Population correlation # (Population analysis) 
    rpop = np.full( (3, t2), np.nan )
    for Ref in [0,1,2]:
        if Ref == 0:
            tref = t1
        elif Ref == 1:
            tref = t1+int((t1+t2)/2)
        else:
            tref = t2-1
        for tt in range(t2):
            vals = np.full((p.n_items),np.nan)
            for c in range(p.n_items):
                vals[c]         = np.corrcoef( data_rank1[tref,:,c]  , data_rank1[tt,:,c] )[0,1] 
            rpop[Ref,tt]    = np.mean(vals)           


    # 1. Scatter Correlations #
    figg = plt.figure(constrained_layout=True,figsize=(3,10))
    plt.get_current_fig_manager().window.setGeometry(100,100,2200,300) 
    mid = int((t2+t1)/2)
    for comp in range(3):
        if comp == 0:       
            xt = t1
            yt = mid
            strx,stry = ('t1','mid')
        elif comp == 1:
            xt = mid
            yt = t2-1
            strx,stry = ('mid','end')
        elif comp == 2:
            xt = t1
            yt = t2-1
            strx,stry = ('t1','end')
        ax = figg.add_subplot(1,6,1+comp)
        ax.scatter(rho[xt,:],rho[yt,:],40,color='k',alpha=0.7)
        PAD = 0.05
        ax.plot([-1-PAD,1+PAD], [0,0], ':', linewidth=1, color='k')
        ax.plot([0,0], [-1-PAD,1+PAD], ':', linewidth=1, color='k')
        ax.set_xlim([-1-PAD,1+PAD])
        ax.set_ylim([-1-PAD,1+PAD])
        ax.set_xlabel('rho, %s' % (strx))
        ax.set_ylabel('rho, %s' % (stry))
        ax.set_aspect('equal', 'box')    


    # 2.  Spearman correlation, population pattern evolution #
    ax = figg.add_subplot(1,6,4)
    for Ref in [0,2]:
        if Ref == 0:
            tref = t1
            clr = 'blue'
        elif Ref == 1:
            tref = mid
            clr = 'red'
        elif Ref == 2:
            tref = t2-1
            clr = 'green'
        refrhos = rho[tref,:]               # reference-time rhos for all N units
        corrtrace  = np.full((t2),np.nan)
        for t in range(t2):
            corrtrace[t] = np.corrcoef( rho[t,:], refrhos )[0,1]

        ax.plot( np.arange(t2), corrtrace, color=clr, linewidth=3 )

        ax.plot( [tref,tref], [-1, 1], '--', color=clr, alpha=0.5, linewidth=2 )

        ax.set_xlabel('Time (dt)')
        ax.set_ylabel('Correlation')

    ax.plot( [0,t2], [0, 0], color='k', alpha=0.3, linewidth=2 )
    ax.set_xlim( [0,t2] )
    ax.set_ylim( [-1,1] )


    # 3.  Fully-ordered (correct rank) evolution #
    ax = figg.add_subplot(1,6,5)
    fraction = np.sum( np.abs(rho) == 1, axis=1 ) / p.N
    # fraction = np.sum( np.abs(rho) > 0.75, axis=1 ) / p.N
    ax.plot( np.arange(t2), fraction, color='k', linewidth=5 )
    ax.plot( [0,t2],    [0, 0],     color='k', alpha=0.3, linewidth=2 )
    maxval = 1
    ax.plot( [t1,t1],   [0, maxval], '--', color='b', alpha=0.5, linewidth=2 )
    ax.plot( [mid,mid], [0, maxval], '--', color='g', alpha=0.5, linewidth=2 )
    ax.plot( [t2,t2],   [0, maxval], '--', color='grey', alpha=0.5, linewidth=2 )
    ax.set_xlabel('Time (dt)')
    ax.set_ylabel('% fully ordered')
    ax.set_xlim( [0,t2+1] )    
    ax.set_ylim( [-0.01,maxval] )    

    # 4.  Average magnitude of Spearman correlation, evolution #
    ax = figg.add_subplot(1,6,6)
    fraction = np.mean( np.abs(rho), axis=1 )
    ax.plot( np.arange(t2), fraction, color='k', linewidth=5 )
    ax.plot( [0,t2],    [0, 0],     color='k', alpha=0.3, linewidth=2 )
    maxval = 1
    ax.plot( [t1,t1],   [0, maxval], '--', color='b', alpha=0.5, linewidth=2 )
    ax.plot( [mid,mid], [0, maxval], '--', color='g', alpha=0.5, linewidth=2 )
    ax.plot( [t2,t2],   [0, maxval], '--', color='grey', alpha=0.5, linewidth=2 )
    ax.set_xlabel('Time (dt)')
    ax.set_ylabel('Average rho')
    ax.set_xlim( [0,t2+1] )    
    ax.set_ylim( [-0.01,maxval] )    


    return ax



def Population_Correlation( p, Data, SUBTRACT_XCM,  TT ):
    
    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch    
    
    from scipy.stats import spearmanr
    from scipy.stats import kendalltau

    INDS,trains,probes     = trial_parser(p)
    T, t1, t2, _, _, _     = tk.parms_test(p,None,TT)    
    d1 = t2-t1

    data    = Data['DATA']

    traces  = data[TT]  # [T,N,B]

    # (optional) subtract XCM #
    if SUBTRACT_XCM:
        xcm_cont        = np.mean( traces, axis = -1)                                 # [T,N_reduced]
        for tt in range(T):
            for cc in range(traces.shape[-1]):
                traces[tt,:,cc] = traces[tt,:,cc] - xcm_cont[tt,:] 

    # Get trial data for Item 1 #
    M           = p.n_items
    rank1vec,_  = Rankvector(p)
    data_rank1  = np.full( ( t2, p.N, p.n_items ), np.nan )   # [T,N,B]
    for r in range(p.n_items):
        matchtraces         = traces[:t2,:,rank1vec==r]    # all traces with this rank1 -- will only be equal up to t1
        data_rank1[:,:,r]   = matchtraces[:,:,0]          # since the same (if noiseless) through delay just take the first matched trace
        
    # Calc within item-1 correlation # 
    rpop = np.full( (3, t2), np.nan )  # Time-reference
    for Ref in [0,1,2]:
        if Ref == 0:        # item 1 #
            tref = t1
        elif Ref == 1:      # middle of delay #
            tref = int((t2+t1)/2)
        else:               # item 2 #
            tref = t2-1
        for tt in range(t2):
            vals = np.full((p.n_items),np.nan)
            for item1 in range(p.n_items):
                vals[item1]     = kendalltau( data_rank1[tref,:,item1]  , data_rank1[tt,:,item1] ).correlation
            rpop[Ref,tt]        = np.mean(vals)           

    # Calc across-trial (by symbolic distance) correlation # 
    distinds        = trial_parser_symbolic_dist(p,flatten_indices=False)
    x = y   = np.arange(M)                  # x, y are index vectors for item 1 and 2, respectively
    xv, yv  = np.meshgrid(x, y, indexing='ij')             # 2D matrices  
    xv, yv  = (xv.flatten(), yv.flatten())
    rsd = np.full( (M-1,t2), np.nan )
    for sd in range(M-1):       # symbolic distance
        inds            = distinds[sd][1,:].flatten()  
        for tt in range(t2):    # timestep
            vals            = np.full((M-1-sd),np.nan) # no. trials of this symdist
            for iii in range(inds.size):    # pairs of items fulfilling this symbolic distance
                i1,i2         = ( xv[inds[iii]], yv[inds[iii]] )
                vals[iii]           = kendalltau( data_rank1[tt,:,i1]  , data_rank1[tt,:,i2] ).correlation
            rsd[sd,tt]            = np.mean(np.array(vals))           


    figg = plt.figure(constrained_layout=True,figsize=(3,10))
    plt.get_current_fig_manager().window.setGeometry(100,500,800,300) 

    #  Correlation (within condition) evolution #
    ax = figg.add_subplot(1,2,1)
    for Ref in [2,0]:
        if Ref == 0:        # item 1
            clr = 'green'
            alpha = 1
            tref = t1
        elif Ref == 1:      # middle
            clr = 'blue'
            alpha = 1
            tref = int((t2+t1)/2)
        elif Ref == 2:      # end of delay
            clr = 'green'
            alpha = 0.25
            tref = t2-1
        ax.plot( np.arange(d1)/(d1-1), rpop[Ref,t1:t2], color=clr, linewidth=3, alpha = alpha )
        # ax.plot( [tref,tref], [-1, 1], '--', color='k', alpha=0.5, linewidth=2 )
        ax.set_xlabel('Time (dt)')
        ax.set_ylabel('Correlation (Kendall)',fontsize=18)
    ax.plot( [0,1], [0, 0], color='k', alpha=0.3, linewidth=2 )
    ax.set_xticks(np.arange(0,1.25,.25))
    ax.set_xticklabels(['0','','','','1'])
    ax.set_yticklabels(['-1','','0','','1'])
    ax.set_xlim( [0,1] )
    ax.set_ylim( [-1,1] )

    # Correlation (across ranks) evolution #
    ax = figg.add_subplot(1,2,2)
    for sd in range(M-1):
        ax.plot( np.arange(d1)/(d1-1), rsd[sd,t1:t2], color=np.array([sd/(M-1),sd/(M-1),sd/(M-1)]), linewidth=3 )
        # ax.plot( [t1,t1], [-1, 1], '--', color='k', alpha=0.5, linewidth=2 )
        # ax.plot( [t2,t2], [-1, 1], '--', color='k', alpha=0.5, linewidth=2 )
        ax.set_xlabel('Time (dt)')
        ax.set_ylabel('Correlation (Kendall)',fontsize=18)
    ax.plot( [0,1], [0, 0], color='k', alpha=0.3, linewidth=2 )
    ax.set_xticks(np.arange(0,1.25,.25))
    ax.set_xticklabels(['0','','','','1'])
    ax.set_yticklabels(['-1','','0','','1'])
    ax.set_xlim( [0,1] )
    ax.set_ylim( [-1,1] )


    return ax


############################################################


# Plot Oscillatory Phases in Grid #######

def plot_phases_grid(p,data,train_table,INDS,X_input,TT):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    tind,eind,pind,sind,iind  = INDS       # 0: train, 1: edges, 2: probe, 3: sames, 4: initial
    n_items                   = p.n_items
    T, t1, t2, _, _, _          = tk.parms_test(p,X_input,TT)

    # Plot #
    fig0 = plt.figure()
    plt.show(block = False)    # prevents busy cursor
    plt.get_current_fig_manager().window.setGeometry(80,100,2400,500) 
    fig0.suptitle('Transitive Inference RNN: %d items\n%d units ' \
                % (n_items,p.N), fontsize=17, weight='bold')

    from mpl_toolkits.axes_grid1 import make_axes_locatable


    Valuematrices = []

    for xx in range(4): 
    
        if xx == 0:
            t = t1
            str = 'Item 1'
        elif xx == 1:
            t = t2-1
            str = 'Pre Item 2'
        elif xx == 2:
            t = t2
            str = 'Post Item 2'

        if xx in range(3):
            mat = np.ones((n_items,n_items))
            COORDINATE = 1  # 0: radius, 1: phase
            for gg in [0,1,2,3]:     # Stim pair types
                ii, _, _, _, _ = groupplot(gg,INDS)
                for LR in [0,1]:
                    for pp in ii[LR]: 
                        ppp = np.unravel_index(pp, (n_items,n_items), order='C')
                        mat[ppp] = data[TT][t,COORDINATE,pp]
            Valuematrices.append(mat)
        else:
            # print(xx)
            mat = Valuematrices[2] - Valuematrices[1]
            # correct for cycle
            mat[mat >  np.pi]  = mat[mat >  np.pi] - np.pi
            mat[mat < -np.pi]  = mat[mat < -np.pi] + np.pi

        ax6 = fig0.add_subplot(1,4,1+xx)
        pl6 = ax6.imshow(mat, vmin=-np.pi, vmax=np.pi, cmap='twilight_shifted')        # matrix
        plot_squares(train_table,1,ax6)  # squares
        # plot_squares(probes,2,ax6)    # circles
        plt.xticks(ticks=range(p.n_items),labels=[])
        plt.yticks(ticks=range(p.n_items),labels=[])
        plt.colorbar(pl6,ax=ax6)
        ax6.axis(xmin=-0.5,xmax=n_items-0.5,ymin=-0.5,ymax=n_items-0.5)
        ax6.invert_yaxis()
        ax6.set_title('%s' % str)

        fig0.subplots_adjust(hspace=.4)



# PCA 2D, multiple scales #
def PCA_2D_multiple(p, X_input, model, data, SCALES, groups_to_plot, TT, pcplist, pca, Fps):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    # manual settings #
    plot_fps        = 1
    plot_FP         = 1
    plot_FP_modes   = 1
    if Fps.Saddles.size > 0:
        FP_to_plot_mode = [Fps.Saddles[0]]
    else:
        FP_to_plot_mode = [0]

    # plot settings #
    MAXVAL      = 5
    INPSCALE    = 2
    COLOR_EMBED = 3

    # data to plot #
    idata   = nna.calc_input_vectors(p, model, X_input )
    ivec    = nna.project_data(idata, pca ,center=False,inverse=False)

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(1100,100,2000,1200) 

    _, t1, t2, _, _, _     = tk.parms_test(p, X_input, TT)

    for pcp in range(pcplist.shape[0]):       # PC Pair

        for SC in range(2):  # zoom scale

            Z = SCALES[SC]
            alph = 0.5

            ax = plt.subplot(3,3,3*SC+pcp+1)  

            pc_x = pcplist[pcp,0]
            pc_y = pcplist[pcp,1]

            if SC == 0:
                trajwidth=2.5
                markersize=150
                mksize2 = 100
                lwidth = 2
                SCALER = 1
            else:
                trajwidth=3.5   
                markersize=400
                mksize2 = 300
                lwidth = 4
                SCALER = 2

            INDS,_,_          = trial_parser(p)  # get important indices
            for gg in groups_to_plot:     #0: Train, 1: Edges, 2: Probes, 3: Sames
                ii, clr1, mksize, _, _ = groupplot(gg,INDS)
                for LR in [0,1]:

                    for pp in ii[LR]: 
                        if COLOR_EMBED > 0:
                            clr = embedcolor(pp,p.n_items,COLOR_EMBED)
                        else:
                            clr = clr0
                        
                        bclr = [tuple([1,0,0]),tuple([0,0,1])]    

                        # BLOCK 1 #
                        plt.scatter(    data[TT][t1,pc_x,pp],  data[TT][t1,pc_y,pp],  color=clr[LR],  marker='.', s=mksize2*3, alpha=alph)       # Input 1 (ball)
                        plt.plot(       data[TT][0:(t2),pc_x,pp],  data[TT][0:(t2),pc_y,pp],  linewidth=trajwidth, color='k',alpha=0.05)      # 1st traj  (line) 

                        # plt.scatter(    data[TT][t2-1,pc_x,pp],  data[TT][t2-1,pc_y,pp],  \
                                        # linewidth=lwidth,color=(.8,.8,.8),  marker='x', s=markersize/2,alpha=alph)      # up to inp2 (open circle)
                        # plt.scatter(    data[TT][t2-1,pc_x,pp],  data[TT][t2-1,pc_y,pp],  color=clr[LR],  marker='.', s=mksize2*3, alpha=alph)       # Input 1 (ball)


                        # # # # BLOCK 2 #
                        plt.scatter(data[TT][t2,pc_x,pp], data[TT][t2,pc_y,pp], color='g',  marker='*', s=markersize*.7,alpha=0.4)             # stim2 (star)
                        plt.plot(       data[TT][(t2-1):(t2+1),pc_x,pp],  data[TT][(t2-1):(t2+1),pc_y,pp],  linewidth=trajwidth, color='k',alpha=0.1)     # 1st traj  (line) 
                        
                        # # # # BLOCK 3 #
                        plt.plot(       data[TT][(t2):,pc_x,pp],  data[TT][(t2):,pc_y,pp],  linewidth=trajwidth, color='k', alpha=0.1)        # 2nd traj  (grey) 
                        plt.plot(       data[TT][(t2):,pc_x,pp],  data[TT][(t2):,pc_y,pp],  linewidth=trajwidth, color=clr1[LR], alpha=0.2)  # 2nd traj  (red/blue) 
                        plt.scatter(    data[TT][-1,pc_x,pp], data[TT][-1,pc_y,pp], color=clr1[LR],  marker='v', s=SCALER*mksize/2,alpha=.8)       # END   (triangle)

            ax.set_aspect('equal', 'box')
            ax.set_xlim([-MAXVAL*Z,MAXVAL*Z])
            ax.set_ylim([-MAXVAL*Z,MAXVAL*Z])
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('C%d' % pc_x, fontsize=20, fontweight='bold')
            ax.set_ylabel('C%d' % pc_y, fontsize=20, fontweight='bold')
            ax.scatter(0,0,color='k', marker='o',s=50)  # plot origin    
            
            if plot_FP:
                nna.plot_FP_2D( ax, Fps, pca , pc_x, pc_y)
            if plot_FP_modes:
                nna.plot_FP_modes_2D(ax, Fps, FP_to_plot_mode, pca, pc_x, pc_y)                        

            # plot FP candidates #
            # if plot_fps:
            #     qmin,qmax = (2,8)
            #     clrs = npf.colorbarmapper(-np.log10(fps.q),qmin,qmax,'jet')  # get colormap colors
            #     proj  = nna.project_data(data=fps.xstar, basis=pca, center=True)
            #     ax.scatter(proj[:, pc_x], proj[:, pc_y], zorder=20, s=25, marker='o', c=clrs.squeeze(), alpha=1)

        #  inputs  #
        ax = plt.subplot(3,3,6+pcp+1)  
        for ii in reversed(range(p.n_items)):
            prop    = .7*(ii+1)/(p.n_items)    # proportion of full color intensity                
            clr = (prop,.7,prop) 
            # ax.plot([0,ivec[ii,pc_x]],[0,dat[ii,pc_y]], color=clr, linewidth=1.5, marker='o',alpha=1)            
            ax.arrow(0,0,ivec[ii,pc_x],ivec[ii,pc_y],width=0.01,head_width=0.04,color=clr)
            # ax.plot([0,ivec[ii,pc_x]],[0,dat[ii,pc_y]], color=[.3,.3,.3], linewidth=1.5, marker='o',alpha=1)
        ax.scatter(0,0,color='k', marker='o',s=40)  # plot origin
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])        
        ax.set_aspect('equal', 'box')
        ax.set_xlim([-INPSCALE,INPSCALE])
        ax.set_ylim([-INPSCALE,INPSCALE])
        ax.set_xlabel('C%d' % pc_x, fontsize=20, fontweight='bold')
        ax.set_ylabel('C%d' % pc_y, fontsize=20, fontweight='bold')

    fig.subplots_adjust(hspace=1)
    plt.tight_layout()

    return ax






def PCA_2D(p, Data, SCALE, groups_to_plot, TT, pcplist, Fps=None):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    plot_FP_modes_2D    = 0
    if Fps != None:
        if Fps.Saddles.size > 0:
            FP_mode_toplot = [Fps.Saddles[0]]
        else:
            FP_mode_toplot = [0]

    data = Data['PDATA']
    pca =  Data['pca']

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(50,100,1800,800) 

    INDS,_,_          = trial_parser(p)
    _, t1, t2, _, _, _     = tk.parms_test( p, None, TT )

    for pcp in range(pcplist.shape[0]):       # PC Pair
        
        ax      = plt.subplot(1,3,pcp+1)  
        c1    = pcplist[pcp,0]
        c2    = pcplist[pcp,1]
        dum     = np.array([0,1,2])
        pc_0    = dum[~np.isin([0,1,2],[c1,c2])]

        print( 'plotting Vector Field in 2D...' )

        # b. Trial trajectories #
        for gg in groups_to_plot:     # Stim pair types
            ii, clr, mksize, _, _ = groupplot(gg,INDS)
            for LR in [0,1]:
                for pp in ii[LR]:
                    clr = embedcolor(pp, p.n_items, 3)                            
                    clrchoice = embedcolor(pp,p.n_items,6)
                    plt.plot(       data[TT][:,c1,pp],  data[TT][:,c2,pp],  color=[.8,.8,.8], linewidth=3, alpha=0.05)                          # traj  (line) 
                    # plt.scatter(    data[TT][0,c1,pp],  data[TT][0,c2,pp],  color=clr[LR],  marker='.', s=mksize)   # start (ball)
                    plt.scatter(    data[TT][t1,c1,pp],  data[TT][t1,c2,pp],  color=clr[LR],  marker='.', s=mksize*2)   # start (ball)
                    plt.scatter(    data[TT][-1,c1,pp], data[TT][-1,c2,pp], color=clrchoice[LR],  marker='v', s=50,zorder=999)       # end   (triangle)
                    plt.scatter(data[TT][t2-1,c1,pp], data[TT][t2-1,c2,pp],   color=clr[LR],  marker='*', s=150, alpha=0.4,zorder=1000)                 # stim2 (star)
        
        ###### FP plot ########
        if Fps != None:
            nna.plot_FP_2D(ax,Fps,pca,c1,c2)

        if plot_FP_modes_2D:
            nna.plot_FP_modes_2D(ax, Fps, FP_mode_toplot, pca, c1, c2)

        # ax.scatter(0,0,color='k', marker='o',s=100,alpha=0.2)  # plot origin
        ax.set_aspect('equal', 'box')
        ax.set_xlim([-SCALE,SCALE])
        ax.set_ylim([-SCALE,SCALE])
        ax.set_xlabel('PC%d' % c1, fontsize=14, fontweight='bold')
        ax.set_ylabel('PC%d' % c2, fontsize=14, fontweight='bold')

        fig.subplots_adjust(hspace=1)
        plt.tight_layout()


def PCA_2D_task_axes(p, Data, SCALE, groups_to_plot, TT, pcplist, Fps=None):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    plot_FP_modes_2D    = 0
    if Fps != None:
        if Fps.Saddles.size > 0:
            FP_mode_toplot = [Fps.Saddles[0]]
        else:
            FP_mode_toplot = [0]

    data = Data['TDATA']
    pca =  Data['taskaxes']

    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(50,100,1800,800) 

    INDS,_,_          = trial_parser(p)
    _, t1, t2, _, _, _     = tk.parms_test( p, None, TT )

    for pcp in range(pcplist.shape[0]):       # PC Pair
        
        ax      = plt.subplot(1,3,pcp+1)  
        c1    = pcplist[pcp,0]
        c2    = pcplist[pcp,1]
        dum     = np.array([0,1,2])
        pc_0    = dum[~np.isin([0,1,2],[c1,c2])]

        print( 'plotting Vector Field in 2D...' )

        # b. Trial trajectories #
        for gg in groups_to_plot:     # Stim pair types
            ii, clr, mksize, _, _ = groupplot(gg,INDS)
            for LR in [0,1]:
                for pp in ii[LR]:
                    clr = embedcolor(pp, p.n_items, 3)                            
                    clrchoice = embedcolor(pp,p.n_items,6)
                    plt.plot(       data[TT][:,c1,pp],  data[TT][:,c2,pp],  color=[.8,.8,.8], linewidth=3, alpha=0.05)                          # traj  (line) 
                    # plt.scatter(    data[TT][0,c1,pp],  data[TT][0,c2,pp],  color=clr[LR],  marker='.', s=mksize)   # start (ball)
                    plt.scatter(    data[TT][t1,c1,pp],  data[TT][t1,c2,pp],  color=clr[LR],  marker='.', s=mksize*2)   # start (ball)
                    plt.scatter(    data[TT][-1,c1,pp], data[TT][-1,c2,pp], color=clrchoice[LR],  marker='v', s=50,zorder=999)       # end   (triangle)
                    plt.scatter(data[TT][t2-1,c1,pp], data[TT][t2-1,c2,pp],   color=clr[LR],  marker='*', s=150, alpha=0.4,zorder=1000)                 # stim2 (star)
        
        ###### FP plot ########
        if Fps != None:
            nna.plot_FP_2D(ax,Fps,pca,c1,c2)

        if plot_FP_modes_2D:
            nna.plot_FP_modes_2D(ax, Fps, FP_mode_toplot, pca, c1, c2)

        # ax.scatter(0,0,color='k', marker='o',s=100,alpha=0.2)  # plot origin
        ax.set_aspect('equal', 'box')
        ax.set_xlim([-SCALE,SCALE])
        ax.set_ylim([-SCALE,SCALE])
        ax.set_xlabel('PC%d' % c1, fontsize=14, fontweight='bold')
        ax.set_ylabel('PC%d' % c2, fontsize=14, fontweight='bold')

        fig.subplots_adjust(hspace=1)
        plt.tight_layout()

# PC 3D #
def PCA_3D(p, Data, AXSCALE, groups_toplot, TT, pcs, COLOR_EMBED_CHOICE = 6):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    pdata = Data['PDATA']

    TT_plot         = TT
    mksize, lw      = (300, 1.78)
    if p.Task_version < 2:
        COLOR_EMBED     = 3         # 1,2 (keeps red+blue): 1st, 2nd //  # 3, 4 (green):  1st, 2nd //  # 5: symbolic dist 
    else:
        COLOR_EMBED     = -1   # circular colormap

    INDS,_,_          = trial_parser(p)  # get important indices

    for A in range(len(AXSCALE)):
        print('(PC 3D) Plotting Axis Scale %d' % (AXSCALE[A]))
        fig     = plt.figure(constrained_layout=True,figsize=(7,7))
        plt.get_current_fig_manager().window.setGeometry(100,100+700*A,1000,1000) 
        ax      = plt.axes(projection="3d")
        for TT in [TT_plot]:
            _, t1, t2, _, _, _     = tk.parms_test(p,None,TT)
            if TT == 2 or TT == 3:            
                ax.plot3D(pdata[TT][:,0,0], pdata[TT][:,1,0], pdata[TT][:,2,0], color='k', linewidth=lw*2, alpha=0.3)       # Traj 1
            else:
                for gg in groups_toplot:    
                    ii, clr0, mksize, lw, mk = groupplot(gg,INDS)    
                    for LR in [1,0]:
                        for pp in ii[LR]:
                            clr = embedcolor(pp, p.n_items, COLOR_EMBED)
                            
                            clrchoice = embedcolor(pp,p.n_items,COLOR_EMBED_CHOICE)
                            bclr = [tuple([1,0,0]),tuple([0,0,1])]     

                            # BLOCK 1 #
                            # ax.scatter(pdata[TT][0,pcs[0],pp], pdata[TT][0,pcs[1],pp], pdata[TT][0,pcs[2],pp], color='k', marker='.',s=mksize/2,alpha=0.08)                  #  Very start #              
                            ax.scatter(pdata[TT][t1,pcs[0],pp], pdata[TT][t1,pcs[1],pp], pdata[TT][t1,pcs[2],pp],color=clr[LR], s=250, alpha=1, marker='o',edgecolors='k',linewidth=2)               # Input 1 #                    
                            ax.plot3D(pdata[TT][t1-1:(t2),pcs[0],pp], pdata[TT][t1-1:(t2),pcs[1],pp], pdata[TT][t1-1:(t2),pcs[2],pp], color='k', linewidth=3, alpha=.05)       # Traj 1

                            # ax.scatter(pdata[TT][t2,pcs[0],pp], pdata[TT][t2,pcs[1],pp], pdata[TT][t2,pcs[2],pp], color='pink', marker='.', s=mksize/2, alpha=0.8)       # Traj 1

                            if t2 <= pdata[TT].shape[0]:

                                ax.scatter(pdata[TT][t2-1,pcs[0],pp], pdata[TT][t2-1,pcs[1],pp], pdata[TT][t2-1,pcs[2],pp], color=clr[LR], linewidth=2, s=600, alpha=1, marker='*', edgecolors='k')  # star   
                                # ax.scatter(pdata[TT][t2-1,pcs[0],pp], pdata[TT][t2-1,pcs[1],pp], pdata[TT][t2-1,pcs[2],pp], color=clr[LR], linewidth=2, s=350, alpha=1, marker='o', edgecolors='k')  # star   

                                # BLOCK 2 #
                                # ax.plot3D(pdata[TT][(t2-1):(t2+1),pcs[0],pp], pdata[TT][(t2-1):(t2+1),pcs[1],pp], pdata[TT][(t2-1):(t2+1),pcs[2],pp], color=bclr[LR], linewidth=lw*2,alpha=0.2)   # Traj 2 (red/blue)
                                # ax.scatter(pdata[TT][t2,pcs[0],pp], pdata[TT][t2,pcs[1],pp], pdata[TT][t2,pcs[2],pp], color=clrchoice[LR], marker='*',s=150,alpha=0.8)   # Input 2
                                # ax.scatter(pdata[TT][t2+1,pcs[0],pp], pdata[TT][t2+1,pcs[1],pp], pdata[TT][t2+1,pcs[2],pp], color=clrchoice[LR], marker='*',s=150,alpha=0.8)   # Input 2

                                # BLOCK 3 #
                                ax.plot3D(pdata[TT][(t2-1):,pcs[0],pp], pdata[TT][(t2-1):,pcs[1],pp], pdata[TT][(t2-1):,pcs[2],pp], color='k', linewidth=3,alpha=0.1)      # Traj 2 (grey)
                                # ax.plot3D(pdata[TT][(t2-1):,pcs[0],pp], pdata[TT][(t2-1):,pcs[1],pp], pdata[TT][(t2-1):,pcs[2],pp], color=bclr[LR], linewidth=lw*2,alpha=0.3)   # Traj 2 (red/blue)
                                # tplus = 5
                                # ax.scatter(pdata[TT][t2+tplus,pcs[0],pp], pdata[TT][t2+tplus,pcs[1],pp], pdata[TT][t2+tplus,pcs[2],pp], color=clrchoice[LR], marker='v',s=mksize/3, alpha=0.75)       # End
                                if mk == 'D':
                                    ax.scatter(pdata[TT][-1,pcs[0],pp], pdata[TT][-1,pcs[1],pp], pdata[TT][-1,pcs[2],pp], edgecolors=None, facecolors=clrchoice[LR], marker='D', linewidth=None, s=mksize*1, alpha=1)       # End
                                else:
                                    ax.scatter(pdata[TT][-1,pcs[0],pp], pdata[TT][-1,pcs[1],pp], pdata[TT][-1,pcs[2],pp], edgecolors=None, facecolors=clrchoice[LR], linewidth=None, marker='v', s=mksize*1, alpha=1)       # End

                                # CCM #
                                if 1:
                                    if pp == 1:
                                        m1 = np.mean( pdata[TT][:,pcs[0],:], axis=1 )
                                        m2 = np.mean( pdata[TT][:,pcs[1],:], axis=1 )
                                        m3 = np.mean( pdata[TT][:,pcs[2],:], axis=1 )
                                        ax.plot3D(m1, m2, m3, color='#FFDE17', linewidth=lw*2.5,alpha=.45)      # Traj 2 (grey)

                            else:
                                print('re-setting t2 since trial is shorter')
                                newt2 = pdata[TT].shape[0]-1
                                ax.scatter(pdata[TT][newt2,pcs[0],pp], pdata[TT][newt2,pcs[1],pp], pdata[TT][newt2,pcs[2],pp], color=clr[LR], s=210, alpha=1,marker='*')  # pre    

                            # Time ticks #
                #            ax.scatter(pdata[TT][:,pcs[0],pp], pdata[TT][:,pcs[1],pp], pdata[TT][:,pcs[2],pp], color=clr[LR], marker='.',s=25)
                            # Same IC plot #
                #            if np.isin(pp,iind):
                #                if STIM_TYPE == 1:
                #                    ax.scatter(pdata[TT][0,0,pp], pdata[TT][0,1,pp], pdata[TT][0,2,pp], color='purple', marker='.',s=mksize)
                #                    ax.scatter(pdata[TT][t2-1,0,pp], pdata[TT][t2-1,1,pp], pdata[TT][t2-1,2,pp], color='purple', marker='.',s=150)
                #                    ax.plot3D(pdata[TT][0:t2,0,pp], pdata[TT][0:t2,1,pp], pdata[TT][0:t2,2,pp], color='purple', linewidth=lw)    
        
        # plot readout plane #
        # ax.scatter(0,0,0, color='k', marker='o',s=50)  # origin
        # if S != 1:
        #     plot_decisionplane(ax,ro_pc,ro_bias)    

        ax.set_xlabel('PC%d' % (pcs[0]),fontsize=20,fontweight='bold')
        ax.set_ylabel('PC%d' % (pcs[1]),fontsize=20,fontweight='bold')
        ax.set_zlabel('PC%d' % (pcs[2]),fontsize=20,fontweight='bold')
        ax.grid(False)
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
        ax.set_xlim3d(-AXSCALE[A],AXSCALE[A])
        ax.set_ylim3d(-AXSCALE[A],AXSCALE[A])
        ax.set_zlim3d(-AXSCALE[A],AXSCALE[A])    
        ax._axis3don = False

        og = (-10,5,-5)
        blength = 2
        ax.plot3D([og[0],og[0]+blength],[og[1],og[1]],[og[2],og[2]],color='k',linewidth=2)
        ax.plot3D([og[0],og[0]],[og[1],og[1]-blength],[og[2],og[2]],color='k',linewidth=2)
        ax.plot3D([og[0],og[0]],[og[1],og[1]],[og[2],og[2]+blength],color='k',linewidth=2)


    plt.draw()     
    plt.show()
    plt.pause(0.0001)

    plt.ion()
    plt.ioff()
    plt.ion()

    
    return ax



def Plot_Oscillatory_2D(p, Data, Data_lz, scales, groups_to_plot, TT, model_lin, Fps, Sys_to_plot=[0,1,2,3], Format=0):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

        # manual FP #
    plot_FP             = 1
    plot_FP_modes_2D    = 0
    # plot_model_readout  = 1
    # if Fps.Saddles.size > 0:
    #     FP_mode_toplot      = [Fps.Saddles[0]]       # 3: saddle FP
    # else:
    #     FP_mode_toplot  = []

        # plot #
    COLOR_EMBED_Choice      = 6   #  0: choice, 6: symdist, 8: end-order
    COLOR_EMBED_1st         = 3     # rank of 1st item
    MAXVAL              = 10
    numgrid             = 25   #   number of grid spacings for flow field
    numSys = len(Sys_to_plot)

    fig = plt.figure( constrained_layout=True, figsize=(14,5) )
    offset = np.random.randint(500)
    plt.get_current_fig_manager().window.setGeometry(50,100+offset,1200,600*numSys) 
    plt.suptitle('Oscillatory dynamics', fontsize=24, fontweight='bold')
    _, t1, t2, _, _, _     = tk.parms_test( p, None, TT )    

    for S in range(numSys):  #[0,1,2,3]:

        Sys = Sys_to_plot[S]

        if Sys == 0:
            if 'DATA_lz' in Data.keys(): 
                title = 'Non-linear data\nfp basis'
                data2    = Data['DATA_lz']           # non-linear system data, lz osci basis
                oscbasis = Data['basis_lz']
            else:
                continue
        
        elif Sys == 1:
            if Data_lz is not None and 'DATA_lz' in Data.keys(): 
                title = 'Linearized data\nfp basis'
                data2    = Data_lz['DATA_lz']           # Linearized data, FP basis
                oscbasis = Data_lz['basis_lz']
            else:
                continue

        elif Sys == 2:
            title = 'Non-linear data\nemp basis'
            data2    = Data['DATA_emp']           # Nonlinear data on Empirical basis
            oscbasis = Data['basis_emp']            

        elif Sys == 3:
            if Data_lz is not None: 
                title = 'Linearized data\nemp basis'
                data2    = Data_lz['DATA_emp']           # Nonlinear data on Empirical basis
                oscbasis = Data_lz['basis_emp']            
            else:
                continue

        for s in range(len(scales)):       # zooms

            # figure basics #
            ax          = plt.subplot(numSys,3,3*S+s+1)  
            gmax        = scales[s]*MAXVAL
            gmin        = -scales[s]*MAXVAL
            grid        = (numgrid,gmin,gmax)

            # if Sys == 1 and Format != 3:
                # nna.plot_flowfield( ax, grid, p, oscbasis, model_lin ,)  ###############################

            # plot flow field #
            # arrowsize = 20
            
            # plot flow test #
            # nna.plot_flowtest( ax, data2[-1], p, oscbasis, model_lin , p.t1 , arrowsize=0.01 ) ##################################

            # Trial trajectories #
            INDS,_,_          = trial_parser(p)  # get important indices
            for gg in groups_to_plot:     # Stim pair types

                ii, clr0, mksize, _, mk = groupplot(gg,INDS)
                
                for LR in [0,1]:

                    for pp in ii[LR]:

                        rank1,rank2 = ranks(p,pp)

                        clr1st      = embedcolor(pp, p.n_items, COLOR_EMBED_1st)
                        clrchoice   = embedcolor(pp, p.n_items, COLOR_EMBED_Choice)

                        # trajectory #
                        if Format == 0:  # LSQ, full trial

                            plt.plot(       data2[TT][:,0,pp],  data2[TT][:,1,pp],  color=(.7,.7,.7), linewidth=3, alpha=0.2, zorder=10)                          # traj  (line) 
                            plt.scatter(    data2[TT][-1,0,pp], data2[TT][-1,1,pp], color=clrchoice[0],  marker=mk, s=mksize/2, zorder=5 , alpha=1)                     # end     (triangle)                
                            plt.scatter(    data2[TT][t1,0,pp],  data2[TT][t1,1,pp],  facecolors=clr1st[LR],  edgecolors='k', linewidth=1.5, marker='.', s=650, zorder=50)      # **** pre-i2  (green)
                            plt.scatter(    data2[TT][t2-1,0,pp],  data2[TT][t2-1,1,pp],  facecolors=clr1st[LR],  edgecolors='k', linewidth=1.5, marker='*', s=300, zorder=20)      # **** pre-i2  (green)

                        elif Format == 1: # FPL, early trial                            plt.scatter(    data2[TT][t1,0,pp],    data2[TT][t1,1,pp],   marker='o', s=200, facecolors=clr1st[LR], linewidth=1.5, edgecolors='k', zorder=300)       # ****     (green)
                            plt.plot(       data2[TT][:t2,0,pp],  data2[TT][:t2,1,pp],  color=(.7,.7,.7), linewidth=3, alpha=0.2)                          # traj  (line) 
                            # plt.scatter(    data2[TT][t2,0,pp], data2[TT][t2,1,pp], color=clrchoice[LR],  marker=mk, s=100, zorder=0, alpha=1)                     # end     (triangle)                
                            plt.scatter(    data2[TT][t1,0,pp],  data2[TT][t1,1,pp],  facecolors=clr1st[LR],  edgecolors='k', linewidth=1.5, marker='.', s=650, zorder=50)      # **** pre-i2  (green)
                            plt.scatter(    data2[TT][t2-1,0,pp],  data2[TT][t2-1,1,pp],  facecolors=clr1st[LR],  edgecolors='k', linewidth=1.5, marker='*', s=300, zorder=30)      # **** pre-i2  (green)
                            # bclr = ('r','b')  
                            # clrchoice = bclr[LR]
                            # plt.scatter(    data2[TT][t2,0,pp], data2[TT][t2,1,pp], color=bclr[LR],  marker='v', s=150,zorder=4000, alpha=1)                     # end     (triangle)                
                        
                        elif Format == 2:   # FPL, item 2

                            if 0:
                                plt.plot(       data2[TT][:t2,0,pp],  data2[TT][:t2,1,pp],  color=(.7,.7,.7), linewidth=3, alpha=0.08)                          # traj  (line) 
                                plt.plot(       data2[TT][(t2-1):t2+1,0,pp],  data2[TT][(t2-1):t2+1,1,pp],  color=(.7,.7,.7), linewidth=3, alpha=0.2)                          # traj  (line) 
                                plt.scatter(    data2[TT][t2,0,pp], data2[TT][t2,1,pp], color=clrchoice[LR],  marker=mk, s=100, zorder=-200, alpha=1)                     # end     (triangle)                
                                plt.scatter(    data2[TT][t2-1,0,pp],  data2[TT][t2-1,1,pp],  facecolors=clr1st[LR],  edgecolors='k', alpha=.2, linewidth=1.5, marker='*', s=300, zorder=-100)      # **** pre-i2  (green)
                            elif 1:
                                dat              = np.copy( data2[TT][:,:,pp]  )     
                                qq               = np.reshape(np.arange(49),(7,7)).T.flatten()[pp]               
                                ff               = np.copy( data2[TT][t1,:,qq] - data2[TT][t1-1,:,qq] )
                                dat[t2,:]        = dat[t2-1,:] + ff   
                                plt.plot(       dat[:t2,0],  dat[:t2,1],  color=(.7,.7,.7), linewidth=3, alpha=0.08)                          # traj  (line) 
                                plt.plot(       dat[(t2-1):(t2+1),0],  dat[(t2-1):(t2+1),1],  color=(.7,.7,.7), linewidth=3, alpha=0.2)                          # traj  (line) 
                                plt.scatter(    dat[t2,0], dat[t2,1], color=clrchoice[LR],  marker=mk, s=100, zorder=-200, alpha=1)                     # end     (triangle)                
                                plt.scatter(    dat[t2-1,0],  dat[t2-1,1],  facecolors=clr1st[LR],  edgecolors='k', alpha=.2, linewidth=1.5, marker='*', s=300, zorder=-100)      # **** pre-i2  (green)

                        elif Format == 3:   # FPL,  outputs only
                            plt.scatter(    data2[TT][t2,0,pp], data2[TT][t2,1,pp], color=clrchoice[LR],  marker=mk, s=120, zorder=-90, alpha=1)                     # end     (triangle)                

                        elif Format == 4:   # FPL, quarter of choice period
                            # plt.scatter(    data2[TT][t1,0,pp],  data2[TT][t1,1,pp],  facecolors=clr1st[LR],  edgecolors='k', linewidth=1.5, marker='.', s=650, zorder=50)      # **** pre-i2  (green)
                            plt.scatter(    data2[TT][t2-1,0,pp],  data2[TT][t2-1,1,pp],  facecolors='None',  edgecolors='k', alpha=.08, linewidth=1.5, marker='*', s=300, zorder=2000)      # **** pre-i2  (green)
                            plt.plot(       data2[TT][t2:(t2+6),0,pp],  data2[TT][t2:(t2+6),1,pp],  color=(.7,.7,.7), linewidth=3, alpha=0.18)                          # traj  (line) 
                            plt.plot(       data2[TT][(t2-1):t2+1,0,pp],  data2[TT][(t2-1):t2+1,1,pp],  color=(.7,.7,.7), linewidth=3, alpha=0.2)                          # traj  (line) 
                            plt.scatter(    data2[TT][(t2+5),0,pp], data2[TT][(t2+5),1,pp], color=clrchoice[LR],  marker=mk, s=100, zorder=1000, alpha=1)                     # end     (triangle)                


                        else:  # until right after 2nd stim
                            # plt.plot(       data2[TT][:,0,pp],  data2[TT][:,1,pp],  color=(.7,.7,.7), linewidth=2, alpha=0.3)                          # traj  (line) 
                            # plt.scatter(    data2[TT][:,0,pp],  data2[TT][:,1,pp],  color=(.7,.7,.7), alpha=0.3)                          # traj  (line) 
                            # plt.plot(       data2[TT][:(t2+1),0,pp],  data2[TT][:(t2+1),1,pp],  color=(.7,.7,.7), linewidth=2, alpha=0.3)                          # traj  (line) 
                            # plt.scatter(    data2[TT][:(t2+1),0,pp],  data2[TT][:(t2+1),1,pp],  color=(.7,.7,.7), alpha=0.3)                          # traj  (line)                        
                            plt.scatter(    data2[TT][-1,0,pp], data2[TT][-1,1,pp], color=clrchoice[0],  marker='v', s=50*2,zorder=10000000)                     # end     (triangle)                
          
                        # clr0 = [tuple([1,0,0]),tuple([0,0,1])]

                        # early events   
                        # plt.scatter(    data2[TT][0,0,pp],    data2[TT][0,1,pp],    color='k',  marker='.', s=mksize, alpha=0.3, zorder=100)       # i1      (green)
                        # plt.scatter(    data2[TT][t1-1,0,pp],    data2[TT][t1-1,1,pp],    color='m',  marker='.', s=mksize*2, zorder=100)       # i1      (green)
                        # plt.scatter(    data2[TT][t1+1,0,pp],  data2[TT][t1+1,1,pp],  color='k',  marker='.', s=mksize,zorder=100)       # i1      (green)
                        
                        # text of rank #
                        # if s == 2:  # Rank (text)
                            # plot_Rank1( p, ax, data2[TT][t1,0,pp], data2[TT][t1,1,pp], pp , zorder=500, fontsize=20)

                        # mid events
                        # plt.scatter(    data2[TT][t2,0,pp], data2[TT][t2,1,pp], color=clrchoice[0],  marker='o', s=50, alpha=0.7, zorder=200)       # post-i2 (blue/red)                
                        # plt.scatter(    data2[TT][t2,0,pp], data2[TT][t2,1,pp], color=bclr[LR],  marker='o', s=50, alpha=0.7, zorder=200)       # post-i2 (blue/red)                
                        # plt.scatter(    data2[TT][t2,0,pp], data2[TT][t2,1,pp], color='pink',  marker='o', s=50, alpha=0.1)                     # post-i2 (pink) 
                        # if s == 1:  # Rank (text)
                            # plot_Rank1( p, ax, data2[TT][t2-1,0,pp], data2[TT][t2-1,1,pp], pp )


            # plt.scatter(0,0,s=40,)                        
            # ###### FP plot ########
            # if plot_FP and Fps is not None:
            #     nna.plot_FP_2D(ax, Fps, oscbasis, 0, 1)

            # if plot_FP_modes_2D:
            #     nna.plot_FP_modes_2D(ax, Fps, FP_mode_toplot, oscbasis, 0, 1)

            # if plot_model_readout:
            #     nna.plot_linear_readout_2D(ax, Fps, oscbasis)

            ax.set_aspect('equal', 'box')
            ax.set_xlim([gmin,gmax])
            ax.set_ylim([gmin,gmax])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            # ax.set_xlabel('Oscillatory axis, X', fontsize=18, fontweight='bold')
            # ax.set_ylabel('Oscillatory axis, Y', fontsize=18, fontweight='bold')
            if s == 0:
                ax.set_ylabel('%s' % (title), fontsize=22, rotation='horizontal')
                ax.yaxis.set_label_coords(-0.6, 0.4)            

    # fig.subplots_adjust(hspace=0.5)
    # plt.tight_layout()






def Calculate_G(p, Data, PCA_top_dims=-1, show_plot=False ):

    ii          = Rank1_trials(p)

    # time indices #
    _, t1, t2, _, _, _     = tk.parms_test( p, None, TT = 0 )
    q1     = int( t1 + 0.25 * (t2-t1))
    q2     = int( t1 + 0.5  * (t2-t1))
    q3     = int( t1 + 0.75 * (t2-t1))

    numMeasures = 6
    numWindows  = 6  #
    Pairs       = list(combinations(range(p.n_items), 2))
    numPairs    = len(Pairs)  #  number of trials in upper triangle

    ### initialize outputs ###
    G           = {}   # dictionary
    G['Pairs']              = Pairs    # [numPairs] list of Pairs
    G['Pairs_symdists']     = Calc_pairs_symdists(Pairs)

    # trace of values #
    Mean_trace   = np.full((2,numMeasures,t2),np.nan)      # [<data,rand>,measure,time]
    Pair_trace   = np.full((2,numMeasures,t2,numPairs),np.nan) # [<data,rand>,measure,time,pair]
    # window mean values (in windowed periods of the delay)
    Mean_win     = np.full((2,numMeasures,numWindows),np.nan)            # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]
    Pair_win     = np.full((2,numMeasures,numWindows,numPairs),np.nan)   # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]
    # dynamic value (late delay - early delay)
    Mean_dyn     = np.full((2,numMeasures,2),np.nan)            # [<data,rand>,index,exact/average time]
    Pair_dyn     = np.full((2,numMeasures,numPairs,2),np.nan)   # [<data,rand>,index,exact/average time]

    ### 1. Random data vs. real data ###################
    if Data['DATA'][0].shape[-1] == p.n_items:
        data_real       = Data['DATA'][0]        # [T,N,B]     data  
    else:
        data_real       = Data['DATA'][0][:,      :, ii]         # [t,N,rank1]     data             
    data_rand       = np.random.randn(data_real.shape[0],data_real.shape[1],data_real.shape[2])
    dat             = np.concatenate( (data_real[np.newaxis,:],data_rand[np.newaxis,:]), axis=0 )

    ### 2. (optional) Transform data to top PCs  ########
    #       all subsequent analysis is in PC space
    if PCA_top_dims > 0:    
        dat_filt = dat.copy()
        dat      = np.full((2,dat.shape[1],PCA_top_dims,dat.shape[3]),np.nan)
        if PCA_top_dims > 3:        # filters for top PCs
            for DT in [0,1]:
                dfilt, _         = nna.PCA_top_dims( dat_filt[DT,:,:,:], PCA_top_dims )    
                dat[DT,:,:,:]    = dfilt[:,np.arange(PCA_top_dims),:]   
        elif PCA_top_dims <= 3:     # assumes already in PCA format, simply takes top dimensions
            for DT in [0,1]:
                dfilt            = dat_filt[DT,:,:PCA_top_dims,:]    
                dat[DT,:,:,:]    = dfilt   

    # (both data 'types': real and random)
    for DT in [0,1]:

        ### Calculate cross-condition mean (CCM) ###
        CCM      = np.mean( dat[DT,:,:,:], axis = -1)               

        ### 3. Calculate each geometric value ####
        for mm in range(2):

            if mm == 0:      # Cosine similarity, all pairs
                str         = 'Cosine similarity\nND'
                ylims       = [-1.05, 1.05]
                numCond     = numPairs

            elif mm == 1:    # Euclidean distance, all pairs
                str         = 'Euclidean dist\nND'
                ylims       = [0, 10]
                numCond     = numPairs

            ## Calculate traces ##
            for t in range(t1,t2):
                vals        = np.full( numCond, np.nan  )
                v_prev   = np.full( numCond, np.nan )
                for pp in range( numCond ):
                    # cosine or euclidean #
                    v1raw = dat[DT,t,:,Pairs[pp][0]]
                    v2raw = dat[DT,t,:,Pairs[pp][1]]
                    v1  = v1raw - CCM[t,:]
                    v2  = v2raw - CCM[t,:]
                    v1  = v1 / np.linalg.norm(v1)           # get direction only
                    v2  = v2 / np.linalg.norm(v2)           # get direction only
                    if mm == 0:         # angle of pair
                        vals[pp] = np.dot(v1,v2)  
                    elif mm == 1:       # distance of pair
                        vals[pp] = np.linalg.norm(v1raw-v2raw)
                Pair_trace[DT,mm,t,:numCond]    = vals
                Mean_trace[DT,mm,t]      = np.nanmean( vals )
                # if np.all(np.isnan(vals)):
                #     print('problem!')
                #     print('hi')

            # Calculate window means #
            for ww in range(6):
                vals = np.full( numCond, np.nan  )
                if ww == 0:                         # 1st quarter
                    inds_win = np.arange(t1,q1)
                elif ww == 1:                       # 2nd
                    inds_win = np.arange(q1,q2)
                elif ww == 2:                       # 3rd
                    inds_win = np.arange(q2,q3)
                elif ww == 3:                       # 4th
                    inds_win = np.arange(q3,t2)
                elif ww == 4:                       # 1st timestep
                    inds_win = np.array([t1]) 
                elif ww == 5:                       # last timestep
                    inds_win = np.array([t2-1]) 
                for pp in range( numCond ):
                    if mm == 0 or mm == 1:
                        v1raw  = np.mean( dat[DT,inds_win,:,Pairs[pp][0]], axis=0) 
                        v2raw  = np.mean( dat[DT,inds_win,:,Pairs[pp][1]], axis=0) 
                        v1  = v1raw - np.mean(CCM[inds_win,:],axis=0)
                        v2  = v2raw - np.mean(CCM[inds_win,:],axis=0)
                        v1  = v1 / np.linalg.norm(v1)           # get direction only
                        v2  = v2 / np.linalg.norm(v2)           # get direction only
                        if mm == 0:    
                            vals[pp] = np.dot(v1,v2)  
                        elif mm == 1:     
                            vals[pp] = np.linalg.norm(v1raw-v2raw)        
                Mean_win[DT,mm,ww]             = np.nanmean(vals)       # (mean across pairs)         
                Pair_win[DT,mm,ww,:numCond]    = vals                   # ()

            ## Calculate Dynamic geometry (last quarter - first quarter) ## 
            for EA in [0,1]:     # exact (time) vs. averaged (over quarter)
                if EA == 0:
                    # inds_early = np.arange(t1,t1+1)  #np.array([t1])
                    # inds_later = np.arange(t2-1,t2)  #np.array([t2-1])
                    inds_early = np.array([t1])
                    inds_later = np.array([t2-1])
                elif EA == 1:
                    inds_early = np.arange(t1,q1)
                    inds_later = np.arange(q3,t2)
        
                vals = np.full( numCond, np.nan  )
                for pp in range(numCond):
                    # early vectors#
                    v1_i_raw = np.mean(dat[DT,inds_early,:,Pairs[pp][0]],axis=0)
                    v2_i_raw = np.mean(dat[DT,inds_early,:,Pairs[pp][1]],axis=0)
                    v1_i  = v1_i_raw - np.mean(CCM[inds_early,:],axis=0)  # 
                    v2_i  = v2_i_raw - np.mean(CCM[inds_early,:],axis=0)  # 
                    # later vectors #
                    v1raw  = np.mean(dat[DT,inds_later,:,Pairs[pp][0]],axis=0) 
                    v2raw  = np.mean(dat[DT,inds_later,:,Pairs[pp][1]],axis=0)
                    v1  = v1raw - np.mean(CCM[inds_later,:],axis=0)  # 
                    v2  = v2raw - np.mean(CCM[inds_later,:],axis=0)  # 
                    # normalize #
                    v1_i  = v1_i / np.linalg.norm(v1_i)
                    v2_i  = v2_i / np.linalg.norm(v2_i)
                    v1  = v1 / np.linalg.norm(v1)           
                    v2  = v2 / np.linalg.norm(v2)     
                    if mm == 0:     # cosine similarity
                        measure_i   =   np.dot(v1_i,v2_i)               # anti-aligned is +1, aligned is -1         
                        measure     =   np.dot(v1,v2)                # anti-aligned is +1, aligned is -1
                    elif mm == 1:   # euclidean distance  
                        measure_i   =   np.linalg.norm(v1_i_raw - v2_i_raw)               # anti-aligned is +1, aligned is -1         
                        measure     =   np.linalg.norm(v1raw - v2raw)                # anti-aligned is +1, aligned is -1
                    vals[pp]    =   measure - measure_i 
                # Dynamic values (last quarter - first quarter of delay)
                Pair_dyn[DT,mm,:numCond,EA]    = vals                                  # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]
                Mean_dyn[DT,mm,EA]             = np.nanmean(vals)                     # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]

    # G['Pairs']          = Pairs    # list of pairs (key to indices)
    # G['Pairs_symdists']     = Calc_pairs_symdists(Pairs)
    G['Mean_trace']     = Mean_trace        # [DT,mm,t]     DT: datatype, mm: measure, t: time
    G['Pair_trace']     = Pair_trace        # [DT,mm,t,pp]  DT: datatype, mm: measure, t: time, pp: pair
    G['Mean_win']       = Mean_win        # [DT,mm,ww]      DT: datatype, mm: measure, ww: window
    G['Pair_win']       = Pair_win        # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
    G['Mean_dyn']       = Mean_dyn        # [DT,mm]         DT: datatype, mm: measure
    G['Pair_dyn']       = Pair_dyn        # [DT,mm,pp]      DT: datatype, mm: measure, pp:pair
 
    return G

def Calc_pairs_symdists(Pairs):
    # calculate 
    symdists = np.full((len(Pairs)),np.nan)
    for pp in range(len(Pairs)):
        symdists[pp] = np.abs( Pairs[pp][0] - Pairs[pp][1] )
    return symdists

def Calculate_GI( p, G ):

    GI  = np.full( (2,10,2) , np.nan)   # [DT,GI,exact/average]   DT: datatype (model, random), index: the geometric measure

    for DT in range(2):      # datatype
        for ii in range(10):  # geometric index    
            for EA in [0,1]:

                # Collinearity #
                if ii == 0:         # Collinearity, ambient space, ND 
                    str         = 'Collinearity, early'
                    conds       = list(combinations([0,1,2,4,5,6], 2))
                elif ii == 1:         # Collinearity, ambient space, ND 
                    str         = 'Collinearity, late'
                    conds       = list(combinations([0,1,2,4,5,6], 2))
                elif ii == 2:         # Collinearity, ambient space, ND 
                    str         = 'Collinearity, change'
                    conds       = list(combinations([0,1,2,4,5,6], 2))

                # Ordered collinearity #
                elif ii == 3:         # Ordered collinearity, ambient space, ND 
                    str         = 'OCI, early'
                    conds       = [(0,6),(0,5),(0,4),(1,6),(1,5),(1,4),(2,6),(2,5),(2,4),(0,1),(0,2),(1,2),(5,6),(4,6),(4,5)]  # all opposites
                elif ii == 4:         # Ordered collinearity, ambient space, ND  
                    str         = 'OCI, late'
                    conds       = [(0,6),(0,5),(0,4),(1,6),(1,5),(1,4),(2,6),(2,5),(2,4),(0,1),(0,2),(1,2),(5,6),(4,6),(4,5)]  # all opposites
                elif ii == 5:         # 
                    str         = 'OCI, change'
                    conds       = [(0,6),(0,5),(0,4),(1,6),(1,5),(1,4),(2,6),(2,5),(2,4),(0,1),(0,2),(1,2),(5,6),(4,6),(4,5)]  # all opposites

                # Mean angle #
                elif ii == 6:         #  
                    str         = 'Mean angle, early'
                    conds       = list(combinations([0,1,2,3,4,5,6], 2))
                elif ii == 7:         # 
                    str         = 'Mean angle, late '
                    conds       = list(combinations([0,1,2,3,4,5,6], 2))
                elif ii == 8:         # 
                    str         = 'Mean angle, change'
                    conds       = list(combinations([0,1,2,3,4,5,6], 2))

                # Mean distance, change #
                elif ii == 9:         # Opposites + linearly arranged, ambient space, ND  
                    str         = 'Mean distance change'
                    conds       = list(combinations([0,1,2,3,4,5,6], 2))


                # calculate each geometric index #

                if EA == 0:     # exact time
                    ind_early = 4   # exact first time point
                    ind_later = 5   # exact last time point
                elif EA == 1:   # average time
                    ind_early = 0   # 1st quarter
                    ind_later = 3   # 4th quarter

                vals = np.full((len(conds)),np.nan)   

                for pp in range(len(conds)):
                    for ppp in range(len(G['Pairs'])):  
                        if conds[pp] == G['Pairs'][ppp] :                    
                            if ii == 0 or ii == 1:              # collinearity
                                mm = 0  # 0: cosine, 1: euclidean dist
                                if ii == 0:
                                    ww       = ind_early 
                                elif ii == 1:
                                    ww       = ind_later
                                vals[pp] =  np.abs( G['Pair_win'][DT,mm,ww,ppp] )     # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
                            elif ii == 2:       # collinearity change
                                mm = 0  # cosine
                                first_quarter =  np.abs( G['Pair_win'][DT,mm,ind_early,ppp]  )
                                last_quarter  =  np.abs( G['Pair_win'][DT,mm,ind_later,ppp]  )                                  
                                vals[pp] =  last_quarter - first_quarter       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
                            elif ii == 3 or ii == 4:      # oci, 1st and 4th
                                mm       = 0  # 0: cosine, 1: euclidean dist
                                if ii == 3:     
                                    ww       = ind_early  # 0-3: quarters, 4: last time point
                                elif ii == 4:       
                                    ww       = ind_later  # 0-3: quarters, 4: last time point
                                # determine sign of the alignment #
                                sign1    = np.sign( conds[pp][0] - 3 )
                                sign2    = np.sign( conds[pp][1] - 3 )
                                if sign1 == sign2:
                                    SIGN = +1         
                                else:
                                    SIGN = -1   
                                vals[pp] =  SIGN * G['Pair_win'][DT,mm,ww,ppp]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
                            elif ii == 5:       # oci change
                                mm = 0
                                sign1    = np.sign( conds[pp][0] - 3 )
                                sign2    = np.sign( conds[pp][1] - 3 )
                                if sign1 == sign2:
                                    SIGN = +1         
                                else:
                                    SIGN = -1   
                                first_quarter = SIGN * G['Pair_win'][DT,mm,ind_early,ppp]
                                last_quarter  = SIGN * G['Pair_win'][DT,mm,ind_later,ppp]                                    
                                vals[pp] =  last_quarter - first_quarter       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
                            elif ii == 6 or ii == 7:              # mean angle
                                mm = 0  # 0: cosine, 1: euclidean dist
                                if ii == 6:
                                    ww       = ind_early 
                                elif ii == 7:
                                    ww       = ind_later
                                vals[pp] =  G['Pair_win'][DT,mm,ww,ppp]      # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
                            elif ii == 8:       # mean angle change   
                                mm = 0
                                vals[pp] =  G['Mean_dyn'][DT,mm,EA]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
                            elif ii == 9:       # mean distance change
                                mm = 1
                                vals[pp] =  G['Mean_dyn'][DT,mm,EA]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
                            # continue
                    if np.isnan( vals[pp] ):
                        print('error')
                        breakpoint()
                
                # calculate and store mean #
                GI[DT,ii,EA] = np.nanmean( vals )

    return GI



def Calculate_C(p, Data, PCA_top_dims=-1, pca=None, show_plot=False ):

    from scipy.stats import kendalltau

    ii          = Rank1_trials(p)

    # time indices #
    _, t1, t2, _, _, _     = tk.parms_test( p, None, TT = 0 )
    q1     = int( t1 + 0.25 * (t2-t1))
    q2     = int( t1 + 0.5  * (t2-t1))
    q3     = int( t1 + 0.75 * (t2-t1))
         
    # Get trial data for Item 1 #
    M = p.n_items
    rank1vec,_      = Rankvector(p)
    distinds        = trial_parser_symbolic_dist(p,flatten_indices=False)
    x = y   = np.arange(M)                  # x, y are index vectors for item 1 and 2, respectively
    xv, yv  = np.meshgrid(x, y, indexing='ij')             # 2D matrices  
    xv, yv  = (xv.flatten(), yv.flatten())

    numMeasures = 4
    numWindows  = 7 
    maxNumConds = p.n_items  # maximum number of conditions

    ### initialize outputs ###
    C           = {}   # dictionary
    # trace of values #
    Mean_trace   = np.full((2,numMeasures,t2),np.nan)      # [<data,rand>,measure,time]
    Cond_trace   = np.full((2,numMeasures,t2,maxNumConds),np.nan) # [<data,rand>,measure,time,pair]
    # window mean values (in windowed periods of the delay)
    Mean_win     = np.full((2,numMeasures,numWindows),np.nan)            # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]
    Cond_win     = np.full((2,numMeasures,numWindows,maxNumConds),np.nan)   # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]
    # dynamic value (late delay - early delay)
    Mean_dyn     = np.full((2,numMeasures,2),np.nan)            # [<data,rand>,index,exact/average time]
    Cond_dyn     = np.full((2,numMeasures,maxNumConds,2),np.nan)   # [<data,rand>,index,exact/average time]

    ### 1. Random data vs. real data ##################
    if Data['DATA'][0].shape[-1] == p.n_items:
        data_real       = Data['DATA'][0]        # [t,N,rank1]     data  
    else:
        data_real       = Data['DATA'][0][:,      :, ii]         # [t,N,rank1]     data     
    data_rand       = np.random.randn(data_real.shape[0],data_real.shape[1],data_real.shape[2])
    dat             = np.concatenate( (data_real[np.newaxis,:],data_rand[np.newaxis,:]), axis=0 )

    ### 2. (optional) Filter lower PCs from data ########
    #   note: since want to calculate correlations, we will NOT discard mean-centering ##
    if PCA_top_dims > 0:       
        if pca == None:
            N        = data_real.shape[1]
        else:
            N        = pca.components_.shape[1]
        dat_filt = dat.copy()
        dat      = np.full((2,dat.shape[1],N,dat.shape[3]),np.nan)
        for DT in [0,1]:
            # calculates PCA, filters for top PCs, converts back to original axes
            if pca == None:
                dat_filtered, _      = nna.PCA_filter( dat_filt[DT,:,:,:], PCA_top_dims )   # nna.PCA_filter restores mean-center
                dat[DT,:,:,:]        = dat_filtered
            else:   # if pca available, zero out lower PCs and inverse_transform
                dat_filt[DT,:,PCA_top_dims:,:]   = 0    # zero, i.e. filter for top PCs
                dat[DT,:,:,:]                    = nna.PCA_inverse(dat_filt[DT,:,:,:],pca)  # invert to ambient


    # Calculate #
    for DT in [0,1]:  # (both data 'types': real and random)

        # ### Calculate cross-condition mean (CCM) ###
        # CCM      = np.mean( dat[DT,:,:,:], axis = -1)               

        ### 3. Calculate each correlation measure ####
        for mm in range(4):

            if mm == 0:      # Correlation (Kendall), sensory   (reference to beginning of delay)
                numCond     = p.n_items
                tref        = t1
                tref_win    = np.arange(t1,q1)
            elif mm == 1:    # Correlation (Kendall), memory    (reference to end of delay)
                numCond     = p.n_items
                tref        = t2-1
                tref_win    = np.arange(q3,t2)
            elif mm == 2:    # Correlation (Kendall), Symbolic distance
                numCond     = p.n_items - 1
            elif mm == 3:    # Correlation, all across-item
                numCond     = 1

            ## Calculate traces ##
            for t in range(t1,t2):
                if mm == 0 or mm == 1:          # within-item
                    vals        = np.full( numCond, np.nan  )
                    for ii in range( numCond ):     # item 1 rank
                        vref        = dat[DT,tref,:,ii]
                        v           = dat[DT,t,:,ii]
                        vals[ii]    = kendalltau(vref,v).correlation  
                    Cond_trace[DT,mm,t,:numCond]    = vals
                    Mean_trace[DT,mm,t]             = np.nanmean( vals )
                elif mm == 2:                   # symbolic distance
                    for symdist in range( M-1 ):    
                        inds    = distinds[symdist][1,:].flatten()       # trial indices fulfilling this symdist
                        vals    = np.full((M-1-symdist),np.nan)         
                        for pp in range(inds.size):    
                            i1,i2         = ( xv[inds[pp]], yv[inds[pp]] )
                            v1            = dat[DT,t,:,i1] 
                            v2            = dat[DT,t,:,i2]
                            vals[pp]      = kendalltau( v1 , v2 ).correlation
                        Cond_trace[DT,mm,t,symdist]    = np.mean(np.array(vals))  
                        Mean_trace[DT,mm,t]            = np.nan                     # omit this mean
                elif mm == 3:                   # all across-pairs
                    indd    = trial_parser_symmetry(p)
                    inds    = indd[1].flatten()       # trial indices with all pairwise combinations
                    # calc correlations #
                    vals    = np.full((inds.size),np.nan)         
                    for pp in range(inds.size):    
                        i1,i2         = ( xv[inds[pp]], yv[inds[pp]] )
                        v1            = dat[DT,t,:,i1] 
                        v2            = dat[DT,t,:,i2]
                        vals[pp]      = kendalltau( v1 , v2 ).correlation
                    Cond_trace[DT,mm,t,0]    = np.mean(vals)
                    Mean_trace[DT,mm,t]      = np.nan                     # omit this mean


            # Calculate window means #
            for ww in range(6):

                if ww == 0:                         # 1st quarter
                    inds_win = np.arange(t1,q1)
                elif ww == 1:                       # 2nd
                    inds_win = np.arange(q1,q2)
                elif ww == 2:                       # 3rd
                    inds_win = np.arange(q2,q3)
                elif ww == 3:                       # 4th
                    inds_win = np.arange(q3,t2)
                elif ww == 4:                       # first timestep
                    inds_win = np.array([t1]) 
                elif ww == 5:                       # last timestep
                    inds_win = np.array([t2-1]) 
                # elif ww == 6:                       # full delay
                #     inds_win = np.arange(t1,t2) 
                    
                if mm == 0 or mm == 1:
                    vals        = np.full( numCond, np.nan  )
                    for ii in range( numCond ):     # item 1 rank
                        vref        = np.mean( dat[DT,tref_win,:,ii], axis=0) 
                        v           = np.mean( dat[DT,inds_win,:,ii], axis=0) 
                        vals[ii]    = kendalltau(vref,v).correlation 
                    Cond_win[DT,mm,ww,:numCond]    = vals                   # ()
                    Mean_win[DT,mm,ww]             = np.nanmean(vals)       # (mean across pairs)  
                elif mm == 2:
                    for symdist in range( M-1 ):    
                        inds    = distinds[symdist][1,:].flatten()       # trial types fulfilling this symdist
                        vals    = np.full((M-1-symdist),np.nan)         
                        for pp in range(inds.size):    
                            i1,i2         = ( xv[inds[pp]], yv[inds[pp]] )
                            v1            = dat[DT,inds_win,:,i1] 
                            v2            = dat[DT,inds_win,:,i2]
                            vals[pp]      = kendalltau( v1 , v2 ).correlation
                        Cond_win[DT,mm,ww,symdist]    = np.mean(vals)    # take mean across trial types
                    Mean_win[DT,mm,ww]                = np.nan                     # omit this mean                           
                elif mm == 3:
                    indd    = trial_parser_symmetry(p)
                    inds    = indd[1].flatten()       # trial indices with all pairwise combinations
                    vals    = np.full((inds.size),np.nan)
                    for pp in range(inds.size):    
                        i1,i2         = ( xv[inds[pp]], yv[inds[pp]] )
                        v1            = dat[DT,inds_win,:,i1] 
                        v2            = dat[DT,inds_win,:,i2]
                        vals[pp]      = kendalltau( v1 , v2 ).correlation
                    Cond_win[DT,mm,ww,0]    = np.mean(vals)              # mean across all across-pairs
                    Mean_win[DT,mm,ww]      = np.nan                     # omit this mean


            ## Calculate Dynamic correlation (delay end - delay begin) ## 
            for EA in [0,1]:     # exact (time) vs. averaged (over quarter)

                if EA == 0:     # exact
                    inds_early = np.array([t1])
                    inds_late = np.array([t2-1])
                    TREF      = np.array([tref])
                elif EA == 1:   # averaged
                    inds_early = np.arange(t1,q1)
                    inds_late = np.arange(q3,t2)
                    TREF      = tref_win
        
                if mm == 0 or mm == 1:
                    vals        = np.full( numCond, np.nan  )
                    for ii in range(numCond):       # item 1 ranks
                        # early delay  #
                        vref_i  = np.mean(dat[DT,TREF,:,ii], axis=0)
                        v_i     = np.mean(dat[DT,inds_early,:,ii], axis=0)
                        # late delay #
                        vref    = np.mean(dat[DT,TREF,:,ii], axis=0)
                        v       = np.mean(dat[DT,inds_late,:,ii], axis=0)
                        measure_i   =   kendalltau(vref_i,v_i).correlation           # early          
                        measure     =   kendalltau(vref,v).correlation               # late
                        vals[ii]    =   measure - measure_i 
                    # Dynamic values (last quarter - first quarter of delay)
                    Cond_dyn[DT,mm,:numCond,EA]    = vals                                  # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]
                    Mean_dyn[DT,mm,EA]             = np.nanmean(vals)                     # [<data,rand>,index,<1st,2nd,3rd,4th quarter of delay>]
                elif mm == 2:
                    for symdist in range( M-1 ):    
                        inds    = distinds[symdist][1,:].flatten()       # trial types fulfilling this symdist
                        vals    = np.full((M-1-symdist),np.nan)         
                        for pp in range(inds.size):    
                            i1,i2         = ( xv[inds[pp]], yv[inds[pp]] )
                            v1_i            = dat[DT,inds_early,:,i1] 
                            v2_i            = dat[DT,inds_early,:,i2]
                            v1              = dat[DT,inds_late,:,i1] 
                            v2              = dat[DT,inds_late,:,i2]
                            corr_i          = kendalltau( v1_i , v2_i ).correlation     # early
                            corr            = kendalltau( v1 , v2 ).correlation         # late
                            vals[pp]        = corr - corr_i
                        Cond_dyn[DT,mm,symdist,EA]    = np.mean(np.array(vals))    # take mean across trial types
                    Mean_dyn[DT,mm,EA]                = np.nanmean(Cond_dyn[DT,mm,:,EA])                     # omit this mean                           
                elif mm == 3:
                    indd    = trial_parser_symmetry(p)
                    inds    = indd[1].flatten()       # trial indices with all pairwise combinations
                    vals    = np.full((inds.size), np.nan)         
                    for pp in range(inds.size):    
                        i1,i2         = ( xv[inds[pp]], yv[inds[pp]] )
                        v1_i            = dat[DT,inds_early,:,i1] 
                        v2_i            = dat[DT,inds_early,:,i2]
                        v1              = dat[DT,inds_late,:,i1] 
                        v2              = dat[DT,inds_late,:,i2]
                        corr_i          = kendalltau( v1_i , v2_i ).correlation     # early
                        corr            = kendalltau( v1 , v2 ).correlation         # late
                        vals[pp]        = corr - corr_i
                    Cond_dyn[DT,mm,0,EA]    = np.mean(vals)    # take mean across trial types
                    Mean_dyn[DT,mm,EA]      = np.nan                    # omit this mean                           

    C['Mean_trace']     = Mean_trace        # [DT,mm,t]     DT: datatype, mm: measure, t: time
    C['Cond_trace']     = Cond_trace        # [DT,mm,t,pp]  DT: datatype, mm: measure, t: time, pp: pair
    C['Mean_win']       = Mean_win        # [DT,mm,ww]      DT: datatype, mm: measure, ww: window
    C['Cond_win']       = Cond_win        # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair
    C['Mean_dyn']       = Mean_dyn        # [DT,mm]         DT: datatype, mm: measure
    C['Cond_dyn']       = Cond_dyn        # [DT,mm,pp]      DT: datatype, mm: measure, pp:pair
 
    return C


def Calculate_CI( p, C ):

    CI  = np.full( (2,3,2) , np.nan)   # [DT,GI,exact/average]   DT: datatype (model, random), index: the geometric measure

    for DT in range(2):      # datatype
        for EA in [0,1]:
            for ii in range(3):  # correlation index    
                if ii == 0:      #  'Sensory corr'
                    mm       = 0    # 0: sensory
                    conds    = np.arange(p.n_items)
                    if EA == 0:
                        vals  = C['Cond_trace'][DT,mm,int(p.t2-1),conds]
                    elif EA == 1:
                        vals  = C['Cond_win'][DT,mm,3,conds]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair    
                elif ii == 1:    #  'Largest symdist corr, early'
                    mm       = 2    # 2: symdist
                    conds    = np.array([3,4,5])  # largest symdist (A-G)
                    if EA == 0:
                        vals     = C['Cond_win'][DT,mm,4,conds]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair    
                    elif EA == 1:
                        vals     = C['Cond_win'][DT,mm,0,conds]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair    
                elif ii == 2:    #  'Largest symdist corr, late'
                    mm       = 2    # 2: symdist
                    conds        = np.array([3,4,5])  # largest symdist (A-G)
                    if EA == 0:
                        vals     = C['Cond_win'][DT,mm,5,conds]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair    
                    elif EA == 1:
                        vals     = C['Cond_win'][DT,mm,3,conds]       # [DT,mm,ww,pp]   DT: datatype, mm: measure, ww: window, pp: pair    
                meancorr = np.mean(vals)
                CI[DT,ii,EA] = meancorr

    return CI


def Task_modes(p, Data, model, F ):

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    # (basic parms and data) #
    _, t1, t2, _, _, _     = tk.parms_test( p, None, TT = 0 )
    ii                     = Rank1_trials(p)
    data                   = Data['DATA'][0]        # [t,N,rank1]

    q1     = int( t1 + 0.25 * (t2-t1))
    q2     = int( t1 + 0.5  * (t2-t1))
    q3     = int( t1 + 0.75 * (t2-t1))
    qafter = int( t2 + 0.25 * (t2-t1))

    inds_q1         = np.arange(t1,q1)
    inds_q4         = np.arange(q3,t2)
    inds_qafter     = np.arange(t2,qafter)


    # if 0:  # filter top PCs
    #     topdims     = 10
    #     inds        = np.arange(topdims)
    #     dat, _      = nna.PCA_top_dims( dat, topdims )
    #     CCM      = np.mean( dat, axis=2)      

    # I. Rank mode #
    # i.e. least-norm solution to linear decoding of Rank #
    #  i.e. y = Ax  #
    # A           = data[t2-1,:,ii]     # final delay timestep
    A           = data[t1,:,ii]         # first delay timestep
    y           = 1 * ( ranks(p,ii)[0] - 3 )        # relative rank (item 1) during delay
    A_pinv      = np.linalg.pinv(A)
    x           = A_pinv @ y
    # x         = ( A.T @ np.linalg.inv( A @ A.T ) ) @ y
    rankaxis    = x / np.linalg.norm(x)

    # II. Choice mode #
    choiceinds  = trial_parser_symmetry(p)
    choice_1    = np.mean( np.mean( data[inds_qafter,:,:] , axis = 0)[:, choiceinds[0]], axis=-1 ).ravel()              
    choice_2    = np.mean( np.mean( data[inds_qafter,:,:] , axis = 0)[:, choiceinds[1]], axis=-1 ).ravel()              
    choiceaxis  = (choice_1 - choice_2) / np.linalg.norm( choice_1 - choice_2 )

    # III.  Input mode #
    if 1:
        # item_input      =  F['X_input'][0,:]   # A
        item_input      =  F['X_input'][3,:]     # D
        wu              = model.wu.detach().numpy()         # wu:[N,U]
        rawaxis         = wu @ item_input[:,np.newaxis]
        rawaxis         = rawaxis.flatten()
        inputaxis       = rawaxis / np.linalg.norm( rawaxis )
    else:
        A           = data[t1,:,ii]
        y           = 10 * ( ranks(p,ii)[0] - 3 )        # relative rank (item 1) during delay
        A_pinv      = np.linalg.pinv(A)
        x           = A_pinv @ y
        inputaxis    = x / np.linalg.norm(x)

    # install #
    tax        = np.full((p.N,3),np.nan)
    tax[:,0]   = choiceaxis                         # CHOICE   
    tax[:,1]   = rankaxis                           # ORDERED REGRESSION
    tax[:,2]   = inputaxis                          # INPUT

    # # # # IV. orthogonalize #
    # w, _        = np.linalg.qr(tax)
    # tax         = w

    # report angles #
    if 1:
        ang1 = np.dot( tax[:,0] , tax[:,1] )
        ang2 = np.dot( tax[:,0] , tax[:,2] )
        ang3 = np.dot( tax[:,1] , tax[:,2] )
        print('(angles) Rank vs. Choice: %0.2f, Rank vs. Input: %0.2f, Choice vs. Input: %0.2f' % (ang1,ang2,ang3) )

    # set up basis # 
    zerovec = np.full((p.N),0)
    basis = nna.Linearbasis_3D(zerovec,tax[:,0],tax[:,1],tax[:,2])

    return basis




def Calculate_Dim_Delay(p, F, model, TT = 0, verbose = False):  # runs model on multiple Time/Input conditions, stores activity

    # R                   = 100      # number of full-matrix trials
    # sig_test            = .0000001  #0.7  #12  # test noise level (intrinsic)

    R                   = 1      # number of full-matrix trials
    sig_test            = 0  #0.7  #12  # test noise level (intrinsic)

    if verbose:
        print('Running dim analysis in condition %d' % (TT))

    T, t1, t2, _, _, _  = tk.parms_test(p,None,TT)
    # INDS, trains, tests = tk.Get_trial_indices(p)
    i1inds    = Rank1_trials(p)
    q1     = int( t1 + 0.25 * (t2-t1))
    q2     = int( t1 + 0.5  * (t2-t1))
    q3     = int( t1 + 0.75 * (t2-t1))

    q                   = copy.deepcopy(p)   # copy so we can change noise level
    q.sig               = sig_test
    q.noise_test        = True
    _, _, _, _, input, _     = tk.parms_test(p, F, TT)
    T                   = input.shape[0]
    # x                   = np.full( (T, R*TR, p.N), np.nan )
    x                   = np.full( (T, p.n_items, p.N), np.nan )
    for rr in range(R):
        with torch.no_grad():
            D           = tk.run_model( model, q, input, hidden_output = True )
            # a           = rr*TR
            # b           = (rr+1)*TR
            # x[:,a:b,:]  = D['Out_h']  #np.reshape(data, (TR,p.N) , order='C')
            x             = D['Out_h'].detach().numpy()[:,i1inds,:]     # [T,B,N]

    ### Calculate effective dim in windows #####
    neff_win = np.full ( 5, np.nan )   # [window]
    for ww in range(5):
        if ww == 0:
            inds_win = np.arange(t1,q1)
        elif ww == 1:
            inds_win = np.arange(q1,q2)
        elif ww == 2:
            inds_win = np.arange(q2,q3)
        elif ww == 3:
            inds_win = np.arange(q3,t2)
        elif ww == 4:
            inds_win = np.array([t2-1])
        xwin = np.mean( x[inds_win,:,:], axis=0 )
        _, s, _    = np.linalg.svd( xwin , full_matrices=False)
        neff_win[ww]   = np.power( np.sum(s), 2 ) / np.sum( np.power(s,2) )
    #############################################

    ### Calculate effective dim in a continuous trace #####
    neff_trace = np.full ( p.t2, np.nan )   # [window]
    for t in range(p.t2):
        _, s, _    = np.linalg.svd( x[t,:,:] , full_matrices=False)
        neff_trace[t]   = np.power( np.sum(s), 2 ) / np.sum( np.power(s,2) )
    ###############################################

    return neff_win, neff_trace

def C_Index(p, Data, Data_lz):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    fig = plt.figure( constrained_layout=True, figsize=(14,5) )
    plt.get_current_fig_manager().window.setGeometry(300,200,800,800) 
    plt.suptitle('C index',fontsize=24)
    from itertools import combinations

    distinds               = trial_parser_symbolic_dist(p,flatten_indices=True)
    T, t1, t2, _, _, _     = tk.parms_test( p, None, TT = 0 )

    for tt in range(2):  

        if tt == 0:             # non-linear
            Dat = Data
            clr = 'k'
            datastr = 'Non-linear data'
        elif tt == 1:                   # linearized
            Dat = Data_lz
            clr = [.7,.7,.7]
            datastr = 'Linearized data'

        for vv in range(0,2):

            if vv == 0:         # Radius on Symbolic Distance, ND

                str         = 'SymDist x Radius\nND'
                inds        = np.arange(p.N)       # Ambient dim
                ylims       = [0,1.05]
                dat         = Dat['DATA'][0]         # [t,N,B]
                if 0:  # filter top PCs
                    topdims     = 12
                    inds        = np.arange(topdims)
                    dat, _      = nna.PCA_top_dims( dat, topdims )
                center      = np.mean( dat, axis=2)               

            elif vv == 1:       # FP basis, 2D

                str         = 'SymDist x Radius\n2D'
                ylims       = [0,1.05]
                dat         = Dat['DATA_lz'][0][:,0:2,:]              # [t,2,rank1]
                center      = Dat['DATA_lz'][0][t1-1,0:2,  0]       

            # elif vv == 2:       # Empirical basis, 2D

            #     str         = 'opposite\n2D (emp))'
            #     # pairs      = [[0,5],[0,4],[1,6],[1,5],[1,4],[2,6],[2,5],[2,4]]
            #     pairs       = [[1,5],[1,4],[2,5],[2,4]]
            #     ylims       = [-1.05, 1.05]
                
            #     inds        = np.arange(2)    # 2D
            #     center      = Dat['DATA_emp'][0][t1-1,   :,  0]              # [2]         
            #     dat         = Dat['DATA_emp'][0][:,      :, ii]              # [t,2,rank1]

            # elif vv == 3:         # Ambient neural space, ND  (top PCs)

            #     str         = 'align\nND'
            #     pairs       = list(combinations([0,1,2,4,5,6], 2))

            #     inds = np.arange(p.N)       # Ambient dim
            #     ylims       = [0, 1.05]
            #     dat         = Dat['DATA'][0][:,      :, ii]         # [t,2,rank1]
            #     if 1:  # filter top PCs
            #         topdims     = 10
            #         inds        = np.arange(topdims)
            #         dat, _      = nna.PCA_top_dims( dat, topdims )
            #     center      = np.mean( dat, axis=2)               

            # elif vv == 4:       # Alignment Index, 2D

            #     str         = 'align\n2D (emp)'
            #     pairs       = list(combinations([0,1,2,4,5,6], 2))
            #     ylims       = [0, 1.05]
            #     inds        = np.arange(2)    # 2D
            #     center      = Dat['DATA_emp'][0][t1-1,   :,  0]              # [2]              
            #     dat         = Dat['DATA_emp'][0][:,      :, ii]              # [t,2,rank1]

            # distance matrix: 1st column: symbolic distance, 2nd col: trial index
            distmatrix = np.array([]).reshape(0,2)
            for dist in range(p.n_items-1):
                numinds    = distinds[dist].size
                ii         = np.concatenate((dist * np.ones(numinds)[:,np.newaxis], distinds[dist][:,np.newaxis]),axis=1)
                distmatrix = np.concatenate((distmatrix,ii),axis=0)
            distmatrix = np.int32(distmatrix)

            numtrials      = distmatrix.shape[0]
            trace       = np.full((T),np.nan)
            # trace_all   = np.full((t2,nump),np.nan)

            for t in range(t2,T):
                dots = np.full( numtrials, np.nan  )
                if np.ndim(center) == 2:         # Time-dependent center
                    coords  = dat[t,:,distmatrix[:,1]] - center[t,:]   # centered coordinates
                else:               # Constant center
                    coords  = dat[t,:,distmatrix[:,1]] - center
                radii   = np.linalg.norm(coords,axis=1)                    
                rho,_   = scipy.stats.spearmanr( radii, distmatrix[:,0] )
                trace[t] = rho
                # trace_all[t,:]    = dots

            # plot #
            ax          = plt.subplot(5,2,vv*2+tt+1)  
            # ax.plot(np.arange(t1,t2),trace_all[t1:t2,:],color=clr, linewidth=2)
            ax.plot(np.arange(0,T),trace[0:T],color='g', linewidth=4)
            ax.set_xlim([t2,T])
            ax.set_ylim(ylims)
            ax.set_xlabel('Time (dt)', fontsize=18,  fontweight='bold')
            if vv == 0:
                ax.set_title(datastr)
            if tt == 0:
                ax.set_ylabel('%s' % (str), fontsize=18, rotation='horizontal')
                ax.yaxis.set_label_coords(-0.3, 0.3)
    plt.tight_layout()

def Linear_Dynamics_Delay_Period( p, F, Data_sys, TT, topdims=10 ):  # runs model on multiple Time/Input conditions, stores activity

    ii                     = Rank1_trials(p)
    N                      = p.N            # ambient dimensionality of RNN
    M                      = p.n_items      # number of trial types (number of first stimuli)

    _, t1, t2, _, _, _     = tk.parms_test( p, F, TT = 0 )

    if 1:           # full delay
        T                      = int(t2-t1-2)   # total number of timesteps to model
        interval               = range(t1,(t2-1))
    # else:           # last half of delay
    #     halfdelay              = int(p.d1/2)
    #     T                      = int(t2-t1-2)-halfdelay   # total number of timesteps to model
    #     interval               = range(t1+halfdelay,t2-1)

    Data        = Data_sys['DATA']          # tensor format data

    # I. Get baseline state #
    center      = np.mean( Data[TT][:t1,:,0] , axis=0)  # pre-trial baseline

    # II. Get states + flows #
    X             = np.full( (M*T,N), np.nan  )
    Xdot          = np.full( (M*T,N), np.nan  )
    for m in range(M):
        traj         = Data[TT][interval,:,ii[m]] - center
        a, b         = m*T, (m+1)*T
        X[a:b,:]     = traj[:-1,:]
        Xdot[a:b,:]  = (p.tau/p.dt)*np.diff(traj, n=1, axis=0 )

    # III. Take only top 10 PC dimensions of the Delay data #
    X, pca  = nna.PCA_top_dims(X, topdims )                # filter states
    Xdot    = nna.project_data(Xdot,   pca, center=False)   # filter flow vectors
                
    # IV.  Estimate Linear Dynamics using Ordinary Least Squares estimate #
    # a_lsq       =  np.linalg.inv( X.T @ X ) @ X.T @ Xdot 
    # a_lsq       = newinv(X.T @ X , X.T @ Xdot) # (X.T @ X)^-1 @ Xdot
    A_lsq, residual, rank, s        =  scipy.linalg.lstsq(X, Xdot)
    residual_manual                 =  np.sum((Xdot - np.dot(X, A_lsq))**2,axis=0)
    assert np.all(np.isclose(residual,residual_manual))     # sanity check

    # V.   Get relevant information from Fit  #
    cond        = s[0] / s[-1]                        # condition number
    R2          = Calculate_R2( residual, Xdot )      # (uncentered sum of squares)
    print('(OLS LD) Rank: %d, Matrix condition #: %g, R sq = %0.2f' % (rank, cond, R2) )

    # VI.  Identify oscillatory mode with period nearest half-delay #
    l, _            = np.linalg.eig(A_lsq)
    v1, v2, freq    = Identify_Oscillatory_Mode( p, A_lsq , Setting = -1)
    V1              = (pca.components_.T @ v1).flatten()
    V2              = (pca.components_.T @ v2).flatten()
    oscbasis        = nna.Linearbasis_2D( center, V1, V2 )

    # install output #
    LD                          = {}        
    LD['A']                     = A_lsq     # dynamics matrix (10D) #
    LD['l']                     = l         # eigenvalues #
    LD['Condition_num']         = cond      # condition number of regressor matrix #
    LD['R2']                    = R2        # R squared of the fit
    LD['oscbasis']              = oscbasis         #     
    LD['oscbasis_freq']         = freq
    LD['pca']                   = pca       # pca for delay period data

    return LD


def Calculate_Angles( p, F, model, Data_sys, ld,  TT=0, topdims=10 ):  # runs model on multiple Time/Input conditions, stores activity

    ii                     = Rank1_trials(p)
    N                      = p.N            # ambient dimensionality of RNN
    M                      = p.n_items      # number of trial types (number of first stimuli)

    _, t1, t2, _, _, _     = tk.parms_test( p, F, TT = 0 )
    T                      = int(t2-t1)   # total number of timesteps to model
    # Data                   = Data_sys['DATA']          # tensor format data

    q1     = int( t1 + 0.25 * (t2-t1))
    q2     = int( t1 + 0.5  * (t2-t1))
    q3     = int( t1 + 0.75 * (t2-t1))
    qafter = int( t2 + 0.25 * (t2-t1))

    inds_q1         = np.arange(t1,q1)
    inds_q4         = np.arange(q3,t2)
    inds_qafter     = np.arange(t2,qafter)

    # (prelim) Transform original data to (top 10 PCs of) Delay period PCA #
    Data_sys    = tk.Project_Data_PCA(p, F, Data_sys, 0, 0, ld['pca'])
    Data        = Data_sys['PDATA']

    # I.  Identify oscillatory mode  #
    v1, v2, freq    = Identify_Oscillatory_Mode( p, ld['A'], Setting = -1 )
    v               = np.concatenate( ( v1[:,np.newaxis], v2[:,np.newaxis] ) , axis=1 ).squeeze()
    w, _            = np.linalg.qr(v)  # orthogonalize

    # II.  Calculate XCM vector #
    #  a. Calculate mean over 1st and 4th quarter delay, then Conditions (item 1) 
    xcm_q1      = np.mean( np.mean( Data[TT][inds_q1,:,:] , axis = 0)[:,ii], axis=-1 ).ravel()              
    xcm_q4      = np.mean( np.mean( Data[TT][inds_q4,:,:] , axis = 0)[:,ii], axis=-1 ).ravel()              
    xcm_vec     = ( xcm_q4 - xcm_q1 ) / np.linalg.norm( xcm_q4 - xcm_q1 )
    #  b. Calculate angle for XCM vs. Osci, take max value (conservative) #
    xcm_ang     = np.full((1001),np.nan)
    for rr in range(1001):
        if rr == 0:
            vec = xcm_vec
        else:
            vec = np.random.randn(topdims)
            vec = vec/np.linalg.norm(vec)
        cosvals     = np.array( [ np.dot(vec,w[:,0]), np.dot(vec,w[:,1]) ]) 
        # xcm_ang[rr] = np.max( np.abs(cosvals) )
        xcm_ang[rr] = np.mean( np.abs(cosvals) )

    # III. Calculate Choice axis  (linear discriminant) #
    #  a. Calculate mean over 'quarter' choice interval after item 2, then over Conditions (item 1)
    choiceinds  = trial_parser_symmetry(p)
    choice_1    = np.mean( np.mean( Data[TT][inds_qafter,:,:] , axis = 0)[:,choiceinds[0]], axis=-1 ).ravel()              
    choice_2    = np.mean( np.mean( Data[TT][inds_qafter,:,:] , axis = 0)[:,choiceinds[1]], axis=-1 ).ravel()              
    choice_vec  = (choice_1 - choice_2) / np.linalg.norm( choice_1 - choice_2 )
    #  b. Calculate angle for Choice vs. Osci, take min value (conservative) #
    choice_ang     = np.full((1001),np.nan)
    for rr in range(1001):
        if rr == 0:
            vec = choice_vec
        else:
            vec = np.random.randn(topdims)
            vec = vec/np.linalg.norm(vec)
        cosvals     = np.array( [ np.dot(vec,w[:,0]), np.dot(vec,w[:,1]) ]) 
        # choice_ang[rr] = np.min( np.abs(cosvals) )  
        choice_ang[rr] = np.mean( np.abs(cosvals) )  
    #  c. Calculate angle for Choice vs. XCM #
    xcm_choice_ang     = np.full((1001),np.nan)
    for rr in range(1001):
        if rr == 0:
            vec = choice_vec
        else:
            vec = np.random.randn(topdims)
            vec = vec/np.linalg.norm(vec)
        cosval     =  np.dot( vec, xcm_vec )
        xcm_choice_ang[rr] = cosval
    #  d. Calculate angle for Choice vs. Readout #
    readout      = model.wz.detach().numpy()[1,:]   # [R,N] readout weights for choice 1
    readout_pc   = np.matmul( readout, np.transpose( ld['pca'].components_ ) )  # [samp,PC#]
    readout_pc   = readout_pc / np.linalg.norm(readout_pc)  # unit-normalize
    choice_readout_ang     = np.full((1001),np.nan)
    for rr in range(1001):
        if rr == 0:
            vec = choice_vec
        else:
            vec = np.random.randn(topdims)
            vec = vec/np.linalg.norm(vec)
        cosval     =  np.dot( vec, readout_pc )
        choice_readout_ang[rr] = cosval

    # install output #
    ANG                       = {}        
    ANG['xcm_ang']             = xcm_ang     # [angles]  0 index is empirical, the rest are random
    ANG['choice_ang']          = choice_ang  #  " " 
    ANG['xcm_choice_ang']      = xcm_choice_ang  #  " " 
    ANG['choice_readout_ang']  = choice_readout_ang

    return ANG






def Calculate_Choice_Measures( p, F, Data_sys, pca,  TT=0):  # runs model on multiple Time/Input conditions, stores activity

    ii                     = Rank1_trials(p)
    M                      = p.n_items      # number of trial types (number of first stimuli)

    _, t1, t2, _, _, _     = tk.parms_test( p, F, TT = 0 )
    # T                      = int(t2-t1)   # total number of timesteps to model
    # Data                   = Data_sys['DATA']          # tensor format data

    q1     = int( t1 + 0.25 * (t2-t1))
    q2     = int( t1 + 0.5  * (t2-t1))
    q3     = int( t1 + 0.75 * (t2-t1))
    qafter = int( t2 + 0.25 * (t2-t1))

    inds_q1         = np.arange(t1,q1)
    inds_q4         = np.arange(q3,t2)
    inds_qafter     = np.arange(t2,qafter)

    ### I. Calculate Choice Availability time ###########################################
    Readout_1      = Data_sys['RDATA'][0][(t2-1):,1,:]  # [TT][T,R,B]   Readout neuron 1
    Readout_2      = Data_sys['RDATA'][0][(t2-1):,0,:]  # [TT][T,R,B]   Readout neuron 1
    choices    = trial_parser_symmetry(p)
    choice1st    = choices[0]
    choice2nd    = choices[1]
    earliest     = np.array([np.nan])
    for t in range(Readout_1.shape[0]):
        available_1 = np.all( Readout_1[t,choice1st] > Readout_1[t,choice2nd] )
        available_2 = np.all( Readout_2[t,choice1st] > Readout_2[t,choice2nd] )
        if available_1 or available_2:
            earliest = np.array([t])
            break
    if np.isnan(earliest):
        print('something is wrong')
        breakpoint()

    ### II. Calculate Choice Projection ######################################################
    # (prelim) Transform original data w/ Delay period PCA #
    Data_sys    = tk.Project_Data_PCA(p, F, Data_sys, 0, 0, pca)
    Data        = Data_sys['PDATA']

    # 1.  Calculate XCM vector #
    #  a. Calculate mean over 1st and 4th quarter delay, then Conditions (item 1) 
    xcm_q1      = np.mean( np.mean( Data[TT][inds_q1,:,:] , axis = 0)[:,ii], axis=-1 ).ravel()  # [N_reduced]        
    xcm_q4      = np.mean( np.mean( Data[TT][inds_q4,:,:] , axis = 0)[:,ii], axis=-1 ).ravel()  # [N_reduced]            
    xcm_vec     = ( xcm_q4 - xcm_q1 ) / np.linalg.norm( xcm_q4 - xcm_q1 )                       # [N_reduced]
    xcm_cont    = np.mean( Data[TT][:,:,ii], axis = -1)                                 # [T,N_reduced]

    # 2. Calculate Choice axis  (linear discriminant) #
    #  a. Calculate mean over 'quarter' choice interval after item 2, then over Conditions (item 1)
    choiceinds  = trial_parser_symmetry(p)
    if 1:   # 1st 'quarter' after choice
        choice_1    = np.mean( np.mean( Data[TT][inds_qafter,:,:] , axis = 0)[:,choiceinds[0]], axis=-1 ).ravel()              
        choice_2    = np.mean( np.mean( Data[TT][inds_qafter,:,:] , axis = 0)[:,choiceinds[1]], axis=-1 ).ravel()              
    else:   # Last two time steps, 'easy'
        choice_1    = np.mean( np.mean( Data[TT][-2:,:,:] , axis = 0)[:,choiceinds[0]], axis=-1 ).ravel()              
        choice_2    = np.mean( np.mean( Data[TT][-2:,:,:] , axis = 0)[:,choiceinds[1]], axis=-1 ).ravel()              
    choice_vec  = (choice_1 - choice_2) / np.linalg.norm( choice_1 - choice_2 )

    # 3. Calculate Projection #
    chp = np.full( (p.T, M) , np.nan )
    for t in range(t1,t2):
        # chp[t,:]  =  np.dot( Data[TT][t,:,ii] - xcm_cont[t,:] , choice_vec)
        chp[t,:]  =  np.dot( Data[TT][t,:,ii] , choice_vec)

    ### IV. Obtain Readout "Projection" ###############################################
    rop         = Data_sys['RDATA'][0][:,1,ii]
    # rop         = rop - np.mean(rop,axis=1)[:,np.newaxis]

    ### III. Calculate End Recoding Index (ERI) ###############################################
    eri = np.array([np.nan,np.nan,np.nan,np.nan])  # [exact, empirical]
    for ee in [0,1,2,3]:
        if ee == 0 or ee == 1:
            dat   = rop                # readout axis
        else:
            dat   = chp                # choice axis
        if ee == 0 or ee == 2:         # exact times
            A_1st =  dat[t1,0] 
            G_1st =  dat[t1,-1] 
            A_4th =  dat[t2-1,0] 
            G_4th =  dat[t2-1,-1] 
        else:                           # quarter averages
            A_1st = np.mean( dat[inds_q1,0] )
            A_4th = np.mean( dat[inds_q4,0] )
            G_1st = np.mean( dat[inds_q1,-1] )
            G_4th = np.mean( dat[inds_q4,-1] )
        eri_A = np.abs(A_4th/A_1st)
        eri_G = np.abs(G_4th/G_1st)
        eri[ee]  = np.mean(np.array([eri_A, eri_G]))  # mean of A and G

    #### install output ####
    CH                       = {}        
    CH['chp']                = chp
    CH['rop']                = rop 

    return CH, earliest, eri


def Identify_Oscillatory_Mode( p, A , Setting = -1):

    # Identifies oscillatory mode that has a frequency nearest the half-delay
    # A: linear dynamics matrix
    # 
    # v1, v2:  eigenvectors
    # freq:     oscillatory frequency (units: cycles / delay)

    l, R            = np.linalg.eig(A)      # (cols of evec are R eigvectors)
    L               = np.linalg.inv(R)     # L: left-eigvec matrix (rows are eigvecs)
    freq            = np.imag(l)  * ( p.d1 * p.dt / p.tau) / (2*np.pi)  # cycle / delay
    real            = np.real(l)

    validinds       = freq > 0          # must be oscillatory
    valid_freq      = freq[ validinds ]
    valid_real      = real[ validinds ]
    ind             = np.argmin( np.sqrt( valid_real**2 + (valid_freq - 0.5)**2 ) )  
    ii              = np.where(validinds)[0][ind]
    l_osc           = l[ ii ]
    ind1           = np.isclose( np.imag(l), np.imag(l_osc) )        # mode 1
    ind2           = np.isclose( np.imag(l), np.imag(-l_osc) )       # mode 2
    v1              = np.real( R[ :, ind1 ] )   # osci axis 1
    v2              = np.imag( R[ :, ind2 ] )   # osxi axis 2

    # fastest         = np.argmax( np.abs( freq ) )   
    # l_osc           = l[ fastest ]
    # ind1           = np.isclose( np.imag(l), np.imag(l_osc) )        # mode 1
    # ind2           = np.isclose( np.imag(l), np.imag(-l_osc) )       # mode 2
    # v1              = np.real( R[ :, ind1 ] )   # osci axis 1
    # v2              = np.imag( R[ :, ind2 ] )   # osci axis 2    

    return v1, v2, freq

    


def Calculate_R2(residuals,y):
    # R2 for linear model with no intercept term #
    #
    # residuals:        squared errors 
    # y:                observations [sample,features]   
    #
    R2          = 1 - np.sum( residuals )  / np.sum( np.square(y) )
    return R2


def Plot_Eigenvalues_Comparison( p, A_ref, As ):

    # A_ref: [N,N]           reference matrix
    # As  : [model][N,N]    comparison matrices

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 18})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    # plot #
    fig = plt.figure(constrained_layout=True,figsize=(14,5))
    plt.get_current_fig_manager().window.setGeometry(100,100,1000,500) 

    plt.suptitle('Empirical Linear Dynamics (red: OLS, black: FP linearization)\nDelay period, top PCs', fontsize=24, fontweight='bold')

    for tt in range(len(As)):

        if tt == 0:             # original rnn, non-linear
            title = 'Linear fit (black) to nonlinear data\nFP eigenvalues (red)\n '
        else:                   # linearized
            title = 'Linearized data (sanity check)\n'

        ax = plt.subplot(1,2,1+tt)

        eigval_lsq, _   = np.linalg.eig(As[tt])
        if A_ref is not None:
            eigval, _       = np.linalg.eig(A_ref)  # (cols of evec are R eigvectors)
    

        if A_ref is not None:
            freq   = np.imag(eigval)  * ( p.d1 * p.dt / p.tau) / (2*np.pi)  # cycle / delay
        freq_lsq   = np.imag(eigval_lsq)  * ( p.d1 * p.dt / p.tau) / (2*np.pi)  # cycle / delay

        minreal = -2
        maxreal = 2
        ax.plot([minreal, maxreal], [-0.5, -0.5], ':',color=(0.8,0.8,0.8), alpha=1)
        ax.plot([minreal, maxreal], [+0.5, +0.5], ':',color=(0.8,0.8,0.8), alpha=1)
        ax.plot([minreal, maxreal], [-1, -1], ':',color=(0.5,0.5,0.5), alpha=0.5)
        ax.plot([minreal, maxreal], [+1, +1], ':',color=(0.5,0.5,0.5), alpha=0.5)
        ax.set_ylabel('Oscillation frequency\n(1/delay)',fontsize=16)
        ax.set_xlim([minreal,maxreal])
        ax.set_ylim([-1,1])
        ax.set_yticks(np.arange(-1,1.5,0.5))
        ax.set_yticklabels(['-1','-0.5','0','0.5','1'])
        ax.set_xticks(np.arange(minreal,maxreal+0.5,0.5))
        ax.set_xticklabels(['-2','','-1','','0','','1','','2'])

        
        if A_ref is not None:
            # ax.scatter( np.real(eigval),  freq,   marker='o', color='#ff6c03', alpha=0.8, s=60, linewidth=None)
            ax.scatter( np.real(eigval),  freq,   marker='o', edgecolors='#ff6c03', facecolors='none', alpha=.95, s=80, linewidth=2.5)
        ax.scatter( np.real(eigval_lsq),  freq_lsq,  alpha=0.65, marker='o', facecolors='k', edgecolors='None', s=120 )

        ax.plot([0, 0], [-1, 1], '-',color=(0.8,0.8,0.8))
        ax.plot([-2, 2], [0,0], '-',color=(0.8,0.8,0.8),zorder=0)
        # ax.set_aspect('equal', 'box')
        ax.set_aspect(2/1)    
        ax.set_xlabel('Exponential decay/growth',fontsize=16)
        plt.title('%s' % (title),fontweight='bold',fontsize=12)
    
        # fig.subplots_adjust(hspace=1,vspace=1)    
        fig.tight_layout()    





def newinv(A,b):
    factor = scipy.linalg.cho_factor(A)
    return scipy.linalg.cho_solve(factor, b)

def Mechanism_1_fixedpoints(Fps,showplot = True):

    import itertools

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch


    # Identify slowest saddle
    ind_s = []
    for f in range(Fps.num):
        if np.sum ( np.real( Fps.eigval[f,:] ) > 0 ) == 1 :   # exactly 1 unstable mode
            ind_s = [f]
            break

    # Identify all attractors
    ind_a = []
    for f in range(Fps.num):
        if np.all ( np.real( Fps.eigval[f,:] ) < 0 ) :  # no unstable modes
            ind_a.append(f)

    # Take 0-index fixed point (i.e. oscillatory FP)
    ind_o = [0]

    FPinds     = list( itertools.chain( ind_o, ind_s, ind_a ) )
    Lintypes   = [4,0,0,0]   # how to treat eigenspectrum

    if showplot:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use("Qt5Agg") # set the backend
        matplotlib.rcParams.update({'font.size': 14})
        matplotlib.rcParams['font.sans-serif'] = "Helvetica"
        matplotlib.rcParams['font.family'] = "sans-serif"
        import matplotlib.patches as patch
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(constrained_layout=True,figsize=(14,5))
        plt.get_current_fig_manager().window.setGeometry(300,500,2000,600) 
        clrs = ['r','g','b','m','y','c','slategrey','bisque','darkorange','plum','deepskyblue','salmon','peru','cadetblue','hotpink','k','k','k']

        for f in FPinds:
            if f < 12:
                clr = clrs[f]
            else:
                clr = [.6,.6,.6]
            ax = plt.subplot(2,8,f+1)
            ax.scatter(np.real(Fps.eigval[f,:]), np.imag(Fps.eigval[f,:]),color=clr,s=30)                 # numerical (black)
            ax.plot([0, 0], [-4, 4], '-',color=(0.5,0.5,0.5))
            # ax.set_aspect('equal', 'box')
            plt.xlabel('Real')
            # if f == 0:
            plt.ylabel('Imaginary')
            speed = -np.log10(Fps.q[f])
            plt.title('%d \n %0.1f' % (f,speed),fontweight='bold')
        # fig.subplots_adjust(hspace=1,vspace=1)    
        fig.tight_layout()  

    return FPinds, Lintypes   


def Plot_Polar_Trajectories(p, F, TT, data2, groups_to_plot, x_2_saddle, TT_linearized ):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 22})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    COLOR_EMBED         = 3

    # transform data #
    data2_polar         = nna.polarcoord( data2 )  # [TT][T,N,B] data in polar coordinates (r,phi) #
    if len(x_2_saddle) != 0:
        data2_polar_sp      = nna.polarcoord( data2, center = x_2_saddle[0,:2] )  

    # plot #
    fig = plt.figure( constrained_layout=True, figsize=(14,5) )
    plt.get_current_fig_manager().window.setGeometry(700,20,1800,1200) 
    INDS,_,_               = trial_parser(p)
    
    for Sys in range(3):       # 0: Cartesian, 1: Polar

        if Sys == 0:                # cartesian
            dat = data2
            titlestr = 'Cartesian wrt Oscillatory FP'
        elif Sys == 1:              # polar, osci FP centered
            dat = data2_polar
            titlestr = 'Polar wrt Oscillatory FP'
        elif Sys == 2:              # polar, saddle point centered
            if len(x_2_saddle) == 0:
                break
            else:
                dat = data2_polar_sp
                titlestr = 'Polar wrt Saddle FP'

        for coord in range(2):

            ax                     = plt.subplot(2,3,coord*3+1+Sys)  
            T, t1, t2, _, _, _     = tk.parms_test( p, F, TT_linearized )
            tvec    = range(T)        
            rank_flag            = np.full( p.n_items, False )

            # (for rank analysis later, determine if circular variable)
            if Sys == 0:
                circular = False
            else:
                if coord == 0:
                    circular = False
                else:
                    circular = True

            # trajectories #
            for gg in groups_to_plot:     # Stim pair types
                ii, clr0, mksize, _, _ = groupplot(gg,INDS)
                for LR in [0,1]:
                    for pp in ii[LR]: 

                        clr = embedcolor(pp,p.n_items,COLOR_EMBED)    
                        B = ii[LR].size

                        plt.plot(       tvec[:t2],   dat[TT][:t2,coord,pp],    color=clr[LR], alpha=0.3,linewidth=4)                                # traj  (line) 
                        plt.plot(       tvec[t2:],   dat[TT][t2:,coord,pp],    color=clr0[LR], alpha=0.3)                                # traj  (line) 

                        plt.scatter(    0,      dat[TT][0, coord, pp],      color='k',      marker='.', alpha=0.5, s=mksize  )          # start (ball)
                        plt.scatter(    t1,     dat[TT][t1, coord, pp],   color=clr[LR],  marker='.', s=mksize*2 , zorder=60 )                     # item1 (ball)
                        plt.scatter(    t2-1,   dat[TT][t2-1, coord,pp],   color=clr[LR],  marker='.', s=mksize*2 , zorder=51 )                     # pre-item2 (ball)
                        plt.scatter(    t2,     dat[TT][t2, coord, pp],     color='g',      marker='*', s=100, alpha=0.4, zorder=100)                 # stim2 (star)
                        plt.scatter(    T-1,    dat[TT][-1, coord, pp],     color=clr0[LR], marker='v', s=50)                         # end   (triangle)

                        # print Rank #
                        rank1, _    = ranks(p,pp)
                        if rank_flag[rank1] == False:
                            tmid = t1 + int( p.d1/2 )
                            tqrt = t1 + int( p.d1/4 )
                            plot_Rank1( p, ax, t1 -0.5,   dat[TT][t1,   coord,pp], pp,  fontsize=16, ha='left', zorder=100 )
                            plot_Rank1( p, ax, tqrt-0.5,  dat[TT][tqrt, coord,pp], pp,  fontsize=16, ha='left', zorder=100 )
                            plot_Rank1( p, ax, tmid-0.5,  dat[TT][tmid, coord,pp], pp,  fontsize=16, ha='left', zorder=100 )
                            plot_Rank1( p, ax, t2-0.5,   dat[TT][t2-1, coord,pp], pp,  fontsize=16, ha='left', zorder=100 )
                            rank_flag[rank1] = True

            fontsz = 22
            if Sys == 0:
                if coord == 0:
                    ax.set_ylabel('Oscillatory axis 1', fontsize=fontsz, fontweight='bold')
                else:
                    ax.set_ylabel('Oscillatory axis 2', fontsize=fontsz, fontweight='bold')
            else:
                if coord == 0:
                    ax.set_ylabel('Radius', fontsize=fontsz, fontweight='bold')
                else:
                    ax.set_ylabel('Phase', fontsize=fontsz, fontweight='bold')
            ax.set_xlabel('Time (dt)', fontsize=fontsz, fontweight='bold')
            if coord == 0:
                ax.set_title(titlestr, fontsize=28, fontweight='bold')

            traces = dat[TT][:,coord,:].squeeze()  
            y_min, y_max = ax.get_ylim()

            # Rank ordered periods #
            t_ord = Rank_ordered_times(p, T, t2, traces, circular=circular)
            plt.scatter( np.arange(T)[t_ord], y_max*np.ones((t_ord.size)), marker='.', s=300, color='g'  )

            # Choices separated periods #        
            # t_sep = Choice_separated_times(p, T, t2, traces, circular=circular)
            # plt.scatter( np.arange(T)[t_sep], y_max*np.ones((t_sep.size)), marker='.', s=300, color='k'  )

            if Sys > 0 and coord == 0:  # radius plot
                ax.set_ylim([-5,50])

            if Sys > 0 and coord == 1:  # radius plot
                ax.set_ylim([-np.pi-1,np.pi+1])
                ax.plot([0,T],[0,0],color=(.5,.5,.5),alpha=0.3)
                ax.plot([0,T],[np.pi,np.pi],color=(.5,.5,.5),alpha=0.3)
                ax.plot([0,T],[-np.pi,-np.pi],color=(.5,.5,.5),alpha=0.3)
                ax.plot([0,T],[np.pi/2,np.pi/2],':',color=(.5,.5,.5),alpha=0.3)
                ax.plot([0,T],[-np.pi/2,-np.pi/2],':',color=(.5,.5,.5),alpha=0.3)


    fig.subplots_adjust(hspace=1)
    plt.tight_layout()

    # # %% Plot grid of Phases #
    # ti = reload(ti)
    # TT = -1
    # plot_phases_grid(p,dat,F['train_table'],INDS,F['X_input'],TT)


def Plot_1D_Projection(p, F, groups_to_plot, rdata, data1, data1_2D, TT_linearized):

    from importlib import reload  
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.use("Qt5Agg") # set the backend
    mpl.rcParams.update({'font.size': 14})
    mpl.rcParams['font.sans-serif'] = "Helvetica"
    mpl.rcParams['font.family'] = "sans-serif"
    import matplotlib.patches as patch
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches   as patch

    COLOR_EMBED         = 3

    fig = plt.figure( constrained_layout=True, figsize=(14,5) )
    plt.get_current_fig_manager().window.setGeometry(300,20,600,2200) 
    INDS,_,_               = trial_parser(p)
    
    for Sys in range(5):      

        if Sys >= 2 and len(data1_2D) == 0:
            break

        if Sys == 0:
            titlestr = 'Non-linear\nND output'
            TT = TT_linearized
            dat = rdata            

        elif Sys == 1:

            titlestr = 'Linearized model\n Trained linear readout'
            TT = -1     
            dat = rdata

        elif Sys == 2:         
            titlestr = 'Non-linear\nND saddle'
            TT = TT_linearized     
            dat = data1

        elif Sys == 3:             
            titlestr = 'Non-linear\nOscillatory (2D) saddle'
            TT = TT_linearized      
            dat = data1_2D
            # # filter the data in the oscillatory basis #
            # for t in range(dat[TT].shape[0]):
            #     dat[TT][t,:,:] =  oscbasis.filter( dat[TT][t,:,:].transpose(), center=True ).transpose()
        
        elif Sys == 4:             
            titlestr = 'Linearized\nOscillatory (2D) saddle'
            TT = -1      
            dat = data1_2D



        ax                   = plt.subplot(5,1,Sys+1)  
        T, t1, t2, _, _, _   = tk.parms_test( p, None, TT_linearized )
        tvec                 = range(T)
        rank_flag            = np.full( p.n_items, False )

        # trajectories #
        for gg in groups_to_plot:     # Stim pair types
            ii, clr0, mksize, _, _ = groupplot(gg,INDS)
            for LR in [0,1]:
                for pp in ii[LR]: 

                    clr = embedcolor(pp,p.n_items,COLOR_EMBED)    
                    B = ii[LR].size
                    repvec = np.ones(B)

                    plt.plot(       tvec[:t2],   dat[TT][:t2,0:1,pp],    color=clr[LR], alpha=0.3, linewidth=4)   
                                                 # traj  (line) 
                    plt.plot(       tvec[t2:],   dat[TT][t2:,0:1,pp],    color=clr0[LR], alpha=0.3)                                # traj  (line) 

                    plt.scatter(    0,      dat[TT][0, 0:1, pp],      color='k',      marker='.', alpha=0.5, s=mksize  )          # start (ball)
                    plt.scatter(    t1,     dat[TT][t1, 0:1, pp],   color=clr[LR],  marker='.', s=mksize*2 , zorder=60 )                     # item1 (ball)
                    plt.scatter(    t2-1,   dat[TT][t2-1, 0:1,pp],   color=clr[LR],  marker='.', s=mksize*2 , zorder=51 )                     # pre-item2 (ball)
                    plt.scatter(    t2,     dat[TT][t2, 0:1, pp],     color=[.5,.5,.5],      marker='*', s=100, alpha=0.4, zorder=100)                 # stim2 (star)
                    plt.scatter(    T-1,    dat[TT][-1, 0:1, pp],     color=clr0[LR], marker='v', s=50)                         # end   (triangle)

                    # print Rank #
                    rank1, _    = ranks(p,pp)
                    if rank_flag[rank1] == False:
                        tmid = t1 + int( p.d1/2 )
                        tqrt = t1 + int( p.d1/4 )
                        # plot_Rank1( p, ax, t1 -0.5,   dat[TT][t1,   0,pp], pp,  fontsize=16, ha='left', zorder=100 )
                        # plot_Rank1( p, ax, tqrt-0.5,      dat[TT][tqrt, 0,pp], pp,  fontsize=16, ha='left', zorder=100 )
                        # plot_Rank1( p, ax, tmid-0.5,      dat[TT][tmid, 0,pp], pp,  fontsize=16, ha='left', zorder=100 )
                        # plot_Rank1( p, ax, t2-0.5,   dat[TT][t2-1, 0,pp], pp,  fontsize=16, ha='left', zorder=100 )
                        rank_flag[rank1] = True

                    # 0-line
                    plt.plot(       [0,T], [0,0],    color=(.8,.8,.8), alpha=0.2)                                # traj  (line) 


        traces = dat[TT][:,0,:].squeeze()  # [t,pp]
        y_min, y_max = ax.get_ylim()

        # Rank ordered periods #
        t_ord = Rank_ordered_times(p, T, t2, traces, circular=False)
        plt.scatter( np.arange(T)[t_ord], y_max*np.ones((t_ord.size)), marker='.', s=300, color='g'  )

        # Choices separated periods #        
        t_sep = Choice_separated_times(p, T, t2, traces, circular=False)
        plt.scatter( np.arange(T)[t_sep], y_max*np.ones((t_sep.size)), marker='.', s=300, color='k'  )

        ax.set_xlabel('Time (dt)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Projection', fontsize=20, fontweight='bold')
        ax.set_title(titlestr, fontsize=20, fontweight='bold')

    fig.subplots_adjust(hspace=.5)
    plt.tight_layout()

    # # %% Plot grid of Phases #
    # ti = reload(ti)
    # TT = -1
    # plot_phases_grid(p,dat,F['train_table'],INDS,F['X_input'],TT)



def Rank_ordered_times(p, T, t2, traces, circular=False):
    # reports times in which the Rank of traces is circularly ordered
    # traces: [time,pp]   pp is the Transitive Inference trial index  
    rank1vec,_  = Rankvector(p)
    ranktraces  = np.full( (p.n_items, T), np.nan )
    for r in range(p.n_items):
        matchtraces   = traces[:, rank1vec==r]   # all traces with this rank1 -- will only be equal up to t1
        ranktraces[r,:] = matchtraces[:,0]          # just take the first matched trace
        # determine whether ordered (circularly)
    wellordered = np.zeros((T))
    for t in range(0,t2):
        slice = ranktraces[np.newaxis,:,t]
        if np.unique(slice).size < p.n_items:
            continue
        sliceorder  = np.argsort( slice )
        sliceorders = sliceorder
        if circular:
            for rr in range(p.n_items):  # circularly permute
                sliceorders = np.concatenate( (sliceorders, np.roll(sliceorder, rr+1)), axis=0)
        if np.any( ( sliceorders==np.arange(p.n_items) ).all(axis=1) )  or \
            np.any(( sliceorders==np.flip(np.arange(p.n_items)) ).all(axis=1)) :
            wellordered[t] = 1
    t_ord = np.where(wellordered)[0]
    return t_ord

def Choice_separated_times(p, T, t2, traces, circular=False):
    # returns times in which choices are perfectly separated
    _, _, _, _, _, ztarget     = tk.parms_test(p, None, 0)
    choice0 = traces[:,ztarget.squeeze()==0]  
    choice1 = traces[:,ztarget.squeeze()==1] 
    separation = np.zeros((T))

    if circular==False:             # linearly separated
        for t in range(t2,T):
            max0 = np.max(choice0[t,:])
            min0 = np.min(choice0[t,:])
            max1 = np.max(choice1[t,:])
            min1 = np.min(choice1[t,:])
            if (max0 > max1 and min0 > max1) or \
                (max1 > max0 and min1 > max0):
                separation[t] = 1
        t_sep = np.where(separation)[0]

    elif circular==True:            # circularly separated (phase)
        for t in range(t2,T):
            # concatenate phase values with choice vector
            num0 = choice0[t,:].size
            num1 = choice1[t,:].size
            v0 = np.concatenate( (choice0[t,:,np.newaxis], np.zeros((num0,1))) , axis=1)
            v1 = np.concatenate( (choice1[t,:,np.newaxis], np.ones((num1,1))) , axis=1)
            V           = np.concatenate( (v0,v1), axis = 0  )
            # sort by phase #
            ind         = np.argsort(V[:,0])
            diffvec     = np.diff( V[ind,1] )
            numswitches = np.ceil( np.sum( np.abs(diffvec) ) / 2 )
            if numswitches < 2:
                separation[t] = 1
        t_sep = np.where(separation)[0]

    return t_sep



