import sys
import numpy as np
import torch
import matplotlib as mpl 
import matplotlib.pyplot as plt

import task_train_test as tk
import nn_analyses     as nna
import ti_functions    as ti

def printparms():

   njobs = singleout(TID=0).njobs  # identify total number of jobs
   dims  = singleout(TID=0).dims   # identify dimensions (coordinates)
   matrix = np.zeros(dims)
   Dict = dict(vars(params()))

   for TID in range(njobs):
      subs = np.unravel_index(TID, matrix.shape, 'C')  # Convert linear index to coord index // 'C', i.e. row-major   
      string = ''
      for i, key in enumerate( Dict['scan'].keys() ) :
         val     = Dict['scan'][key][subs[i]]
         string = string + '%s: %g ' % (key,val) 
      print( 'TID: %d %s' % (TID, string) ) 


def read_model_performance(jobname, localdir = '',  TT_test = [0], Model_select = -1, verbose = True, only_perfect=False):

   # (preliminary) #
   import sys
   sys.path.insert(0, sys.path[0] + jobname)
   from init import singleout, params, genfilename
   pdict = dict(vars(params()))
   p = singleout()                         # instantiate simply to access .dims conveniently

   print('(%s) Getting all models\' performance' % (localdir + p.jobname))

   # i.  Initialize output measures #
   loss_min          = np.full(p.dims, np.nan, dtype='float64')
   perf_train        = np.full(p.dims, np.nan, dtype='float64')
   perf_test         = np.full(p.dims, np.nan, dtype='float64')
   TIDs_dims         = np.full(p.dims, np.nan, dtype='float64')
   # myObservable    = np.full(p.dims, np.nan, dtype='float64')


   # ii. Get stored values from all models ####
   matrix         = np.zeros(p.dims)     # dummy matrix of parameter slots
   TIDs           = np.array([])    # TIDs of models analyzed
   input_trains   = np.array([])
   RTs            = []              # RT values 
   Choicevals     = []              # Choice values
   Delaypercs     = np.array([])

   for TID in range(p.njobs):

      p_TID             = singleout(TID = TID)  # get parms object
      fname             = genfilename(p_TID,localdir)
      file              = tk.readfile(fname,".p",silent=True)
      if len(file) == 0:
         print('(no file) TID: %d || %s' % (TID,fname))
         continue
      F                 = file[0]  # retrieve model
      subs              = np.unravel_index( TID, matrix.shape, 'C')  # coordinates of model 
      if 'epoch' in F:  # new file format  
         loss_min[subs]    = F['log_train_losses'][F['epoch'],0]   # get value
      else:             # old file format
         loss_min[subs]    = F['loss_min']   # get value

      # Identify model 
      model, model_ep, i, _ = tk.select_model( p, F, Model_select )
      
      #  Get performance #
      Train = np.full(len(TT_test),np.nan)  # [tt]
      Test  = np.full(len(TT_test),np.nan)  # [tt]

      # (if calculated before)
      if 'performance' in F.keys():   
         Train  =  F['performance'][i,TT_test,0]
         Test   =  F['performance'][i,TT_test,1]
      if 'performance_vd' in F.keys(): 
         DELAY_PERC = 100
         trainperfs = F['performance_vd'][i,0,:,1]  # [t]  t: time (in timesteps) shortened from full delay
         testperfs  = F['performance_vd'][i,1,:,1]  # [t]
         for t in range(p.d1):
            if trainperfs[t] != 100 or testperfs[t] != 100:
               DELAY_PERC = 100 * t/p.d1    # percentage of delay for which the performance is still correct
               break
      else: 
         DELAY_PERC = -1  # dummy value if not available

      # tk.print_model_performance( p, model, F['X_input'], TT = 0, silent = False )   

      ### Simulate model to obtain Performance, Choice, RTs ###
      for tt in range(len(TT_test)):   # 0: default, 6: 1st item max-forward, 7: 2nd item max-back
         TT = TT_test[tt]
         _, _, _, X, input, _     = tk.parms_test(p,F,TT)
         with torch.no_grad():
     
            # (if not previously calculated) get performance #
            if np.any(np.isnan(Train)):
               train,test            = tk.print_model_performance(p, model, F['X_input'], TT_test, silent = True)
               Train[tt]             = train
               Test[tt]              = test     

            # get RT and quantitative choice output #
            D                     = tk.run_model( model, p, input, hidden_output = True )
            _, Choiceval, RT      = tk.parse_output( p, D['Out_z'] )
            Choicevalmat          = Choiceval.reshape(p.n_items, p.n_items)  
            if p.T > 1:
               RTmat                 = RT.reshape( p.n_items, p.n_items ) 
            else:
               RTmat                = None

      TRAIN        = np.mean(Train)    # combined train performance
      TEST         = np.mean(Test)     # combined test performance

      perf_train[subs]      = TRAIN  
      perf_test[subs]       = TEST   
      TIDs_dims[subs]       = TID

      if verbose:
         if only_perfect == 1:
            if TEST != 100 or TRAIN != 100:
               continue
         elif only_perfect == 2:
            if TRAIN != 100:
               continue
         string = ''
         for i, key in enumerate( pdict['scan'].keys() ) :
            val     = pdict['scan'][key][subs[i]]
            string = string + '%s: %g ' % (key,val) 
         print( 'TID: %d || %s|| Train: %d Test: %d || Delay: %d  || (ep: %d)' % (TID, string, TRAIN, TEST, DELAY_PERC, model_ep) )

      # Save other outputs #
      TIDs              = np.append(TIDs, int(TID))
      input_trains      = np.append(input_trains, p_TID.input_train)      
      Delaypercs        = np.append(Delaypercs, DELAY_PERC )
      # Choicevals.append(  ti.List_Mirror_Values(p, Choicevalmat, signflip=True)  ) 
      Choicevals.append(   Choicevalmat )  
      if p.T > 1:
         # RTs.append(      ti.List_Mirror_Values(p, RTmat, signflip=False)        ) 
         RTs.append(      RTmat         ) 
      
   # output #
   Perfs = {}
   Perfs['loss_min']      = loss_min
   Perfs['perf_train']    = perf_train
   Perfs['perf_test']     = perf_test
   Perfs['TIDs']          = TIDs
   Perfs['TIDs_dims']     = TIDs_dims
   Perfs['input_trains']  = input_trains
   Perfs['Choicevals']    = Choicevals
   Perfs['RTs']           = RTs
   Perfs['Delaypercs']    = Delaypercs
   
   return Perfs


def collapse(x,dim):
   # input:
   #  x    N-dimensional array  (dim1,dim2,...dimN)
   #  dim  dimension to maintain
   # output:
   #  y    2-dimensional array  (dim,prod(collapsed_dims))
   
   size_u  = x.shape[dim]                   # size of uncollapsed dimension
   xnew    = np.swapaxes(x,0,dim)          # re-ordered data
   size_c  = np.product(xnew.shape[1:])     # size of collapsed dimension
   y       = np.reshape(xnew, ( size_u , size_c ), order='C' ) 

   return y

def plot_allmodels_perf(loss_min, perf_train, perf_test, independent_dim):

   import matplotlib as mpl 
   import matplotlib.pyplot as plt
   mpl.use("Qt5Agg") # set the backend  
   mpl.rcParams.update({'font.size': 18})
   mpl.rcParams['font.sans-serif'] = "Helvetica"
   mpl.rcParams['font.family'] = "sans-serif"
   import _tkinter  


   # (prelim) Identify independent dim's values #
   xlabel  = list(dict(vars(params()))['scan'].keys())[independent_dim]
   xvalues = list(dict(vars(params()))['scan'].values())[independent_dim]

   ###  Plot ######
   fig0 = plt.figure()
   plt.show(block = False)    # prevents busy cursor
   plt.get_current_fig_manager().window.setGeometry(80,100,700,1000) 
   fig0.suptitle('%s models' % (params().jobname), fontsize=24, weight='bold')

   ALPHA = 0.3

   # 1.  Training loss #
   plt.subplot(311)
   data     = collapse(loss_min,independent_dim)
   for x in range(data.shape[0]):
      jit = width*np.random.rand(data.shape[1]) - width*0.5
      plt.scatter(x+jit,data[x,:],alpha=ALPHA,color='k')
   plt.xticks(range(data.shape[0]),xvalues)
   plt.ylabel('Loss',fontsize=20)

   # 2.   Train Performance #
   plt.subplot(312)
   data     = collapse(perf_train,independent_dim)
   for x in range(data.shape[0]):
      jit = width*np.random.rand(data.shape[1]) - width*0.5
      plt.scatter(x+jit,data[x,:],alpha=ALPHA,color='k')
   plt.ylim([0,102])
   plt.xticks(range(data.shape[0]),xvalues)
   plt.ylabel('Performance (Train)',fontsize=20)

   # 3.   Test Performance #
   plt.subplot(313)
   data     = collapse(perf_test,independent_dim)
   for x in range(data.shape[0]):
      jit = width*np.random.rand(data.shape[1]) - width*0.5
      plt.scatter(x+jit,data[x,:],alpha=ALPHA,color='k')
   plt.ylim([0,102])
   plt.xticks(range(data.shape[0]),xvalues)
   plt.xlabel(xlabel,fontsize=20)
   plt.ylabel('Performance (Test)',fontsize=20)

   plt.show()
   plt.ion()
   plt.ioff()
   plt.ion()

def report_allmodels_perf( params, perf_train, perf_test ):

   # (prelim) Identify independent dim's values #
   numdims = len(list(dict(vars(params()))['scan'].keys()))

   for independent_dim in range(numdims-1):
      
      xlabel  = list(dict(vars(params()))['scan'].keys())[independent_dim]
      xvalues = list(dict(vars(params()))['scan'].values())[independent_dim]

      trains  = collapse(perf_train, independent_dim)
      tests   = collapse(perf_test,  independent_dim)

      print('------------------------------------------' )      
      
      for xx in range(trains.shape[0]):
         numperf = np.sum(np.logical_and( trains[xx,:]==100,tests[xx,:]==100 ))
         print('%s = %g: %d of %d models perfect performers' % (xlabel, xvalues[xx], numperf, trains.shape[1]))


def Behavior_Plot_Variants( p, taskfunc, params, TID_variant, Behaviors , Variant_name, Variants_to_plot=range(4), Num_models_plot=-1):

   numdims = len(list(dict(vars(params()))['scan'].keys())) # (prelim) Identify independent dim's values #

   for independent_dim in range(numdims-1):
      xlabel  = list(dict(vars(params()))['scan'].keys())[independent_dim]
      xvalues = list(dict(vars(params()))['scan'].values())[independent_dim]
      print(xlabel)
      if xlabel == Variant_name:
         TID_matrix    = collapse(TID_variant, independent_dim).astype(np.int32)  # [variant,seed]
         break

   numvar = TID_matrix.shape[0]      
   for vv in Variants_to_plot:
      wshift = 100*vv
      num_models_available = TID_matrix[vv,:].size
      if Num_models_plot == -1:
         subinds       = np.arange(num_models_available)
      else:  #subsample
         subinds       = np.random.choice( np.arange(num_models_available), size=Num_models_plot, replace=False)
      taskfunc.Behavior_plot( p, Behaviors , TID_matrix[vv,subinds], windowshift=wshift)

   

def read_performance(localdir = '', TT_test = [0], Model_select = -1, verbose = True, only_perfect=False):

   # (preliminary) #
   p = singleout()                         # instantiate simply to access .dims conveniently
   sys.path.insert(0, localdir[:-1] + jobname)
   pdict = dict(vars(params()))

   print('(%s) Getting all models\' performance' % (localdir + p.jobname))
   print('Stimulus conditions (TT): %s' % str(TT_test) )
   matrix = np.zeros(p.dims)  # dummy matrix of parameter slots

   for TID in range(p.njobs):

      p_TID             = singleout(TID = TID)  # get parms object
      fname             = genfilename(p_TID,localdir)
      file              = tk.readfile(fname,".p",silent=True)
      if len(file) == 0:
         # print('(no file) TID: %d || %s' % (TID,fname))
         continue
      F                 = file[0]  # retrieve model
      subs              = np.unravel_index( TID, matrix.shape, 'C')  # coordinates of model 

      numfrozen         = len(F['models_frozen'])
      M                 = numfrozen + 2   # total no. of models (all frozen models + the lowest loss + "best" model

      Perf              = F['performance'][:,TT_test,:]    # [modelnum,TT,<train,test>]   output

      # print((M,Perf.shape[0]))

      # iii. Test performance #
      Train = np.full(len(TT_test),np.nan)  # [TT]
      Test  = np.full(len(TT_test),np.nan)

      for tt in range(len(TT_test)):

         if Model_select == -1:     # best model
            model = F['model']
            epoch = F['epoch']
            train = Perf[-2,tt,0]
            test  = Perf[-2,tt,1]
         elif Model_select == -2:   # lowest-loss model
            model =  F['model_lowest']     
            epoch =  F['epoch_lowest']    
            train = Perf[-1,tt,0]
            test  = Perf[-1,tt,1]
         elif Model_select == -3:   # choose model w/ minimum 0.1 task loss
            losses         = F['log_train_losses']
            numfrozen      = len(F['models_frozen'])
            epochs_frozen  = np.arange(0,numfrozen*p.nEpochsfreeze,p.nEpochsfreeze)
            taskprop       = losses[epochs_frozen,1] / losses[epochs_frozen,0]
            inds           = taskprop > 0.1
            tasklosses     = losses[epochs_frozen[inds],1]
            i              = np.argmin(tasklosses)
            epoch          = epochs_frozen[i]
            model          = F['models_frozen'][i]         
         elif Model_select == -4:

            model = F['model_earliest']
            epoch = F['epoch_earliest']
            train = 100
            test  = 100

         elif Model_select == -5:      # earliest frozen model that performs perfectly

            perfinds = np.where( np.all( Perf == 100, axis=(1,2) ) )[0]  # perfect across stimulus conditions + train/test

            if perfinds.size > 0:          # if there is perfect performing model
               i     = np.min(perfinds)    # earliest perfect performance
            else:                          # otherwise, take model w/ best performance
               i     = np.argmax( np.mean( Perf, axis=(1,2) )   )   # best performance
            train = Perf[i,tt,0]
            test  = Perf[i,tt,1]
            if i == numfrozen:         # best model
               epoch = F['epoch']
            elif i == numfrozen + 1:   # last model
               epoch = F['epoch_lowest']
            else:
               epoch = i * p.nEpochsfreeze 

         else:

            index = int( Model_select / p.nEpochsfreeze )
            model = F['models_frozen'][index] 
            epoch = index*p.nEpochsfreeze    

         Train[tt]             = train
         Test[tt]              = test     
               
      TRAIN        = np.mean(Train)    # average train performance
      TEST         = np.mean(Test)     # average test performance

      if verbose:
         if only_perfect:
            if TEST != 100 or TRAIN != 100:
               continue
         string = ''
         for i, key in enumerate( pdict['scan'].keys() ) :
            val     = pdict['scan'][key][subs[i]]
            string = string + '%s: %g ' % (key,val) 
         print( 'TID: %d || %s|| Train: %d Test: %d (ep: %d)' % (TID, string,TRAIN,TEST,epoch) ) 

def plot_pca_1D(localdir = '', TT_plot = 0, TT_test = [0], Model_select = -1, verbose = True, only_perfect=False): 

   # (preliminary) #
   p = singleout()                         # instantiate simply to access .dims conveniently
   sys.path.insert(0, localdir[:-1] + jobname)
   pdict = dict(vars(params()))

   print('(%s) Getting all models\' performance' % (localdir + p.jobname))
   print('Stimulus conditions (TT): %s' % str(TT_test) )
   matrix = np.zeros(p.dims)  # dummy matrix of parameter slots

   for TID in range(p.njobs):

      p_TID             = singleout(TID = TID)  # get parms object
      fname             = genfilename(p_TID,localdir)
      file              = tk.readfile(fname,".p",silent=True)
      if len(file) == 0:
         # print('(no file) TID: %d || %s' % (TID,fname))
         continue
      F                 = file[0]  # retrieve model
      subs              = np.unravel_index( TID, matrix.shape, 'C')  # coordinates of model 

      numfrozen         = len(F['models_frozen'])
      M                 = numfrozen + 2   # total no. of models (all frozen models + the lowest loss + "best" model

      Perf              = F['performance'][:,TT_test,:]    # [modelnum,TT,<train,test>]   output

      # print((M,Perf.shape[0]))

      # iii. Test performance #
      Train = np.full(len(TT_test),np.nan)  # [TT]
      Test  = np.full(len(TT_test),np.nan)

      for tt in range(len(TT_test)):

         if Model_select == -1:     # best model
            model = F['model']
            epoch = F['epoch']
            train = Perf[-2,tt,0]
            test  = Perf[-2,tt,1]
         elif Model_select == -2:   # lowest loss model
            model =  F['model_lowest']     
            epoch =  F['epoch_lowest']    
            train = Perf[-2,tt,0]
            test  = Perf[-2,tt,1]
         elif Model_select == -3:   # choose model w/ minimum 0.1 task loss
            losses         = F['log_train_losses']
            numfrozen      = len(F['models_frozen'])
            epochs_frozen  = np.arange(0,numfrozen*p.nEpochsfreeze,p.nEpochsfreeze)
            taskprop       = losses[epochs_frozen,1] / losses[epochs_frozen,0]
            inds           = taskprop > 0.1
            tasklosses     = losses[epochs_frozen[inds],1]
            i              = np.argmin(tasklosses)
            epoch          = epochs_frozen[i]
            model          = F['models_frozen'][i]               
         elif Model_select == -4:      # choose best performing model
            epoch          = F['epoch_earliest']
            model          = F['model_earliest']
            train          = Perf[-1,tt,0]
            test           = Perf[-1,tt,1]            
         else:
            index = int( Model_select / p.nEpochsfreeze )
            model = F['models_frozen'][index] 
            epoch = index*p.nEpochsfreeze    

         Train[tt]             = train
         Test[tt]              = test     
               
      TRAIN        = np.mean(Train)    # average train performance
      TEST         = np.mean(Test)     # average test performance

      if verbose:
         if only_perfect:
            if TEST != 100 or TRAIN != 100:
               continue
         string = ''
         for i, key in enumerate( pdict['scan'].keys() ) :
            val     = pdict['scan'][key][subs[i]]
            string = string + '%s: %g ' % (key,val) 
         print( 'TID: %d || %s|| Train: %d Test: %d (ep: %d)' % (TID, string,TRAIN,TEST,epoch) ) 

      # plot pca #
      YSCALE  = 20
      Data, DataR, Datatag, Xs = tk.run_model_get_data(p, model, F['X_input'] )  # (!!) running all conditions
      pca                      = nna.PCA_Calc( Data, Reference_TT = 0 )
      pdata, rdata             = nna.PCA_Proj( p, Data, DataR, Xs, pca)  # calc PCA in default condition (TT = 0)      
      ti.PC_1D_plot(p,pdata,rdata,[TT_plot],YSCALE,title='TID ' + str(TID))
      plt.ion()
      plt.ioff()
      plt.ion()      
 
def report_delay_percentages( TIDs, Delaypercs, numtoprint=3 ):

   # print most + least robust #
   inds = np.argsort( Delaypercs )
   print('Most Delay robust:')
   if inds.size < numtoprint:
      numtoprint = inds.size
   for mm in range(1,1+numtoprint):
      TID         = TIDs[inds[-mm]]
      Delayperc   = Delaypercs[inds[-mm]] 
      print('Delay percentage: %d (TID: %d)' % (Delayperc,TID))
   print('Least Delay robust:')
   for mm in range(0,numtoprint):
      TID         = TIDs[inds[mm]]
      Delayperc   = Delaypercs[inds[mm]] 
      print('Delay percentage: %d (TID: %d)' % (Delayperc,TID))

   #  histogram #
   fig0 = plt.figure()
   plt.show(block = False)    # prevents busy cursor
   plt.get_current_fig_manager().window.setGeometry(300,500,500,600) 
   # mngr = plt.get_current_fig_manager()
  
   minval   = -5
   maxval   =  105
   binsize  =  1

   bin_edges   = np.arange(minval, maxval, binsize)

   vals        = Delaypercs
   ax = fig0.add_subplot(1,1,1)
   ax.set_title('Delay robustnesses', fontsize=20)
   histout = ax.hist(vals, bins=bin_edges)  # arguments are passed to np.histogram  
   ax.axis(xmin=minval,xmax=maxval)
   ax.set_xlabel('Delay percentage')
   ax.set_ylabel('Number of models')

   fig0.subplots_adjust(hspace=.1)

   plt.draw()     
   plt.show()
   plt.pause(0.0001)      

   plt.ion()
   plt.ioff()
   plt.ion()

def plot_mirror_distributions( TIDs, value_list, titlestring, max, bin, MINPROP = 0.5):

   fig0 = plt.figure()
   plt.show(block = False)    # prevents busy cursor
   plt.get_current_fig_manager().window.setGeometry(80,100,2200,2000) 
   fig0.suptitle('%s' % (titlestring), fontsize=30, weight='bold')
   # mngr = plt.get_current_fig_manager()
  
   bin_edges   = np.arange(-max, max, bin)

   numplot = TIDs.size
   counter = 0
   for pp in range(numplot):  #range(numplot):

      vals        = value_list[pp]
      vals_diag   = np.diff(vals, axis=1).ravel()
      sd          = np.std(vals_diag)

      if counter == 100:
         break
      else:
         counter += 1

      ax0 = fig0.add_subplot(8,15,counter)
      
      ax0.set_title('%g \n sd = %0.2f' % (TIDs[pp],sd), fontsize=14)
      ax0.hist(vals_diag, bins=bin_edges)  # arguments are passed to np.histogram  
      ax0.axis(xmin=-max,xmax=+max,ymin=0,ymax=10)
      ax0.plot([max * MINPROP, max * MINPROP],[0,10],'r')
      ax0.plot([-max * MINPROP,-max * MINPROP],[0,10],'r')
      if np.all( np.abs(vals_diag) < max * MINPROP ):
         ax0.set_facecolor((.8,.8,.8))
      # ax0.set_ylabel('# of pairs')

   fig0.subplots_adjust(hspace=1)

   plt.draw()     
   plt.show()
   plt.pause(0.0001)      

   plt.ion()
   plt.ioff()
   plt.ion()

def print_best_models( TIDs, RTs, input_trains, Delaypercs, Delay, MIN_DELAY_PROP, MAX_ASYMMETRY, numtoprint=10 ):

   M                     = TIDs.size

   inds_1                = np.full((M),False)
   inds_2                = np.full((M),False)    

   # i. Get models that satisfy minimum RT symmetry #
   numpairs              = RTs[0].shape[0]
   Diffs                 = np.full((M,numpairs),np.nan)
   for mm in range(M):  #range(numplot):
      diffs       = np.diff(RTs[mm], axis=1).ravel() / Delay  # all pairwise differences
      Diffs[mm,:] = diffs  # store for later sorting 
      if np.all( diffs < MAX_ASYMMETRY ):
         inds_1[mm] = True
         
   # ii. Get models that satisfy minimum Delay robustness #
   inds_2 = ( Delaypercs >= MIN_DELAY_PROP*100 )

   # Filter for models satisfying both #
   inds     = np.logical_and(inds_1,inds_2)

   for II in range(2):
      
      Inds = np.logical_and( input_trains == II, inds )

      M_min    = np.sum(Inds)    
      TID_min  = TIDs[Inds]      # [model]
      Diffvals = Diffs[Inds,:]   # [model,pair]

      # iv.  Rank models by smallest asymmetry #
      sort_rt     = np.argsort( np.std(Diffvals, axis=1) ) # ascending 

      # iii. Rank models by delay robustness #
      sort_delay  = np.argsort( -Delaypercs[Inds] )

      ####### print report ########
      if M_min < numtoprint:
         Numprint = M_min
      else:
         Numprint = numtoprint

      print('%d of %d models (input_train: %d) meet minimum criteria' % (M_min, M, II) )

      print('-------Most Symmetric---------')
      for m in range(Numprint):
         i           = sort_rt[m]
         TID         = TID_min[ i ]
         rt_sd       = np.std(  Diffvals[ i, : ]  )
         delayperc   = Delaypercs[ TIDs == TID ] 
         print('RT diff SD: %d (Delay perc: %d) (TID: %d)' % ( rt_sd * 100, delayperc ,TID))

      print('-------Most Delay-robust-------')
      for m in range(Numprint):
         i           = sort_delay[m]
         TID         = TID_min[ i ]
         rt_sd       = np.std(  Diffvals[ i, : ]  )
         delayperc   = Delaypercs[ TIDs == TID ] 
         print('Delay %%: %d (RT diff SD: %d) (TID: %d)' % ( delayperc, rt_sd *100 , TID))



   







