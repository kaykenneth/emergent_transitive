import numpy as np
import task_train_test as tk
import nn_analyses     as nna


###### Fixed-point finding #################

def run_FPF_search(p, F, earliest_only = False):

    # 1. Set Parms #
   add_rand_seeds     = True #True    # (optional) get random trials for seeds
   rand_numbatch      = 5       #  " " no. of batches to run  
   rand_batchsize     = 50      #  " " no. of seeds / batch
   numeps             = 50000  # no. of optimization epochs per batch
   numeps_stop        = 5000   # no. optimization epochs for improvement before stopping

   # 2. Run FPF #

   fps       = []

   if earliest_only:    # only calculate FP for the earliest perfect model

      if 'model_earliest' in F.keys():
         model = F['model_earliest']  
      else:   # if perfect performer never was trained, default to 'best' model
         model = F['model']
      fp   = nna.FPF_search(model,p, F['X_input'], numepochs=numeps,numepochs_stop=numeps_stop,\
                           add_rand_seeds=add_rand_seeds,rand_numbatch=rand_numbatch,rand_batchsize=rand_batchsize)
      fps.append(fp)      

   else:

      numfrozen = len( F['models_frozen'] )
      nummodels = numfrozen + 3   

      for mm in range(nummodels):
         if mm == numfrozen:              # best model
            model = F['model']
         elif mm == numfrozen + 1:        # lowest loss model
            model = F['model_lowest']  
         elif mm == numfrozen + 2:        # earliest model
            model = F['model_earliest']  
         else:                            # all other frozen models
            model = F['models_frozen'][mm]         

      fp   = nna.FPF_search(model,p, F['X_input'], numepochs=numeps,numepochs_stop=numeps_stop,\
                           add_rand_seeds=add_rand_seeds,rand_numbatch=rand_numbatch,rand_batchsize=rand_batchsize)

      fps.append(fp)
      
   return fps

def run_FPF_detect(p, F, fps, pca = []):

   # 1. Parameters #
   tol_q              = 3          # speed in -log10
   tol_unique         = 0.00001           # distance or activation difference for eliminating redundant Fixed Pts.
   detect_baseline    = True   # (optional) detect FP nearest the baseline, i.e. time of item 1
   detect_delay       = False   # (optional) detect FP nearest the baseline, i.e. time of item 1

   # 2. Run FPF #
   numfrozen = len(F['models_frozen'])
   nummodels = numfrozen + 2   # all frozen models + the lowest loss + "best" model
   Fps       = []
   for mm in range(nummodels):
      if mm == numfrozen:              # best model
         model = F['model']
      elif mm == numfrozen + 1:        # lowest loss model
         model = F['model_lowest']         
      else:                            # frozen model
         model = F['models_frozen'][mm]         
      Fp    = nna.FPF_detect(fps[mm], model, p, F['X_input'], pca, tol_q=tol_q, tol_unique=tol_unique, \
                           detect_baseline = detect_baseline, detect_delay = detect_delay)
      # 3.   For each FP, identify unstable modes #
      Fp         = nna.find_unstable_modes(Fp)      # identify unstable modes + update Fps object

      Fps.append(Fp)

   return Fps


###### Performance #################

def run_Performance(p, F):

   if p.T == 1:
      TT_list = range(1)
   elif p.T > 1:
      TT_list = range(16)

   # 1. Determine # of models to check #
   numfrozen   = len(F['models_frozen'])
   M           = numfrozen + 3   # total no. of models (all frozen) + "best" + "lowest" + "earliest" models
   numTT       = len( TT_list )

   # 2. Test #
   Perf       = np.full( (M,numTT,2),  np.nan)   # [modelnum, TT, <train,test>]   
   Perf_vd    = np.full( (M,2,p.d1,2), np.nan)   # [modelnum, <train,test>, delay_change, <t1 vs. t2>]

   for mm in range(M):

      print('Performance testing: %d of %d frozen models' % (mm+1,M))

      # i. Identify the model for this index #
      if mm == numfrozen:              # best (lowest loss + 50% task loss) model
         model = F['model']
      elif mm == numfrozen + 1:        # lowest loss model
         model = F['model_lowest']        
      elif mm == numfrozen + 2:
         if 'model_earliest' in F.keys():
            model_epoch    = F['epoch_earliest']
            model          = F['model_earliest']
         else:       
            model_epoch    = F['epoch']
            model          = F['model']    
      else:                            # frozen model
         model = F['models_frozen'][mm]         

      # ii. Test performance for (specified) time-trial (TT) conditions #
      for TT in [0]:  #TT_list:
         train, test  = tk.print_model_performance(p, model, F, TT, silent = True)  
         Perf[mm,TT,0] = train
         Perf[mm,TT,1] = test

      # iii. Test performance for variable delays #
      Perf_vd[mm,:,:,:] = tk.Evaluate_Performance_Variable_Delay(p, model, F['X_input'])

   F['performance']     = Perf      # [modelnum, TT, <train,test>]
   F['performance_vd']  = Perf_vd   # [modelnum, <train,test>, delay_change, <t1 vs. t2>]

   return F



def run_Behavior_Errors(p, F):  # updates F with behavior fields

   ### Simulation parameters ########
   TT_test        = 0
   NumSims        = 500    # number of test simulations
   RNN_guess      = True
   if p.Model > 0:  # Logistic regression
      noise_sigs     = np.array([1, 2, 4, 8, 16, 32, 64, 128])  # original manuscript
      # noise_sigs     = np.arange(0.1,10.1,0.1)
      if p.Model == 8:
         Model_select   = -1   # lowest-loss
      else:
         Model_select   = -4
   else:             # All other models
      if p.actfn == 1:
         # noise_sigs     = np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
         noise_sigs = np.arange(0,5.05,0.05)
      else:
         # noise_sigs     = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0, 1.2])
         # noise_sigs     = np.array([0.05, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
         # noise_sigs     = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
         noise_sigs = np.arange(0.3,2.2,0.01)
      Model_select   = -4   # earliest performer

   ##################################

   taskfunc           = tk.Get_taskfunc(p)
   taskfunc.Error_study( p, F, Model_select, NumSims, noise_sigs, TT = 0, RNN_guess=RNN_guess)

   return F




