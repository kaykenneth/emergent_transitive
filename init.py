from __future__ import division
import numpy as np
from collections import OrderedDict

from model_variants import Model_Variant_Parameters

class params():
   def __init__(self):
      
      self.PREFIX       = "TI"        # Prefix for filename  
      self.jobname      = "ti0/"      # Directory

      #   Basics    #
      self.Task             = 1       #   1: Transitive inference
      self.Task_version     = 0       #   0: standard task
      self.Model            = -1      #  -1: RNN, 7: MLP, 8: LR
      self.Model_variant    = 0       #   Constraint regime: 0: highest, 1: high, 2: intermediate, 3: low, 4: lowest 
      self.n_items          = 7       #   No. of Input patterns ("items" in task)

      #   Network    # 
      self.actfn         = 1          # Hidden unit non-linearity   (0: linear, 1: tanh, 2:tanh+, 3: relu,  4:softplus, 5:sigm)
      self.N             = 100        # No. of hidden units 
      self.h             = 1          # Initial scale factor for input weights
      self.g             = 0.5        # Initial scale factor for recurrent connectivity weights
      self.sig           = 0.2        # Standard Deviation of intrinsic network noise (each timestep)
      self.input_train   = True       # Feedforward input weights: 0: static input weights, 1: trainable input weights
      self.output_train  = True       # Feedforward output weights: 0: static input weights, 1: trainable input weights
      self.rec_train     = False      # Recurrent weights: 0: static input weights, 1: trainable input weights
      self.model_bias    = True       # Bias allowed for hidden units
      self.noise_test    = False      # Toggle intrinsic noise during testing
      self.dt            = 0.01    # (RNN) in seconds (e.g. 10 ms  ) timestep duration
      self.tau           = 0.1     # (RNN) in seconds (e.g. 100 ms ) unit time constant

      #   Model instance seeds    #
      self.Seed          = 1          # Seed for initial connectivity, inputs, and noise
      self.seedn         = self.Seed  # Seed for noise   
      self.seedJw        = self.Seed  # Seed for J and w matrices
 
      #   Training    #
      self.max_error     = 0.01        # Maximum error for the Task component of Error function
      self.reg_weight    = 0           # "alpha" hyperparameter for weight regl'n    
      self.reg_metabolic = 0           # "beta" hyperparameter for metabolic regl'n
      self.L2            = 0.1         # (feedforward) weight regularization
      self.blank_trials  = 0           # fraction of training trials that are Blank (no input)
      self.catch_trials  = 0           # fraction of training trials that are Catch (1-item trials) (should not respond w/ choice)
      self.B             = 128         # Batch size (no. of training trials per training batch)
      self.eta           = 0.001       # learning rate
      self.nEpochs       = 15000       # maximum # epochs to train 
      self.n_epochs_stop = 10000       # num epochs to attempt training without improvement of loss
      self.min_train_ep  = 1           # minimum # epochs to train
      self.check_perf    = True        # check model performance every training epoch + saves earliest 100% performer
      self.nEpochsfreeze = 300         # if >0, then saves frozen models during training for these number of epochs
      self.stop_perf     = 0           # -1: does not stop when perform, 0: stops training as soon as get 100% perfect performer 
                                       # >0: stops training when perfect perf + after this many more frozen models

      #   input       #        
      self.U              = self.N     # input dimension
      self.input_noise    = 0          # Gaussian noise to training inputs
      self.opt_train_args = {}         # (optional) for training function
      self.seedB          = self.Seed  # seed for input                 (batch)

      #   output      #
      self.Z              = 3      # 2: cross-entropy, 3: neural MSE
      self.outbias        = True   # bias term allowed for readout
      self.Choice_value  = 5       # (if p.Z = 3) target value for choice readout unit
      self.Rest_value    = 0.5     # (if p.Z = 3) target value for rest readout unit
      self.Choice_thresh = 0.85    # (if p.Z = 3) RT threshold for choice readout unit, [0,1]

      #   (unused parameters) #
      self.epochShow     = False       
      self.nEpochsplot   = 20     
      self.Curriculum    = [[]]       
      self.n_triplets     = int(self.n_items/3)    # for a different task 

      ####  Trial time structure   ####
      Trial_Timing_Parameters(self)
      ###################################

      ##### Model Variants #########
      Model_Variant_Parameters(self)
      ##############################

      # extra #
      self.seedRun       = self.Seed       # seed for post-processing simulation

      ##### Cluster parameters (ignore otherwise) #########
      self.runTime      = "0-11:45"
      self.cpuMem       = "20G"
      self.Device       = 'cpu' 
      self.DIR          = "/" + self.jobname    
      self.scan         = OrderedDict([   ('Model',         np.array([-1])),
                                          ('Model_variant', np.array([0,1,2,3,4])) , \
                                          ('actfn',         np.array([1])), \
                                          ('Delay',         np.array([60])), \
                                          ('input_train',   np.array([1])), \
                                          ('output_train',  np.array([1])), \
                                          ('rec_train',     np.array([0])), \
                                          ('jit',           np.array([0.67])), \
                                          ('Seed',          np.arange(0,200))  ] ) 
      #########################################################################

      ###########################################


def Trial_Timing_Parameters(x):
  
  if not isinstance(x, dict):

    # Specify #
    if x.Model < 0:        # RNNs
      ##### manually specify here #########################
      t_pre                      = 5
      delay                      = 20
      dur                        = 1
      jit                        = np.array([0])
      #####################################################
      # (calculate t1, t2, T) #
      d1 = delay
      d2 = delay
      t1 = int( t_pre )                 
      t2 = int( t_pre + dur + d1 - 1 )  
      T  = int( t2 + dur + d2 - 1 ) 
    elif x.Model > 0:          # Feedforward models
      t_pre,delay,dur,d1,d2,t1,t2,T,jit   = ( 0,0,1,0,0,0,0,1,-1 )   # default     

    # Set #
    x.t_pre         = t_pre      # Duration of pre-stim period
    x.dur           = dur        # Stimulus duration
    x.delay         = delay      # Delay
    x.d1            = d1         # Delay from stim1 to stim2
    x.d2            = d2         # Delay from stim2 to readout
    x.t1            = t1               # Time of stim1
    x.t2            = t2               # Time of stim 20            
    x.T             = T          # Total dur of trial
    x.jit           = jit

  else:     # dictionary (cluster) #

   # Specify #
   if x['Model'] < 0:  # recurrent model
      t_pre              = 5
      dur                = 1
      jit                = x['jit']
      delay              = x['Delay']
      # Re-evaluate these parmeters  #
      t1                 = t_pre
      d1                 = delay
      d2                 = delay 
      t2                 = int( t_pre + dur + d1 - 1 )  
      T                  = int( t2    + dur + d1 - 1 )  
   elif x['Model'] > 0:  # feedforward model
      t_pre,delay,dur,d1,d2,t1,t2,T,jit   = ( 0,0,1,0,0,0,0,1,-1 )   # default    

   # Set values #
   x['t_pre']         = t_pre      # Duration of pre-stim period
   x['dur']           = dur        # Stimulus duration
   x['d1']            = d1         # Delay from stim1 to stim2
   x['d2']            = d2         # Delay from stim2 to readout
   x['t1']            = t1               # Time of stim1
   x['t2']            = t2               # Time of stim 20            
   x['T']             = T          # Total dur of trial
   x['jit']           = jit
 

def genfilename(p,localdir = '',omit_jobname=False):
    if localdir != '':
      if omit_jobname:
        DIR = localdir
      else:
        DIR = localdir + p.jobname    # appends jobname (a directory)
    else:
      DIR = p.DIR
    name = "%s%s_M%d_V%d_I%d_it%d_D%d_af%d_Jit%0.2f_sig%0.2f_g%0.1f_meta%0.2f_weight%0.2f_S%d" \
        %( DIR, p.PREFIX,  p.Model, p.Model_variant, p.n_items, p.input_train, p.d1, p.actfn, p.jit, p.sig, p.g, p.reg_metabolic, p.reg_weight, p.Seed )
    # print(name)
    return name



q = params()    # instantiate here so can refer to parms below


class Struct(object):              
   def __init__(self, **entries):
      self.__dict__.update(entries)

def singleout(TID = 0):      # Get parameter object p for this TID
   
   # 1. Get array coordinate [x1,x2,x3,..] corresponding to the Task ID (TID)
   Dict = dict(vars(q))               # get dictionary of params
   njobs = 1
   dims = []                          # list of dimensions
   for k in Dict['scan'].keys():      # iterate the parameters to Scan over
      dim        =  len(Dict['scan'][k]) # dimension: number of values to scan
      njobs     *=  dim    # enumerate more jobs
      dims.append(  dim  )
   matrix = np.zeros(dims)
   subs = np.unravel_index(TID, matrix.shape, 'C')  # Convert linear index to coord index // 'C', i.e. row-major
   
   # 2. Set the (scanning) parameters to this iterations' values #
   for i, k in enumerate( Dict['scan'].keys() ) :
      val     = Dict['scan'][k][subs[i]]
      Dict[k] = val
      # if k == 'jit':
      #   if val > 0:
      #     print(k,val)
      # print(k,val)

   #### (optional) manually re-evaluate any default parameters that depend on scanned parameters ####

    # seed #
   Dict['seedJw']   = Dict['Seed']
   Dict['seedn']    = Dict['Seed']
   Dict['seedB']    = Dict['Seed']    
   Model_Variant_Parameters(Dict)   # re-evaluate for model variant #
   Trial_Timing_Parameters(Dict)    # re-evaluate for trial timing #

    # (AI task) #
   if Dict['Task'] == 2:
      Dict['n_triplets'] = int( Dict['n_items'] / 3 ) 

    # outputs #
   Dict['njobs'] = njobs
   Dict['dims'] = dims
   p = Struct(**Dict)

  #  print( "%d, %d" % (p.jit, p.Seed))

   return p



def genscript(njobs):
   script = "#!/bin/bash\n" +\
   "#SBATCH --account=theory\n" +\
   "#SBATCH --job-name=" + q.jobname[-8::] + "\n" +\
   "#SBATCH -c 1\n" +\
   "#SBATCH -o " + q.DIR + "slurm-%A-%a.out\n" +\
   "#SBATCH --time=" + q.runTime + "\n" +\
   "#SBATCH --mem=" + q.cpuMem + "\n" +\
   "#SBATCH --array=0-" + str(njobs-1) + "\n" +\
   "/burg/opt/anaconda3-2020.11/anaconda3/bin/python drvr.py $SLURM_ARRAY_TASK_ID\n"
   filename = "jobscript.sh"
   with open(filename, "w") as myfile:
      myfile.write(script)
   myfile.close()

