from __future__ import division
import numpy as np

def Model_Variant_Parameters(x):

  if not isinstance(x, dict):   # params class (not a dict)

    if x.Model < 0:               ###### RNN ##########
      x.check_perf       = True    
      x.stop_perf        = 0
      x.Z                = 3
      if x.Model_variant == 0:          # highest 
        x.h              = 1
        x.g              = 0.5
        x.sig            = 0.1
        x.reg_weight     = 1
        x.reg_metabolic  = 1
        x.Task_version   = 0
      elif x.Model_variant == 1:        # high  
        x.h              = 1
        x.g              = 0.5
        x.sig            = 0.1
        x.reg_weight     = 0.001
        x.reg_metabolic  = 0.001
        x.Task_version   = 0
      elif x.Model_variant == 2:        # intermediate 
        x.h              = 1
        x.g              = 1.0
        x.sig            = 0.1
        x.reg_weight     = 0.001
        x.reg_metabolic  = 0.001
        x.Task_version   = 0        
      elif x.Model_variant == 3:        # low 
        x.h              = 1
        x.g              = 2.0
        x.sig            = 0.1
        x.reg_weight     = 0
        x.reg_metabolic  = 0
        x.Task_version   = 0
      elif x.Model_variant == 4:        # lowest
        x.h              = 1.5
        x.g              = 4.0
        x.sig            = 0.1
        x.reg_weight     = 0
        x.reg_metabolic  = 0
        x.Task_version   = 0


    elif x.Model > 0:          ###### feedforward models ##########
      x.max_error        = -np.inf      
      x.nEpochs          = 10000
      x.Z                = 2
      if x.Model == 8:   # logistic
        x.n_epochs_stop  = 1000          
        x.check_perf     = False     # find global loss minimum (instead of early stopping)
        x.stop_perf      = -1        # -1: don't stop due to performance
        x.L2             = 0.1
        x.Task_version   = 0    #   0: classic, 1: train-all
      else:                   # MLP
        x.n_epochs_stop  = 2000          
        x.check_perf     = True    
        x.stop_perf      = 0          # 0: stop when get perfect performance
        if x.Model_variant == 0:     
          x.h              = 1
          x.sig            = 0.2
          x.L2             = 0.001     
          x.Task_version   = 0
        elif x.Model_variant == 1:   
          x.h              = 1
          x.sig            = 0.2
          x.L2             = 0.1     
          x.Task_version   = 0
        elif x.Model_variant >= 2:    
          x.h              = 1.5
          x.sig            = 0.2
          x.L2             = 0     
          if x.Model_variant == 2:   
            x.Task_version  = 0
          elif x.Model_variant == 3:              
            x.Task_version   = 1   

  else:  # dictionary

    if x['Model'] < 0:               ###### RNN ##########
      x['check_perf']       = True    
      x['stop_perf']        = 0
      x['Z']                = 3
      if x['Model_variant'] == 0:       
        x['h']              = 1
        x['g']              = 0.5
        x['sig']            = 0.1
        x['reg_weight']     = 0
        x['reg_metabolic']  = 0
        x['Task_version']   = 0
      elif x['Model_variant'] == 1:     
        x['h']              = 1
        x['g']              = 0.5
        x['sig']            = 0.1
        x['reg_weight']     = 0.005
        x['reg_metabolic']  = 0.005
        x['Task_version']   = 0
      elif x['Model_variant'] == 2:        
        x['h']              = 1
        x['g']              = 1.0
        x['sig']            = 0.1
        x['reg_weight']     = 0.005
        x['reg_metabolic']  = 0.005
        x['Task_version']   = 0
      elif x['Model_variant'] == 3:        
        x['h']              = 1
        x['g']              = 2.0
        x['sig']            = 0.1
        x['reg_weight']     = 0
        x['reg_metabolic']  = 0
        x['Task_version']   = 0
      elif x['Model_variant'] == 4:        
        x['h']              = 1.5
        x['g']              = 4.0
        x['sig']            = 0.1
        x['reg_weight']     = 0
        x['reg_metabolic']  = 0
        x['Task_version']   = 0

        # if x['Model_variant'] == 2:
        #   x['Task_version']  = 0
        # elif x['Model_variant'] == 3:
        #   x['Task_version'] = 1

    elif x['Model'] > 0:              ###### feedforward variants ##########
      x['max_error']        = -np.inf      
      x['nEpochs']          = 10000
      x['Z']                = 2
      if x['Model'] == 8:   # logistic
        x['n_epochs_stop']  = 1000          
        x['check_perf']     = False     # find global loss minimum (instead of early stopping)
        x['stop_perf']      = -1        # -1: don't stop due to performance
        x['L2']             = 0.1
        x['Task_version']   = 0    #   0: classic, 1: train-all
      else:                   # MLP
        x['n_epochs_stop']  = 2000          
        x['check_perf']     = True    
        x['stop_perf']      = 0          
        if x['Model_variant'] == 0:     
          x['h']              = 1
          x['sig']            = 0.2
          x['L2']             = 0.001     
          x['Task_version']   = 0
        elif x['Model_variant'] == 1:   
          x['h']              = 1
          x['sig']            = 0.2
          x['L2']             = 0.1     
          x['Task_version']   = 0
        elif x['Model_variant'] >= 2:    
          x['h']              = 1.5
          x['sig']            = 0.2
          x['L2']             = 0     
          if x['Model_variant'] == 2:  
            x['Task_version']  = 0
          elif x['Model_variant'] == 3:                     
            x['Task_version']   = 1   
    


