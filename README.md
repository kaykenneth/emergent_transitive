# emergent_transitive #

 Code and human behavioral data for "Emergent neural dynamics and geometry for a transitive inference task" 
 Pre-trained models presented in the study are available at datadryad: https://doi.org/10.5061/dryad.83bk3jb0v
 Contact Kenny Kay (kaykenneth@gmail.com) any clarification or requests.


###### Code ####################################

# Python notebooks  # 
- single_analysis.ipynb     - single model instances
- batch_analysis.ipynb      - multiple model instances
- human_analysis.ipynb      - human behavioral data (mechanical Turk)

# general #
- nn_models.py              - definitions of model architectures (logistic regression, MLP, RNN)
- task_train_test.py        - functions related to training/testing on the task
- init.py                   - set model parameters
- model_variants.py         - set model variant parameters

# task #
- ti_functions.py           - functions related to the transitive inference task 

# additional #
- nn_analyses.py            - functions for analyzing neural activity
- behavior_functions.py     - functions for reporting/plotting behavior from batches of models 
- nn_plot_functions.py      - plotting functions
- postproc.py               - (postprocessing analyses for models trained on cluster)


###### Model files #############################

# model files available at DataDryad (https://doi.org/10.5061/dryad.83bk3jb0v)

Feedforward models
ti254 — MLP, 100 seeds (regularization: 0.001)
ti253 — LR, 100 seeds  (regularization: 0.1)

Basic-Delay RNN
ti350 — tanh, delay=20, input_train=1, output_train=1, 200 seeds   "f-RNN"
ti351 — tanh, delay=20, input_train=0, output_train=0, 200 seeds   "r-RNN"
ti352 — tanh, delay=20, input_train=0, output_train=1, 200 seeds   "r-RNN" with feedforward outputs trainable

Extended-Delay RNN
ti353 — tanh, delay=60, input_train=1, output_train=1, 200 seeds   "f-RNN"
ti354 — tanh, delay=60, input_train=0, output_train=0, 200 seeds   "r-RNN"

Variable-Delay RNN
ti360 — tanh, delay=60, input_train=1, output_train=1, jit=0.67, 200 seeds  "f-RNN"
ti361 — tanh, delay=60, input_train=0, output_train=0, jit=0.67, 200 seeds  "r-RNN"

Feedforward-trainable RNN ("ff-RNN")
ti365 — tanh, delay=20, input_train=1, output_train=1, 200 seeds, 500 simulations               Basic-delay
ti366 — tanh, delay=60, input_train=1, output_train=1, 200 seeds, 500 simulations               Extended-delay
ti367 — tanh, delay=60, input_train=1, output_train=1, jit=0.67, 200 seeds, 500 simulations     Variable-delay




