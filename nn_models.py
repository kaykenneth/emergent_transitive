from __future__ import division  # Ramin
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.normal import Normal




#########  Helper functions  #########################

def Normsamp( sig, time, batch, numcell ):
    dist = Normal(loc = 0, scale = ( sig * torch.ones( [time, batch, numcell] ) ) )
    return dist.sample()

def Tanh_readout(p,input):
    out = 0.5 * np.tanh(input - p.Choice_value*0.5) + 0.5   # Non-linear function to model Reaction Time (after Chaisangmongkon et. al 2017)
    return out

def Parse_3_readouts(p,rawout):      # rawout [T,B,R]
    
    # initialize #
    numCond     = rawout.shape[1]
    Choice      = np.zeros([numCond])   # choice
    RT          = np.full([numCond],np.nan)   # reaction time

    choiceout = Tanh_readout(p,rawout)

    for bb in range(numCond):

        choiceclip  = choiceout[p.t2:,bb,0:2]       # [T,R] Units z1, z2 output values after t2
        postchoice  = choiceclip > p.Choice_thresh  # [T,R] (exceed choice threshold)
        # postchoice  = choiceclip > (.8/.85) * p.Choice_thresh  # [T,R] (exceed choice threshold)
        chosen      = np.any(postchoice,axis=0)     # [2]   Did z1, z2 choose at any time point after t2?

        if np.all(~chosen):         # Neither 1 or 2 chosen
            if 1:           # normal
                Choice[bb]      = -1    
            elif 0:         # Beiran estimator  (output unit that is nearest threshold)
                maxvals_outputs = np.max( choiceclip, axis=0 ) 
                C               = np.argmax( maxvals_outputs  )
                Choice[bb]      = C  # 0: output cell 0, 1: output cell 1
                RT[bb]          = np.argmax( choiceclip[:,C] ) / p.d2
        elif np.all(chosen):        # "Both" 1 and 2 chosen (erroneous)
            Choice[bb]      = -2    
        else:                       # Either 1 or 2 chosen
            C = np.where(chosen)[0].copy()
            if C == 0:      # Output cell 0 activated   (e.g. TI (lower triangle): item 2 is closer to "A")
                Choice[bb]  = 0
            elif C == 1:    # Output cell 1 activated   (e.g. TI (upper triangle): item 1 is closer to "A")
                Choice[bb]  = 1
            RT[bb]          = np.min( np.where(postchoice[:,C])[0] ) / p.d2  # RT is the proportion of d2
            # RT[bb]          = (.85/.80) * np.min( np.where(postchoice[:,C])[0] ) / p.d2  # RT is the proportion of d2
            
    return Choice, RT    

def nonlinearfunc(v):
    if v == 0:                  # linear
        f = torch.nn.Identity()
    elif v == 1:                # tanh
        f = torch.tanh
    elif v == 2:                # tanh+
        def tanhrect(x):
            r = torch.max( torch.zeros(x.shape), torch.tanh(x) )
            return r
        f = tanhrect
    elif v == 3:                # relu
        f = torch.relu    
    elif v == 4:                # softplus
        f = torch.nn.functional.softplus 
    elif v == 5:                # sigmoid
        f = torch.sigmoid  
    return f


def PCA_filter(data, pca, center, dim_keep=np.arange(0,6) ):  
    # Data is list of data matrices
    # pca: scikit learn PCA object
    # dim_filt: numpy array of PCs to filter out  (default: top 3 PCs)
    import nn_analyses as nna
    N       = data.shape[1]
    inds    = np.invert( np.isin( np.arange(N) ,  dim_keep ) )
    datap   = nna.project_data(data,   pca,  center=center)       # forward project
    datap[:,inds] = 0           # zero-out all but top dims
    dataf   = nna.project_data(datap, pca, center=center, inverse=True) # inverse
    if not torch.is_tensor(dataf):
        dataf   = torch.from_numpy(dataf)
    return dataf.float()






#########  Model definitions  #########################

class Model8(nn.Module):   # Logistic Regression 
    def __init__(self,p):                                
        super(Model8, self).__init__()
        self.readout = nn.Linear(int(2*p.U), p.Z)         # 100 >> 1
    def forward(self, p, input, **kwargs):
        if p.sig == 0:
            out          = self.readout(input)
        else:
            if 0:   # add noise to input
                noise        = Normsamp(p.sig, 1, input.shape[1], p.N)  
                out          = self.readout(input + noise)
            elif 1:   # add noise to readout
                # print('hqwepf')
                out          = self.readout(input) + Normsamp(p.sig, 1, input.shape[1], p.Z)
        return out, 0   


class Model7(nn.Module):   # MLP with 1-layer

    def __init__(self,p):                                
        
        super(Model7, self).__init__()
        self.actfn      = p.actfn
            # define layers #
        hid        = nn.Linear(int(2*p.U), p.N, bias = p.model_bias)         # 100 >> 1
        readout    = nn.Linear(p.N, p.Z, bias = p.outbias)         # 100 >> 1
            # initialize #
        nn.init.normal_(      hid.weight, mean=0.0, std=p.h/np.sqrt(p.U)  )       # B initialization
        nn.init.constant_(    readout.weight, 0  )                                  # W initialization
        if p.model_bias:
            nn.init.constant_(  hid.weight, 0  )                                    # bias
            # set #
        self.hid        = hid
        self.readout    = readout

    def activation(self,x):  ###  1:tanh, 2:relu, 3:softplus, 4:sigmoid, 5:tanh rect
        f = nonlinearfunc(self.actfn)
        r = f(x)
        return r

    def forward(self, p, input, **kwargs):   
        hidden  = self.hid(input)

        if p.sig == 0 or kwargs['noisefree'] == True:   
            nonlin       = self.activation(hidden)
        else:        # intrinsic noise
            noise        = Normsamp(p.sig,1,hidden.shape[1],p.N)  
            nonlin       = self.activation(hidden + noise)
        
        if 'lesion' in kwargs:
            inds = kwargs['lesion']
            if inds.size > 0:
                nonlin[:,:,inds] = 0  # set to 0

        out     = self.readout(nonlin)
        return out, (hidden,nonlin)
  


class Model6():            # continuous-time RNN #

    def __init__(self,p):     

        np.random.seed(p.Seed)         # neural seed

        # set non-linear function #
        self.actfn  = p.actfn

        # recurrent weights #
        J_init  = np.random.normal( 0 , p.g / np.sqrt(p.N), size=(p.N,p.N) ).astype(np.float32)
    
        # input weights #
        wu_init = np.random.normal( 0 , p.h / np.sqrt(p.U), size=(p.N,p.U) ).astype(np.float32)

        # output weights #
        if p.output_train:
            wz_init = np.zeros([p.Z,p.N]).astype(np.float32)   # [R,N]  initial output connect
        else:    
            wz_init = np.random.normal( 0, 1 / np.sqrt(p.N),  size=(p.Z,p.N) ).astype(np.float32)   # [R,N]  initial output connect
        
        # unit bias terms #
        b_init    = np.zeros([p.N,1]).astype(np.float32)    # network unit bias terms 
        bz_init   = np.zeros([p.Z,1]).astype(np.float32)    # output unit bias terms  
        x_ic_init = np.zeros([p.N,1]).astype(np.float32)    # network unit bias terms 

        # set model parameters (+ convert to torch tensors) #
        self.wu     = Variable( torch.from_numpy(wu_init),   requires_grad = p.input_train )                 # (actual weights)
        self.J      = Variable( torch.from_numpy(J_init),    requires_grad = p.rec_train)                  # ( "   " )
        self.wz     = Variable( torch.from_numpy(wz_init),   requires_grad = p.output_train)                 # ( "   " )
        self.b      = Variable( torch.from_numpy(b_init),    requires_grad = p.rec_train)                  # ( "   " )
        self.bz     = Variable( torch.from_numpy(bz_init),   requires_grad = (p.outbias and p.output_train) )             
        self.x_ic   = Variable( torch.from_numpy(x_ic_init), requires_grad = True)                          # experimenting right now 

    def activation(self,x):  ###  1:tanh, 2:relu, 3:softplus, 4:sigmoid, 5:tanh+
        f = nonlinearfunc(self.actfn)
        r = f(x)
        return r

    def dynamical(self, p, wu, J, wz, b, u, bz, x_ic, noisefree, initrand, *xinit):

        T = list(u.size())[0]                       # input duration
        B = list(u.size())[2]                       # batch

        # outputs #
        z       = torch.zeros(T, B, p.Z)              # output          [T, B, R]          for ti in range(T):  
        xs      = torch.zeros(T, p.N, B, requires_grad=False)              # cell activation [T, N, B]
        rs      = torch.zeros(T, p.N, B, requires_grad=False)              # cell activity   [T, N, B]
        Jrs     = torch.zeros(T, p.N, B, requires_grad=False)              # cell rec input      [T, N, B]
        Wus     = torch.zeros(T, p.N, B, requires_grad=False)              # external input

        # Initialization #
        if len(xinit) == 0:     # No manual initial state
            if initrand:               # Gaussian initial state
                xinit   = torch.randn(p.N,1)                
            else:                      # Model initial condition           
                # xinit   = 0 * torch.randn(p.N,1)        # (old) zero IC
                xinit   = x_ic        # test              # (new) trainable IC
        else:                          #  Manually set initial state 
            B_length = xinit[0].size()[0]    
            if B_length == 1:   
                xinit   = xinit[0].squeeze().unsqueeze(0).t()              # cell initial activation [N,1]
            else:
                xinit   = xinit[0].t()

        # Noise #
        if (noisefree < 0) or (noisefree == False):      # for Training 
            sig_noise = p.sig
        else:                       # for Testing
            sig_noise = 0
        
        # Initial state #   (Note that this is NOT stored or plotted)
        x       = xinit           # [N,1] + [N,B]  init activation                
        r       = self.activation(x)

        # Run simulation #
        for ti in range(T):                   # simulate           

            # Recurrent input #
            Jr         = J.mm(r)                   # (J x r) [N,B] rec input (act * rec weights)

            # Intrinsic noise #
            noise      = sig_noise * torch.randn(p.N, B)

            # External input #
            Wu         = wu.mm( u[ti,:,:] )

            # Dynamical update #
            x          = x + (p.dt/p.tau) * (  -x + Jr + b + Wu + noise )    # new activation, u[ti,:,:]:[U,B], wu:[N,U] 
                   
        #      [N,B]                [N,B] [N,B] [N,1]  [N,B]          [N,B]

            # Non-linearity #
            r           = self.activation(x)                        # [N,B], current activity (i.e. non-linearity of activation)
        #                 wz: [z,N], r: [N,B]

            # store values
            z[ti,:,:]   = ( wz.mm(r) + bz ).t()                # [B,z] output activity      
            xs[ti,:,:]  = x                                    # [N,B] install activation
            rs[ti,:,:]  = r                                    # [N,B] install activity
            Jrs[ti,:,:] = Jr                                   # [N,B] install rec input 
            Wus[ti,:,:] = Wu

        return z, (xs, rs, Jrs, Wus)
    
    def forward(self, p, input, noisefree, initrand, *xinit):
        out, hidden  = self.dynamical( p, self.wu, self.J, self.wz, self.b, input, self.bz, self.x_ic, noisefree, initrand, *xinit)
        return out, hidden   

    def inputvector(self,p,u):  # input direction  # (no contribution of state, recurrence, or noise)
        # u:  external input
        wu  = self.wu
        b   = self.b
        #      [N,B]                [N,B] [N,B] [N,1]  [N,B]          [N,B]
        # xs   = (p.dt/p.tau) * ( b + wu.mm(u[0,:,:] )   )    # new activation, u[ti,:,:]:[U,B], wu:[N,U] 
        xs   = (p.dt/p.tau) * ( wu.mm(u[0,:,:] )   )   
        rs   = self.activation(xs)                        # [N,B], current activity (i.e. non-linearity of activation)
        return (xs,rs)





class Model11():        # Linearized RNN (Switching Linear Dynamical System)

    def __init__( self, p, Lsp, oscbasis, pca, RNN , Filter_mode = 1):                                
                
        import nn_analyses          as nna

        # Linear parms #
        self.Jac       = Variable( torch.from_numpy( Lsp['Jac']   ),   requires_grad=False )      # Jacobian
        self.xstar     = Variable( torch.from_numpy( Lsp['xstar'] ),   requires_grad=False )    # fixed point
        self.oscbasis  = oscbasis
        self.pca       = pca
        self.ZEROTH    = 1
        if Filter_mode == 1:         # Osci-filtered     @ FP
            print('2D filtered')
            self.FILTER    = 1    # 1: oscbasis filtered
            self.IC        = 2    # 0: zero, 1: trained IC, 2: FP
        elif Filter_mode == 0:           # Unfiltered        @ IC / FP
            print('unfiltered')
            self.FILTER    = 0    # 0: none, 1: oscbasis,   2: PCA
            self.IC        = 2    # 0: zero, 1: trained IC, 2: FP
        elif Filter_mode == 2:         # PCA-filtered      @ FP
            self.FILTER    = 2    # 0: none, 1: oscbasis, 2: PCA
            self.IC        = 2    # 0: zero, 1: trained IC, 2: FP

        # Copy RNN parms (needed for 0th-order approximation term + readout mimic) #
        self.J      = Variable( RNN.J.detach().clone(),      requires_grad=False )      
        self.wu     = Variable( RNN.wu.detach().clone(),     requires_grad=False )      
        self.wz     = Variable( RNN.wz.detach().clone(),     requires_grad=False )                 
        self.b      = Variable( RNN.b.detach().clone(),      requires_grad=False )                  
        self.bz     = Variable( RNN.bz.detach().clone(),     requires_grad=False )           
        self.x_ic   = Variable( RNN.x_ic.detach().clone(),   requires_grad=False )    
        self.actfn  = p.actfn
    
    def activation(self,x):  ###  1:tanh, 2:relu, 3:softplus, 4:sigmoid, 5:tanh+
        f = nonlinearfunc(self.actfn)
        r = f(x)
        return r

    def forward(self, p, u , noisefree, initrand, *xinit):
        out, hidden  = self.dynamical( p, u, noisefree, initrand, *xinit)
        return out, hidden   

    def dynamical(self, p, u, noisefree, initrand, *xinit):

        # (basic values) #
        T       = u.shape[0]                       # trial / input duration
        B       = u.shape[2]                       # batch size
        numFP   = self.xstar.shape[-1]                       # no. of fixed points

        # Initialize outputs #
        xs      = torch.zeros(T, p.N, B)            # cell activation [T, N, B]
        rs      = torch.zeros(T, p.N, B)            # cell activity   [T, N, B]
        z       = torch.zeros(T, B, p.Z)            # output          [T, B, R]          for ti in range(T):  
        fs      = np.zeros((T, B, 1))              # FP referenced          [T, B, 1]          for ti in range(T):  
        fds     = np.zeros((T, B, numFP))          # Distance to FPs

        # 1. Set initial condition  #
        if len(xinit) == 0:     # none specified outside of .forward()
            initialized = False   
            if initrand:
                x0   = torch.randn(B,p.N)            # train
            else:
                # initial conditions
                if self.IC == 1:    # Trained Initial Conditions
                    x0   = torch.zeros(B, p.N)
                    x0   = self.x_ic.t()                 # Jan 2022
                elif self.IC == 2:  # FP of the Linearization
                    x0   = self.xstar[np.newaxis,:,0]   
                else:               # zero-point
                   x0   = torch.zeros(B, p.N)        # if testing  
        else:                    
            initialized = True
            if B != xinit[0].shape[1] :  # check that Batch size is the same as input
                raise Exception('B needs to be matched between input and xinit')
            else:
                x0 = xinit[0].t()   # [B,N]

        # (optional) #
        if self.FILTER == 1:
            x0     =  self.oscbasis.filter( x0, center=True ).float()
        elif self.FILTER == 2:
            x0     =  PCA_filter(x0, self.pca, center=True ).float()        

        # 2. Identify the FP to adopt linearized dynamics #
        FP      = np.zeros( (B) , dtype='int32')   # assume FP 0 is the first FP 
        xstar   = self.xstar[:,FP].t()  
        # FP_dists = torch.zeros(B,numFP)
        # for bb in range(B):
        #     FP_dists[bb,:] = torch.norm( xinit[bb,:] - xstars  , p=2, dim=1)   # distance to each FP
        # if 0:
        #     FP       = np.argmin(FP_dists.cpu().detach().numpy(), axis=1)                            # index of the nearest FP
        # else:
        #     FP      = np.zeros( (B) )
 
        # 3. Noise level (optional) #
        if noisefree == False:      # for Training 
            sig_noise = p.sig
        else:                       # for Testing
            sig_noise = 0

        # 4. Calculate y0, set #     (i.e. initial condition for y, the displacement from FP) #
        y0    = x0 - xstar
        y0    = y0.t()
        y     = y0                # [N,B] 

        # Run simulation #
        for ti in range(T):

            # i.  Linear dynamics #    (Jacobian, A)
            Ay = torch.zeros( p.N, B )
            for b in range(B):   
                ff        = FP[b]                # FP index
                A         = self.Jac[:,:,ff]   # Jacobian
                Ay[:,b]   = A.mm( y[:,b,np.newaxis] ).flatten()     
                # Ay[:,bb]   = Jacs[:,:,int(FP[bb])].mm( y[:,bb,np.newaxis] ).flatten()        # (J x r) [N,B] rec input (act * rec weights)
                # Ay[:,bb]   = torch.matmul( Jacs[:,:, int(FP[bb]) ] , y[:,bb] )
                #   Ay:[N,B]                    Jacs: [N,N]   y: [N,B]
            # Ay        = torch.matmul( A, y )                               

            # Calc 0th order term for linear approx (i.e. at slow/fixed point) #
            if self.ZEROTH:           # Sussillo Barak 2013, eq'n 3.2 
                rstar       = self.activation( xstar ).t()   # [N,B]
                Jr          = self.J.mm( rstar ).t()                   # (J x r) [N,B] rec input (act * rec weights)\
                zeroth      = -xstar + Jr + self.b.t()                    
                zeroth      = zeroth.t()
                # x         = x + (p.dt/p.tau) * (  -x + Jr + b + Wu +  noise  ) 
            else:
                zeroth      = 0  
                
            # Noise #
            noise       = sig_noise * torch.randn(p.N, B)

            # Ext input #
            Wu          = self.wu.mm( u[ti,:,:] )           # [N,B]
            if self.FILTER == 1:
                Wu      = self.oscbasis.filter(  Wu.t() , center=False  ).t().float()
            elif self.FILTER == 2:
                Wu     =  PCA_filter( Wu.t(), self.pca, center=False ).t()     

            # Dynamical update #
            y               = y + (p.dt/p.tau) * (  zeroth + Ay + Wu + noise  )    # new activation, u[ti,:,:]:[U,B], wu:[N,U] 
            # y               = y + (p.dt/p.tau) * ( Ay + Wu )    # new activation, u[ti,:,:]:[U,B], wu:[N,U] 
        #      [N,B]                [N,B] [N,B] [N,1]  [N,B]          [N,B]

            # De-center from FP to recover x #
            x               = y + xstar.t()
            # x               = y.detach().clone() + xstars[FP,:].t().detach().clone()

            # Non-linearity (solely for readout) #
            r               = self.activation(x)                        # [N,B], current activity (i.e. non-linearity of activation)
        #                 wz: [z,N], r: [N,B]

            # Store values #
            z[ti,:,:]       = ( self.wz.mm(r) + self.bz ).t()                    # [B,z] output activity      
            xs[ti,:,:]      = x                                    # [N,B] install activation
            # rs[ti,:,:]      = r                                    # [N,B] install activity
            # fs[ti,:,0]      = Variable(torch.from_numpy(FP.squeeze()),requires_grad=False)
            # fs[ti,:,0]      = FP
            # fds[ti,:,:]     = FP_dists
            
        return z, (xs, rs, fs, fds)        





