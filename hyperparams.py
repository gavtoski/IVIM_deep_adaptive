"""
Created September 2020 by Oliver Gurney-Champion & Misha Kaandorp
oliver.gurney.champion@gmail.com / o.j.gurney-champion@amsterdamumc.nl
https://www.github.com/ochampion

Adapted January 2022 by Paulien Voorter
p.voorter@maastrichtuniversity.nl 
https://www.github.com/paulienvoorter

Adaptive constraints + multi-phase tune implemented by Bin Hoang (2025)
nhat_hoang@urmc.rochester.edu

Code is uploaded as part of the publication: Voorter et al. Physics-informed neural networks improve three-component model fitting of intravoxel incoherent motion MR imaging in cerebrovascular disease (2022)

requirements:
numpy
torch
"""
import torch
import numpy as np


class train_pars:
    def __init__(self):
        self.optim='adam' #these are the optimisers implementd. Choices are: 'sgd'; 'sgdr'; 'adagrad' adam
        self.lr = 0.00003 # this is the learning rate.
        self.patience= 10 # this is the number of epochs without improvement that the network waits untill determining it found its optimum
        self.batch_size= 128 # number of datasets taken along per iteration
        self.maxit = 500 # max iterations per epoch
        self.split = 0.9 # split of test and validation data
        self.load_nn= False # load the neural network instead of retraining
        self.loss_fun = 'rms' # what is the loss used for the model. rms is root mean square (linear regression-like); L1 is L1 normalisation (less focus on outliers)
        self.skip_net = False # skip the network training and evaluation
        self.scheduler = True # as discussed in the publications Kaandorp et al, LR is important. This approach allows to reduce the LR itteratively when there is no improvement throughout an 5 consecutive epochs
        # use GPU if available
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.use_cuda else "cpu")
        self.select_best = False


class net_pars:
    def __init__(self, model_type="3C", tissue_type="mixed", pad_fraction=None, IR=False):
        self.model_type = model_type
        self.tissue_type = tissue_type

        # Architecture settings
        self.dropout = 0.1
        self.batch_norm = True
        self.parallel = True
        self.con = 'sigmoidabs'
        self.fitS0 = True
        self.IR = IR
        self.depth = 2
        self.width = 0

        if model_type == "3C":
            self.param_names = ['Dpar', 'Fint', 'Dint', 'Fmv', 'Dmv', 'S0']
            if tissue_type == "mixed":
                self.cons_min = [0.0001, 0.0,   0.000, 0.0,   0.004, 0.9]
                self.cons_max = [0.0022, 0.50,  0.005, 0.50,  0.25,  1.1]
            elif tissue_type == "NAWM":
                self.cons_min = [0.0003, 0.00,  0.000, 0.004, 0.001, 0.9]
                self.cons_max = [0.0009, 0.15,  0.005, 0.015, 0.15,  1.1]
            elif tissue_type == "WMH":
                self.cons_min = [0.0003, 0.01,  0.001, 0.004, 0.001, 0.9]
                self.cons_max = [0.00105, 0.30, 0.005, 0.025, 0.15,  1.1]
            elif tissue_type == "original":
                self.cons_min = [0.0001, 0.0,   0.000, 0.0,   0.004, 0.9]
                self.cons_max = [0.0015, 0.40,  0.004, 0.2,   0.2,   1.1]
            else:
                raise ValueError(f"[net_pars] Unknown 3C tissue type: {tissue_type}")

        elif model_type == "2C":
            self.param_names = ['Dpar', 'Fmv', 'Dmv', 'S0']
            if tissue_type == "NAWM":
                self.cons_min = [0.0001, 0.002, 0.005, 0.9]
                self.cons_max = [0.00080, 0.075, 0.030, 1.1]
            elif tissue_type == "WMH":
                self.cons_min = [0.0001, 0.002, 0.005, 0.9]
                self.cons_max = [0.00080, 0.125, 0.020, 1.1]
            elif tissue_type == "mixed":
                self.cons_min = [0.0001, 0.002, 0.005, 0.9]
                self.cons_max = [0.00090, 0.125, 0.030, 1.1]
            else:
                raise ValueError(f"[net_pars] Unknown 2C tissue type: {tissue_type}")

        else:
            raise ValueError(f"[net_pars] Unknown model_type: {model_type}")

        # Padded constraints
        if pad_fraction is None:
            pad_fraction = 0.3 if tissue_type in ["mixed", "original"] else 0.25

        range_pad = pad_fraction * (np.array(self.cons_max) - np.array(self.cons_min))
        self.cons_min = np.clip(np.array(self.cons_min) - range_pad, a_min=0, a_max=None)
        self.cons_max = np.array(self.cons_max) + range_pad



class lsqfit:
    def __init__(self):
        self.do_fit = True # skip lsq fitting
        self.fitS0 = True # indicates whether to fit S0 (True) or fix it to 1 in the least squares fit.
        self.jobs = 1 # number of parallel jobs. If set to 1, no parallel computing is used
        self.bounds = ([0.9, 0.0001, 0.0, 0.0015, 0.0, 0.004], [1.1, 0.0015, 0.4, 0.004, 0.2, 0.2]) # S0, Dpar, Fint, Dint, Fmv, Dmv

class sim:
    def __init__(self):
        self.bvalues = np.array([0, 5, 7, 10, 15, 20, 30, 40, 50, 60, 100, 200, 400, 700, 1000]) # array of b-values
        self.SNR = 35 # the SNR to simulate at
        self.sims = 11500000 # number of simulations to run
        self.num_samples_eval = 10000 # number of simualtiosn te evaluate. This can be lower than the number run. Particularly to save time when fitting. More simulations help with generating sufficient data for the neural network
        self.distribution = 'uniform' #Define distribution from which IVIM parameters are sampled. Try 'uniform', 'normal' or 'normal-wide'
        self.repeats = 1 # this is the number of repeats for simulations to assess the stability
        self.n_ensemble = 20 # this is the number of instances in the network ensemble
        self.jobs = 1 # number of processes used to train the network instances of the ensemble in parallel (advised when training on cpu)
        #self.IR = False #True for IR-IVIM, False for IVIM without inversion recovery
        self.rician = True # add rician noise to simulations; if false, gaussian noise is added instead
        self.range = ([0.0001, 0.0, 0.0015, 0.0, 0.004], [0.0015, 0.40, 0.004, 0.2, 0.2]) # Dpar, Fint, Dint, Fmv, Dmv
 
class rel_times:
    """relaxation times and acquisition parameters, which are required when accounting for inversion recovery"""
    def __init__(self):
        self.bloodT2 = 275 #ms [Wong et al. JMRI (2020)]
        self.tissueT2 = 95 #ms [Wong et al. JMRI (2020)]
        self.isfT2 = 503 # ms [Rydhog et al Magn.Res.Im. (2014)]
        self.bloodT1 =  1624 #ms [Wong et al. JMRI (2020)]
        self.tissueT1 =  1081 #ms [Wong et al. JMRI (2020)]
        self.isfT1 =  1250 # ms [Wong et al. JMRI (2020)]
        self.echotime = 84 # ms
        self.repetitiontime = 6800 # ms
        self.inversiontime = 2230 # ms
        
class hyperparams:
    def __init__(self, model_type="3C", tissue_type="mixed", IR=False):
        self.model_type = model_type
        self.tissue_type = tissue_type
        self.use_three_compartment = model_type == "3C"

        profile_model = "brain3" if model_type == "3C" else "brain2"
        profile = f"{profile_model}_{tissue_type}"
        self.train_pars = train_pars()
        self.net_pars = net_pars(model_type=model_type, tissue_type=tissue_type, IR=IR)
        self.fit = lsqfit()
        self.sim = sim()
        self.rel_times = rel_times()

