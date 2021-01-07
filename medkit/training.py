from medkit.environments import TForceEnv, SVAEEnv, StateSpaceEnv, CRNEnv
from medkit.domains import ICUDomain, WardDomain, CFDomain, standard_dataset
from medkit.policies import LSTMPol, LinearPol, MLPPol
from medkit.initialisers import VAEInit
import torch
from medkit.scenario import scenario

domain = CFDomain(y_dim=8)

# Would normally load a pretrained model, set load=False so it doesn't
test_env = TForceEnv(domain,load=False)
test_pol = LSTMPol(domain,load=False)
test_init = VAEInit(domain,load=False)

data = standard_dataset(domain)

# Note hyperparameters for training including learning rate and epochs are
# stored in domain.{env,pol,init}_config
test_pol.train(data,batch_size=64)

# Save model by uncommenting
test_pol.model.save_model()

#test_init.train(data,batch_size=64)
#test_init.model.save_model()