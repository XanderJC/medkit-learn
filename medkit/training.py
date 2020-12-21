from medkit.environments.RNN import RNNEnv
from medkit.domains.ICU import ICUDomain, icu_dataset
from medkit.policies.RNNp import RNNPol
from medkit.initialisers.VAE import VAEInit,VAE
import torch

domain = ICUDomain()

# Would normally load a pretrained model, set load=False so it doesn't
test_env = RNNEnv(domain,load=False)
test_pol = RNNPol(domain,load=False)
test_init = VAEInit(domain,load=False)

data = icu_dataset()

# Note hyperparameters for training including learning rate and epochs are
# stored in domain.{env,pol,init}_config
test_pol.train(data,batch_size=64)

# Save model by uncommenting
test_pol.model.save_model()

test_init.train(data,batch_size=64)
test_init.model.save_model()