from medkit.environments import TForceEnv, SVAEEnv, StateSpaceEnv, CRNEnv
from medkit.domains import ICUDomain, WardDomain, CFDomain
from medkit.policies import LSTMPol, LinearPol, MLPPol
from medkit.initialisers import VAEInit
from medkit.bases import standard_dataset
import torch
from medkit.scenario import scenario

'''
domain = ICUDomain(y_dim=2)
data = standard_dataset(domain,save_scale=True)

test_init = VAEInit(domain,load=False)
test_init.train(data,batch_size=64)
test_init.model.save_model()
'''

'''
env_dict = {'TForce':TForceEnv,'SVAE':SVAEEnv,'StateSpace':StateSpaceEnv,'CRN':CRNEnv}
for env in ['TForce']:
    for y_dim in [2,4,8]:

        domain = ICUDomain(y_dim=y_dim)
        test_env = env_dict[env](domain,load=False)

        data = standard_dataset(domain)
        print(f'{env}: {y_dim}')
        test_env.train(data,batch_size=64)
        #test_env.model.save_model()
'''

pol_dict = {'LSTM':LSTMPol,'Linear':LinearPol,'MLP':MLPPol}
for pol in ['LSTM']:
    for y_dim in [2]:

        domain = ICUDomain(y_dim=y_dim)
        test_pol = pol_dict[pol](domain,load=False)

        data = standard_dataset(domain)
        validation_data = standard_dataset(domain,test=True).get_whole_batch()
        print(f'{pol}: {y_dim}')
        test_pol.fit(data, batch_size=64, validation_set=validation_data)
        #test_pol.model.save_model()


'''
domain = WardDomain(y_dim=8)

# Would normally load a pretrained model, set load=False so it doesn't
test_env = StateSpaceEnv(domain,load=False)
test_pol = LSTMPol(domain,load=False)
test_init = VAEInit(domain,load=False)

data = standard_dataset(domain)

# Note hyperparameters for training including learning rate and epochs are
# stored in domain.{env,pol,init}_config
test_env.train(data,batch_size=64)

# Save model by uncommenting
test_env.model.save_model()

#test_pol.train(data,batch_size=64)
#test_pol.model.save_model()


#test_init.train(data,batch_size=64)
#test_init.model.save_model()
'''