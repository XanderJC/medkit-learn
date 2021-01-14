from medkit.scenario import scenario
from medkit.tools import scaler

from medkit.environments import * 
from medkit.domains import * 
from medkit.policies import * 

import pandas as pd
import numpy as np
import torch 


def batch_generate(domain       = 'ICU',
                    environment = 'SVAE',
                    policy      = 'LSTM',
                    actions     = 2,
                    size        = 100,
                    valid_size  = None,
                    test_size   = None,
                    max_length  = 50,
                    scale       = False,
                    out         = 'numpy',
                    confounders = None,
                    overlooked  = None,
                    stochastic  = False,
                    variation   = 1.0,
                    seed        = None,
                    **kwargs):
    '''
    Base API function for generating a batch dataset 
    '''

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    dom_dict = {'ICU':ICUDomain,'Ward':WardDomain,'CF':CFDomain}
    env_dict = {'TForce':TForceEnv,'SVAE':SVAEEnv,'StateSpace':StateSpaceEnv,'CRN':CRNEnv}
    pol_dict = {'LSTM':LSTMPol,'Linear':LinearPol,'MLP':MLPPol}

    if type(domain) is str:
        assert domain in dom_dict, 'Not a valid domain.'
        dom = dom_dict[domain](y_dim=actions)
    else:
        assert issubclass(type(domain),BaseDomain)
        dom = domain

    if type(environment) is str:
        assert environment in env_dict, 'Not a valid environment.'
        env = env_dict[environment](dom)
    else:
        assert issubclass(type(environment),BaseEnv)
        env = environment

    if type(policy) is str:
        assert policy in pol_dict, 'Not a valid policy.'
        pol = pol_dict[policy](dom)
    else:
        assert issubclass(type(policy),BasePol)
        pol = policy


    # Build scenario 
    scene = scenario(
                domain      = dom,
                environment = env,
                policy      = pol,
                confounders = confounders,
                overlooked  = overlooked,
                stochastic  = stochastic,
                variation   = variation
    )           

    '''
    Produce the appropriate data.
    '''
    assert (type(size) is int)  & (size > 0), 'size must be a positive integer.'
    print('Producing training data...')
    training_data = scene.batch_generate(num_trajs=size,max_seq_length=max_length)
    data = {'training':training_data}

    if valid_size is not None:
        assert (type(valid_size) is int) & (valid_size > 0), 'valid_size must be a positive integer.'
        print('Producing validation data...')
        valid_data = scene.batch_generate(num_trajs=valid_size,max_seq_length=max_length)
        data['validation'] = valid_data

    if test_size is not None:
        assert (type(test_size) is int) & (test_size > 0), 'test_size must be a positive integer.'
        print('Producing testing data...')
        test_data = scene.batch_generate(num_trajs=test_size,max_seq_length=max_length)
        data['testing'] = test_data

    '''
    Rescale if appropriate
    '''
    if not scale:
        data_scale = scaler(dom)
        data_scale.load_params()

        for split in data.keys():
            static,series,actions = data[split]
            static = torch.tensor(static)
            series = torch.tensor(series)
            static = data_scale.rescale_static(static).detach().numpy()
            mask = (series[:,:,0] != 0)
            series = data_scale.rescale_series(series,mask).detach().numpy()

            data[split] = (static,series,actions)

    '''
    Re-format for appropriate output.
    '''
    valid_output = set(['numpy','pandas','csv'])
    assert out in valid_output, 'Invalid output method.'

    if out == 'numpy':
        pass
    else:
        for split in data.keys():
            data_split = data[split]

            static = data_split[0]
            n = static.shape[0]
            static = np.concatenate(((np.array(range(n))+1).reshape((n,1)),static),1)
            columns = ['id'] + scene.dom.static_names
            static = pd.DataFrame(static,columns=columns)

            series = data_split[1]
            actions = data_split[2]
            mask = (series[:,:,0] != 0)
            total = mask.sum()
            p = scene.dom.out_dim
            
            stacked_series = np.zeros((total,(p+3)))
            index = 0
            for i,traj in enumerate(series):
                length = mask[i].sum()
                stacked_series[index:index+length,0] = i+1
                stacked_series[index:index+length,1] = np.array(range(length))
                stacked_series[index:index+length,2:-1] = traj[:length,:]
                stacked_series[index:index+length,-1] = actions[i,:length].reshape(length)
                index += length
            
            action_name = ''
            for name in scene.dom.action_names:
                action_name += f'{name} - '
            action_name = action_name[:-3]

            columns = ['id','time'] + scene.dom.series_names + [action_name]

            series = pd.DataFrame(stacked_series,columns=columns)

            if out == 'csv':
                series_name = f'{scene.dom.name}_series_{split}_data.csv'
                static_name = f'{scene.dom.name}_static_{split}_data.csv'

                series.to_csv(series_name)
                static.to_csv(static_name)


            data_split = (static,series)
            data[split] = data_split

    return data

def live_simulate(domain        = 'ICU',
                    environment = 'RNN',
                    policy      = 'RNN',
                    **kwargs):

    dom_dict = {'ICU':ICUDomain}
    env_dict = {'RNN':RNNEnv}
    pol_dict = {'RNN':RNNPol}

    if type(domain) is str:
        assert domain in dom_dict, 'Not a valid domain.'
        dom = dom_dict[domain]()
    else:
        assert issubclass(type(domain),BaseDomain)
        dom = domain

    if type(environment) is str:
        assert environment in env_dict, 'Not a valid environment.'
        env = env_dict[environment](dom)
    else:
        assert issubclass(type(environment),BaseEnv)
        env = environment

    if type(policy) is str:
        assert policy in pol_dict, 'Not a valid policy.'
        pol = pol_dict[policy](dom)
    else:
        assert issubclass(type(policy),BasePol)
        pol = policy

    # Build scenario 
    scene = scenario(dom,env,pol)

    return scene

if __name__ == '__main__':


    data = batch_generate(domain='Ward',environment='TForce',actions=2,max_length=30,
                            size=100,scale=True,seed=41310)
    
    

    
