import numpy as np
import gym
from gym import spaces
import warnings
from tqdm import tqdm
import torch

from medkit.environments.RNN import RNNEnv
from medkit.domains.ICU import ICUDomain
from medkit.policies.RNNp import RNNPol

class scenario(gym.Env):
    def __init__(self,domain,environment,policy):

        dom_dict = {'ICU':ICUDomain}
        env_dict = {'RNN':RNNEnv}
        pol_dict = {'RNN':RNNPol}

        assert domain in dom_dict, 'Not a valid domain.'
        assert environment in env_dict, 'Not a valid envrionment.'
        assert policy in pol_dict, 'Not a valid policy.'

        self.dom = dom_dict[domain]()
        self.env = env_dict[environment](self.dom)
        self.pol = pol_dict[policy](self.dom)

    def batch_generate(self,num_trajs=100,max_seq_length=50):
        static_data = np.zeros((num_trajs,self.dom.static_in_dim))
        series_data = np.zeros((num_trajs,max_seq_length,self.dom.series_in_dim))
        action_data = np.zeros((num_trajs,max_seq_length,1))

        for i in tqdm(range(num_trajs),desc = 'Generating trajectories'):

            static_obs,observation,info = self.reset()

            done = False
            seq_length = 1
            while not (done & (seq_length < max_seq_length)):

                action = info['predicted_action']
                observation,reward,info,done = self.step(action)
                seq_length += 1

            prev_obs,prev_acts = self.history
 
            static_data[i,:] = static_obs
            series_data[i,:seq_length-1,:] = prev_obs[:,:seq_length-1,:]
            action_data[i,:seq_length-1,:] = prev_acts[:,:seq_length-1,:]

        return static_data,series_data,action_data

    def step(self,action):
        observation,reward,info,done = self.env.step(action)

        prev_obs,prev_acts = self.history
        if prev_acts is not None:
            prev_acts = torch.cat((prev_acts,action.reshape(1,1,1)),1)
        else:
            prev_acts = action.reshape((1,1,1))
        prev_obs = torch.cat((prev_obs,observation.reshape(1,1,self.dom.series_in_dim)),1)
        self.history = (prev_obs,prev_acts)

        pred_action = self.pol.select_action(self.history)
        info = {'predicted_action':pred_action}

        return observation,reward,info,done

    def reset(self):
        static_obs,observation = self.env.reset()
        self.history = (observation.reshape(1,1,self.dom.series_in_dim),None)

        pred_action = self.pol.select_action(self.history)
        info = {'predicted_action':pred_action}

        return static_obs,observation,info



def generate_data(domain        = 'ICU',
                    environment = 'RNN',
                    policy      = 'RNN',
                    size        = 100,
                    test_split  = False,
                    **kwargs):
    '''
    Base API function for generating a batch dataset 

    '''
    scene = scenario(domain,environment,policy)

    data = scene.batch_generate(**kwargs)

    return data


if __name__ == '__main__':

    data_total = generate_data()
    print(data_total[0].shape)
    print(data_total[1].shape)
    print(data_total[2].shape)
