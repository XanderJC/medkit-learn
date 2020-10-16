import numpy as np
import gym
from gym import spaces
import warnings

from .environments.toy_env import ToyEnv
from .policies.toy_policy import ToyPolicy


def generate_data(setting = 'Optimal Stopping',
                    policy = 'Tester',
                    size = 100,
                    test_split = False,
                    **kwargs):

    env_dict = {'Optimal Stopping':ToyEnv}
    policy_dict = {'Tester':ToyPolicy}

    assert setting in env_dict, 'Not a valid envrionment.'
    assert policy in policy_dict, 'Not a valid policy.'

    env = env_dict[setting](**kwargs)
    agent = policy_dict[policy](env,**kwargs)

    data = agent.batch_generator(size)

    if test_split:
        indx = int(np.ceil(len(data)*0.8))
        train_data = data[:indx]
        test_data = data[indx:]

        data = {'training':train_data,'testing':test_data}

    return data

def make_gym(setting = 'Optimal Stopping',
                **kwargs):

    env_dict = {'Optimal Stopping':ToyEnv}
    env = env_dict[setting](**kwargs)

    return env

if __name__ == '__main__':

    data_total = generate_data(test_split       = True,
                                n_diseases      = 6,
                                test_accuracy   = 0.3,
                                conf_threshold  = 0.8,
                                test_acc_belief = 0.3)

    data = data_total['testing']
    print(len(data))
    print(data[0])  

