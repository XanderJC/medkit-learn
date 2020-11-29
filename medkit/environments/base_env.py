from .__head__ import *

class BaseEnv(gym.Env):
    '''
    Base Environment class
    '''
    def __init__(self,domain):
        super(BaseEnv, self).__init__()
        self.name = None
        self.model_config = domain.get_env_config(self.name)
        # model_config is a dictionary of hyperparameters (e.g. layer sizes)
        # for the model

        return

    def unpack_domain(self,domain):
        '''
        Set meta-data and hyperparameters
        '''
        return

    @abstractmethod
    def load_pretrained(self):
        pass

    @abstractmethod
    def train(self):
        pass

    '''
    Envrionment inherits from the gym.Env class: the following are the required
    methods 
    '''
    def step(self, action):

        observation = None
        reward = None
        done = None
        info = None

        return observation,reward,info,done

    def reset(self):

        raise NotImplemenedError

    def render(self, mode='human', close=False):

        raise NotImplemenedError
