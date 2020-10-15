from .__head__ import *

class ToyEnv(gym.Env):
    '''
    Super basic, partially observed optimal stopping problem. Set the number of diseases
    (will be an additional 'healthy' state) and the accuracy of the test in [0,1].
    Strategy would be to keep testing until confident enough in diagnosis.
    '''
    def __init__(self,
                n_diseases    : int,
                test_accuracy : float):
        super(ToyEnv, self).__init__()

        self.action_space = spaces.Discrete(n_diseases + 2)
        self.state_space = spaces.Discrete(n_diseases + 1)
        self.observation_space = spaces.Discrete(n_diseases + 1)

        self.test_acc = test_accuracy

        self.state = self.state_space.sample()
        self.t = 0

    def step(self, action):

        self.t += 1
        self.state = self._next_state(action)

        observation = self._observe(action)
        reward = self._get_reward(action)
        done = action != (self.action_space.n - 1)
        info = None

        return observation,reward,info,done

    def reset(self):

        self.state = self.state_space.sample()
        self.t = 0
        return None 

    def render(self, mode='human', close=False):

        raise NotImplemenedError

    def _next_state(self, action):

        return self.state

    def _observe(self, action):

        if action == (self.action_space.n - 1):
            test_accurate = bool(np.random.binomial(1,self.test_acc))
            if test_accurate:
                obs = self.state
            else:
                obs = self.state_space.sample()

        else:
            obs = self.state

        return obs

    def _get_reward(self, action):

        if action == self.action_space.n - 1:
            reward = 0

        else:
            if action == self.state :
                reward = 1
            else:
                reward = 0

        return reward