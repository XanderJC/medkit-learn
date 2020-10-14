from .__head__ import *

class ToyEnv(gym.Env):
    '''
    Super basic, 2 diseases (3 states) and four actions (TreatA,TreatB,discharge,Test)
    Designed to test until confident of diagnosis, test only correct 70% of time.
    Reward only for correct diagnosis and treatement. Diagnosis ends simulation. 
    '''
    def __init__(self):
        super(ToyEnv, self).__init__()

        self.action_space = spaces.Discrete(4)
        self.state_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(3)


    def step(self, action):

        self.t += 1

        self.state = self._next_state(action)
        observation = self._observe(action)
        reward = self._get_reward(action)
        done = action != 3
        info = None

        return observation,reward,done,info

    def reset(self):

        self.state = self.state_space.sample()
        self.t = 0
        return None 

    def render(self, mode='human', close=False):

        raise NotImplemenedError

    def _next_state(self, action):

        return self.state

    def _observe(self, action):

        if action == 3:

            test_accurate = bool(np.random.binomial(1,0.7))
            if test_accurate:
                obs = self.state
            else:
                obs = self.state_space.sample()

        else:
            obs = self.state

        return obs

    def _get_reward(self, action):

        if action == 3:
            reward = 0
        else:
            if self.state == action:
                reward = 1
            else:
                reward = 0

        return reward