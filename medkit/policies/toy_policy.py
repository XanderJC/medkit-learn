from .__head__ import *

class ToyPolicy():
    '''
    Toy policy for our toy environment. Keep testing until confident enough.
    '''
    def __init__(self,
                env             : gym.Env,
                n_diseases      : int,
                conf_threshold  : float,
                test_acc_belief : float,
                **kwargs):

        self.env = env        
        self.n_diseases = n_diseases
        self.states = n_diseases + 1
        self.num_actions = n_diseases + 2 
        self.conf_threshold = conf_threshold
        self.test_acc_belief = test_acc_belief

        self.belief = np.ones(self.states) / self.states
        self.buffer = []

    def batch_generator(self,num_trajs):
        
        self.buffer = []
        for n in range(num_trajs):
            self.rollout(save_in_buffer = True)

        return self.buffer

    def rollout(self,save_in_buffer = False):
        
        traj = []
        self.belief = np.ones(self.states) / self.states
        env = self.env
        env.reset()

        action = self._select_action(self.belief)
        observation,reward,info,done = env.step(action)
        traj.append((observation,action,reward))

        while not done:
            
            likelihood = np.ones(self.states) / self.states
            likelihood[observation] = (likelihood[observation] * \
                (1-self.test_acc_belief)) + self.test_acc_belief
            self.belief = self.belief * likelihood
            self.belief = self.belief / self.belief.sum()

            action = self._select_action(self.belief)
            next_observation,reward,info,done = env.step(action)

            traj.append((observation,action,reward))
            observation = next_observation

        if save_in_buffer:
            self.buffer.append(traj)

    def _select_action(self,belief):
        
        if np.max(belief) < self.conf_threshold:
            action = self.num_actions - 1

        else:
            action = np.argmax(belief) 
        return action
