

class BaseReward():
    def __init__(self):
        
        self.state_only = True
    
    def get_reward(self,state,action):

        reward = None

        return reward