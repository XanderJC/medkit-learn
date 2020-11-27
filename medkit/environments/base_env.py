

class BaseEnv():
    '''
    Base Environment class
    '''
    def __init__(self,domain):
        
        self.name = None

        self.model_config = domain.get_env_config(self.name)

        return

    def unpack_domain(self,domain):
        
        return

    def load_pretrained(self):

        return
    
    def next_state(self):

        return

    def train(self):

        return
