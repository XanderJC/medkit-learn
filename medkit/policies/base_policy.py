
class BasePol():
    '''
    Base Policy class
    '''
    def __init__(self,domain):
        
        self.name = None
        self.model_config = domain.get_pol_config(self.name)
        # model_config is a dictionary of hyperparameters (e.g. layer sizes)
        # for the model 
        return

    def unpack_domain(self,domain):
        
        return

    def load_pretrained(self):

        return

    def train(self):

        return

    def select_action(self,state):

        return