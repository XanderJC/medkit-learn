from .__head__ import *

class BasePol(ABC):
    '''
    Base Policy class
    '''
    def __init__(self,domain):
        
        self.name = None
        self.model_config = domain.get_pol_config(self.name)
        # model_config is a dictionary of hyperparameters (e.g. layer sizes)
        # for the model 

        self.unpack_domain(domain)
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

    @abstractmethod
    def select_action(self,history):
        pass