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

        self.model = None # torch.nn.Module which for unified save/load/train
        return

    def unpack_domain(self,domain):
        '''
        Set meta-data and hyperparameters
        '''
        return

    def load_pretrained(self):
        path = resource_filename("policies",f"saved_models/{self.domain.name}_{self.name}.pth")
        self.model.load_state_dict(torch.load(path))
        pass
    
    def train(self,data_loader):
        self.model.train(data_loader)
        return

    @abstractmethod
    def select_action(self,history):
        pass