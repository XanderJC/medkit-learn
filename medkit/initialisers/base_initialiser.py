from .__head__ import *

class BaseInit(ABC):
    '''
    Base Initialiser class
    '''
    def __init__(self,domain):
        
        self.name = None
        #self.model_config = domain.get_init_config(self.name)
        self.model_config = None
        # model_config is a dictionary of hyperparameters (e.g. layer sizes)
        # for the model 

        self.model = None # torch.nn.Module which for unified save/load/train
        return

    def load_pretrained(self):
        path = resource_filename("initialisers",f"saved_models/{self.domain.name}_{self.name}.pth")
        self.model.load_state_dict(torch.load(path))
        pass
    
    def train(self,data_loader):
        self.model.train(data_loader)
        return

    @abstractmethod
    def sample(self):
        pass