from .__head__ import *
from .base_model import BaseModel

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

        self.model = BaseModel() # torch.nn.Module which for unified save/load/train
        return

    def load_pretrained(self):
        path = resource_filename("medkit",f"initialisers/saved_models/{self.domain.base_name}_{self.name}.pth")
        self.model.load_state_dict(torch.load(path))
        pass
    
    def train(self,*args,**kwargs):
        self.model.train(*args,**kwargs)
        return

    @abstractmethod
    def sample(self):
        pass