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

    def load_pretrained(self):
        path = resource_filename("policies",f"saved_models/{self.domain.name}_{self.name}.pth")
        self.model.load_state_dict(torch.load(path))
        pass

    def save_model(self):
        path = resource_filename("policies",f"saved_models/{self.domain.name}_{self.name}.pth")
        torch.save(self.model.state_dict(), path)
    
    def train(self,dataset,batch_size=128):
        self.model.train(dataset,batch_size=batch_size)
        return

    @abstractmethod
    def select_action(self,history):
        pass