from .__head__ import *
from .base_model import BaseModel


class BasePol(ABC):
    """
    Base Policy class
    """

    def __init__(self, domain):
        super(BasePol, self).__init__()
        self.name = None
        self.model_config = domain.get_pol_config(self.name)
        # model_config is a dictionary of hyperparameters (e.g. layer sizes)
        # for the model

        self.model = BaseModel  # torch.nn.Module which for unified save/load/train
        return

    def load_pretrained(self):
        path = resource_filename(
            "medkit", f"policies/saved_models/{self.domain.name}_{self.name}.pth"
        )
        self.model.load_state_dict(torch.load(path))
        pass

    def save_model(self):
        path = resource_filename(
            "medkit", f"policies/saved_models/{self.domain.name}_{self.name}.pth"
        )
        torch.save(self.model.state_dict(), path)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return

    @abstractmethod
    def select_action(self, history):
        pass
