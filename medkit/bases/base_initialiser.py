from .__head__ import *
from .base_model import BaseModel


class BaseInit(ABC):
    """
    Base Initialiser class
    """

    def __init__(self, domain):
        super(BaseInit, self).__init__()
        self.name = None
        # self.model_config = domain.get_init_config(self.name)
        self.model_config = None
        # model_config is a dictionary of hyperparameters (e.g. layer sizes)
        # for the model

        self.model = BaseModel  # torch.nn.Module which for unified save/load/train
        return

    def save_model(self):
        path = resource_filename(
            "medkit",
            f"{self.form}/saved_models/{self.domain.base_name}_{self.name}.pth",
        )
        torch.save(self.model.state_dict(), path)

    def load_pretrained(self):
        path = resource_filename(
            "medkit",
            f"initialisers/saved_models/{self.domain.base_name}_{self.name}.pth",
        )
        self.model.load_state_dict(torch.load(path))
        pass

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return

    @abstractmethod
    def sample(self):
        pass
