from .__head__ import *
from .base_model import BaseModel
from .base_reward import BaseReward


class BaseEnv(gym.Env):
    """
    Base Environment class.
    """

    def __init__(self, domain):
        super(BaseEnv, self).__init__()
        self.name = None  # unique string
        # self.model_config = domain.get_env_config(self.name)
        # model_config is a dictionary of hyperparameters (e.g. layer sizes)
        # for the model
        self.model = BaseModel  # torch.nn.Module which for unified save/load/train

        self.reward = BaseReward()

        return

    def load_pretrained(self):
        """
        Load pretrained model.
        """
        path = resource_filename(
            "medkit", f"environments/saved_models/{self.domain.name}_{self.name}.pth"
        )
        self.model.load_state_dict(torch.load(path))

        return

    def save_model(self):
        path = resource_filename(
            "medkit", f"environments/saved_models/{self.domain.name}_{self.name}.pth"
        )
        torch.save(self.model.state_dict(), path)

    def fit(self, *args, **kwargs):
        self.model.fit(*args, **kwargs)
        return

    """
    Envrionment inherits from the gym.Env class: the following are the required
    methods.
    """

    def step(self, action):
        """
        Standard gym.Env.step() - no reward atm.
        """
        observation = None
        reward = None
        done = None
        info = None

        return observation, reward, info, done

    def reset(self):
        """
        Returns both the static features of the patient trjaectory as well as the first
        time series observation.
        """
        static_obs = None
        observation = None
        return static_obs, observation

    def render(self, mode="human", close=False):

        raise NotImplementedError
