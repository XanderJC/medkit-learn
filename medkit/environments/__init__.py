from .TForce import TForceEnv
from .SequentialVAE import SVAEEnv
from .StateSpace import StateSpaceEnv
from .CounterfactualRNN import CRNEnv


def get_env(env):

    env_dict = {
        "TForce": TForceEnv,
        "SVAE": SVAEEnv,
        "StateSpace": StateSpaceEnv,
        "CRN": CRNEnv,
    }

    return env_dict[env]
