from .CounterfactualRNN import CRNEnv
from .SequentialVAE import SVAEEnv
from .StateSpace import StateSpaceEnv
from .TForce import TForceEnv


def get_env(env):

    env_dict = {
        "TForce": TForceEnv,
        "SVAE": SVAEEnv,
        "StateSpace": StateSpaceEnv,
        "CRN": CRNEnv,
    }

    return env_dict[env]
