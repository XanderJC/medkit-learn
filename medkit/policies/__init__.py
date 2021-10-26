from .Linear import LinearPol
from .LSTM import LSTMPol
from .Mixture import MixturePol
from .MLP import MLPPol


def get_pol(pol):

    pol_dict = {
        "LSTM": LSTMPol,
        "Linear": LinearPol,
        "MLP": MLPPol,
        "Mixture": MixturePol,
    }

    return pol_dict[pol]
