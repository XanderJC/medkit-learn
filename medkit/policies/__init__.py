from .LSTM import LSTMPol
from .Linear import LinearPol
from .MLP import MLPPol
from .Mixture import MixturePol


def get_pol(pol):

    pol_dict = {
        "LSTM": LSTMPol,
        "Linear": LinearPol,
        "MLP": MLPPol,
        "Mixture": MixturePol,
    }

    return pol_dict[pol]
