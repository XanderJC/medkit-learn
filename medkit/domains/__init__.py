from .cf import CFDomain
from .icu import ICUDomain
from .mimic import MimicDomain
from .ward import WardDomain


def get_domain(domain, y_dim):

    dom_dict = {
        "icu": ICUDomain,
        "cf": CFDomain,
        "ward": WardDomain,
        "mimic": MimicDomain,
    }

    return dom_dict[domain](y_dim=y_dim)
