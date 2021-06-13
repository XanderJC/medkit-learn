from .icu import ICUDomain
from .ward import WardDomain
from .cf import CFDomain
from .mimic import MimicDomain


def get_domain(domain, y_dim):

    dom_dict = {
        "icu": ICUDomain,
        "cf": CFDomain,
        "ward": WardDomain,
        "mimic": MimicDomain,
    }

    return dom_dict[domain](y_dim=y_dim)
