from .__head__ import *


class MixturePol(BasePol):
    def __init__(self, domain, pol_list, mixing_prob):

        self.name = "mixture"

        self.domain = domain
        self.policies = [pol(domain) for pol in pol_list]
        self.mixing = torch.distributions.categorical.Categorical(probs=mixing_prob)

    def select_action(self, history, stochastic=False, temperature=1.0):

        pol = self.policies[self.mixing.sample()]
        action = pol.select_action(history, stochastic, temperature)

        return action
