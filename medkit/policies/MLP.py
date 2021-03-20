from .__head__ import *


class mlp_pol(BaseModel):
    def __init__(self, domain):
        super(mlp_pol, self).__init__(domain, "policies")
        self.name = "mlp"
        self.input_size = domain.series_in_dim
        self.hidden_size = domain.pol_config["hidden_dim"]
        self.hidden_layers = domain.pol_config["hidden_layers"]
        self.in_layer = nn.Linear(self.input_size, self.hidden_size)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.hidden_layers)
            ]
        )
        self.out_layer = nn.Linear(self.hidden_size, domain.y_dim)

    def forward(self, x):

        x = self.in_layer(x)
        for layer in self.linears:
            x = layer(x)
            x = F.elu(x)
        x = self.out_layer(x)

        return x

    def loss(self, batch):
        x_static, x_series, mask, y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]

        pred = F.softmax(self.forward(x_series), 2)

        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.masked_select(mask.bool()).mean()


class MLPPol(BasePol):
    def __init__(self, domain, load=True):

        self.name = "mlp"

        self.domain = domain
        self.domain.get_pol_config(self.name)

        self.model = mlp_pol(domain)
        if load:
            self.load_pretrained()
            self.model.eval()

    def select_action(self, history, stochastic=False, temperature=1.0):

        prev_obs, prev_acts = history
        pred = F.softmax(self.model.forward(prev_obs)[:, -1] / temperature, 1)

        if stochastic:
            act = torch.distributions.categorical.Categorical(probs=pred)
            action = act.sample()
        else:
            action = torch.argmax(pred)

        return action
