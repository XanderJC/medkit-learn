from .__head__ import *


class RNN_pol(BaseModel):
    def __init__(self, domain):
        super(RNN_pol, self).__init__(domain, "policies")
        self.name = "lstm"

        self.hidden_size = self.hyper["hidden_dim"]
        self.num_layers = self.hyper["hidden_layers"]
        self.lstm_layers = self.hyper["lstm_layers"]
        self.dropout = self.hyper["dropout"]
        self.input_size = domain.series_in_dim

        self.lstm = opacus.layers.DPLSTM(
            self.input_size,
            self.hidden_size,
            self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        self.fc = nn.Linear(self.hidden_size, domain.y_dim)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size)

        # Forward LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        for layer in self.linears:
            out = layer(out)
            out = F.elu(out)

        pred = self.fc(out)
        return pred

    def loss(self, batch):
        x_static, x_series, mask, y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]

        pred = F.softmax(self.forward(x_series), 2)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.masked_select(mask.bool()).mean()


class LSTMPol(BasePol):
    def __init__(self, domain, load=True):

        self.name = "lstm"

        self.domain = domain
        self.domain.get_pol_config(self.name)

        self.model = RNN_pol(domain)
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
