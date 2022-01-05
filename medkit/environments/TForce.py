from medkit.initialisers import VAEInit

from .__head__ import *


class RNN_env(BaseModel):
    def __init__(self, domain):
        super(RNN_env, self).__init__(domain, "environments")
        self.name = "tforce"
        self.hidden_size = self.hyper["hidden_dim"]
        self.num_layers = self.hyper["hidden_layers"]
        self.lstm_layers = self.hyper["lstm_layers"]
        self.dropout = self.hyper["dropout"]

        self.input_size = domain.series_in_dim + domain.y_dim + domain.static_in_dim

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
        self.fc_bin = nn.Linear(self.hidden_size, domain.bin_out_dim)
        self.fc_cont = nn.Linear(self.hidden_size, 2 * domain.con_out_dim)

        self.bin_out_dim = domain.bin_out_dim
        self.con_out_dim = domain.con_out_dim

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size)  # .to(device)
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size)  # .to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        for layer in self.linears:
            out = layer(out)
            out = F.elu(out)

        cont_params = self.fc_cont(out)
        bin_params = torch.sigmoid(self.fc_bin(out))
        return cont_params, bin_params

    def loss(self, batch):
        x_static, x_series, mask, y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]
        mask = mask[:, 1:].reshape((batch_size, seq_length - 1, 1))

        y_one_hot = F.one_hot(y_series.long(), self.domain.y_dim)

        concat = torch.cat(
            (x_series, y_one_hot, x_static.unsqueeze(1).expand((-1, seq_length, -1))), 2
        )
        inputs = concat[:, :-1, :]
        outputs = x_series[:, 1:, :]

        cont_pars, bin_pars = self.forward(inputs)

        cont_dist = torch.distributions.normal.Normal(
            cont_pars[:, :, : self.con_out_dim],
            F.softplus(cont_pars[:, :, self.con_out_dim :]),
        )
        cont_log_l = cont_dist.log_prob(outputs[:, :, : self.con_out_dim])

        bin_dist = torch.distributions.bernoulli.Bernoulli(bin_pars)
        bin_log_l = bin_dist.log_prob(outputs[:, :, self.con_out_dim :])

        log_l = (cont_log_l * mask.expand((-1, -1, self.con_out_dim))).sum() + (
            bin_log_l * mask.expand((-1, -1, self.bin_out_dim))
        ).sum()

        return -log_l / mask.sum()


class TForceEnv(BaseEnv):
    def __init__(self, domain, load=True):
        super(TForceEnv, self).__init__(domain)
        self.name = "tforce"

        self.domain = domain
        self.domain.get_env_config(self.name)

        self.model = RNN_env(domain)
        if load:
            self.load_pretrained()

        self.initialiser = VAEInit(domain)

    def step(self, action):

        action = action.reshape((1, 1))
        action_one_hot = F.one_hot(action, self.domain.y_dim)

        x = torch.cat((self.prev_obs, action_one_hot, self.static.expand(1, 1, -1)), 2)

        out, (self.hn, self.cn) = self.model.lstm(x, (self.hn, self.cn))

        cont_pars = self.model.fc_cont(out[:, -1, :])
        bin_pars = torch.sigmoid(self.model.fc_bin(out[:, -1, :]))

        cont_dist = torch.distributions.normal.Normal(
            cont_pars[:, : self.domain.con_out_dim],
            F.softplus(cont_pars[:, self.domain.con_out_dim :]),
        )
        cont_sample = cont_dist.sample()

        bin_dist = torch.distributions.bernoulli.Bernoulli(bin_pars)
        bin_sample = bin_dist.sample()

        observation = torch.cat((cont_sample, bin_sample), 1)
        reward = self.reward.get_reward(observation)
        info = None
        done = (
            torch.distributions.bernoulli.Bernoulli(self.domain.terminate)
            .sample()
            .bool()
        )

        self.prev_obs = observation.reshape((1, 1, self.domain.series_in_dim))

        return observation.reshape((self.domain.series_in_dim)), reward, info, done

    def reset(self):

        # Initialise LSTM hidden states
        self.hn = torch.zeros(self.model.lstm_layers, 1, self.model.hidden_size)
        self.cn = torch.zeros(self.model.lstm_layers, 1, self.model.hidden_size)

        init_obs, static_obs = self.initialiser.sample()
        self.prev_obs = init_obs

        self.static = static_obs

        return static_obs.reshape((self.domain.static_in_dim)), init_obs.reshape(
            (self.domain.series_in_dim)
        )

    def render(self):

        obs = list(self.prev_obs.reshape((self.domain.series_in_dim)))

        for name, value in zip(self.domain.series_names, obs):
            print(f"{name}: {value}")
