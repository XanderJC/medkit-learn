import numpy as np
import torch
from torch.autograd import Function

from medkit.initialisers.VAE import VAEInit

from .__head__ import *


class CRN_env(BaseModel):
    def __init__(self, domain):
        super(CRN_env, self).__init__(domain, "environments")
        self.name = "CRN"
        self.hidden_size = self.hyper["hidden_dim"]
        self.num_layers = self.hyper["hidden_layers"]
        self.lstm_layers = self.hyper["lstm_layers"]
        self.dropout = self.hyper["dropout"]

        self.input_size = domain.series_in_dim + domain.y_dim
        self.y_dim = domain.y_dim

        self.lstm = opacus.layers.DPLSTM(
            self.input_size,
            self.hidden_size,
            self.lstm_layers,
            batch_first=True,
            dropout=self.dropout,
        )

        self.fc_bin = nn.Linear(self.hidden_size + self.y_dim, domain.bin_out_dim)
        self.fc_cont = nn.Linear(self.hidden_size + self.y_dim, 2 * domain.con_out_dim)

        self.action_fc_1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_out = nn.Linear(self.hidden_size, self.y_dim)

        self.bin_out_dim = domain.bin_out_dim
        self.con_out_dim = domain.con_out_dim

        self.domain = domain
        self.hyper = domain.env_config

    def forward(self, x):
        current_action = x[:, :, -self.y_dim :].reshape(
            shape=(x.size(0), x.size(1), self.y_dim)
        )
        previous_action = torch.cat(
            (torch.zeros(x.size(0), 1, self.y_dim), current_action[:, :-1, :]), dim=1
        )
        x = torch.cat((x[:, :, : -self.y_dim], previous_action), dim=-1)

        # Set initial hidden and cell states
        h0 = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.lstm_layers, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_size)

        balancing_rep = out[:, :, :]

        act_pred = GradientReversal()(balancing_rep)
        act_pred = F.relu(self.action_fc_1(act_pred))
        act_pred = self.action_out(act_pred)

        # Decode the hidden state of the last time step
        pred_input = torch.cat((balancing_rep, current_action), dim=-1)
        cont_params = self.fc_cont(pred_input)
        bin_params = torch.sigmoid(self.fc_bin(pred_input))
        return cont_params, bin_params, act_pred

    def loss(self, batch):
        x_static, x_series, mask, y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]
        mask = mask[:, 1:].reshape((batch_size, seq_length - 1, 1))

        y_one_hot = F.one_hot(y_series.long(), self.domain.y_dim)

        concat = torch.cat((x_series, y_one_hot), 2)
        inputs = concat[:, :-1, :]
        outputs = x_series[:, 1:, :]

        cont_pars, bin_pars, act_pred = self.forward(inputs)
        action_prob = F.softmax(act_pred, dim=-1)
        actions_true = y_one_hot[:, :-1, :]

        loss_action = (
            (-actions_true * torch.log(action_prob + 1e-8))
            * mask.expand((-1, -1, self.y_dim))
        ).sum()

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

        total_loss = -log_l + loss_action

        return total_loss / mask.sum()


class CRNEnv(BaseEnv):
    def __init__(self, domain, load=True):
        super(CRNEnv, self).__init__(domain)
        self.name = "CRN"

        self.domain = domain
        self.domain.get_env_config(self.name)

        self.model = CRN_env(domain)
        if load:
            self.load_pretrained()

        self.initialiser = VAEInit(domain)

    def step(self, action):
        action = action.reshape((1, 1))
        action_one_hot = F.one_hot(action, self.domain.y_dim)

        x = torch.cat((self.prev_obs, action_one_hot), 2)

        current_action = x[:, :, -self.domain.y_dim :].reshape(
            shape=(x.size(0), x.size(1), self.domain.y_dim)
        )
        previous_action = torch.cat(
            (torch.zeros(x.size(0), 1, self.domain.y_dim), current_action[:, :-1, :]),
            dim=1,
        )
        x = torch.cat((x[:, :, : -self.domain.y_dim], previous_action), dim=-1)

        out, (self.hn, self.cn) = self.model.lstm(x, (self.hn, self.cn))

        balancing_rep = out[:, -1, :]
        pred_input = torch.cat(
            (balancing_rep, action_one_hot.reshape(-1, self.domain.y_dim)), dim=-1
        )

        cont_pars = self.model.fc_cont(pred_input)
        bin_pars = torch.sigmoid(self.model.fc_bin(pred_input))

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
        done = torch.distributions.bernoulli.Bernoulli(0.1).sample().bool()

        self.prev_obs = observation.reshape((1, 1, self.domain.series_in_dim))

        return observation.reshape((self.domain.series_in_dim)), reward, info, done

    def reset(self):

        # Initialise LSTM hidden states
        self.hn = torch.zeros(self.model.lstm_layers, 1, self.model.hidden_size)
        self.cn = torch.zeros(self.model.lstm_layers, 1, self.model.hidden_size)

        init_obs, static_obs = self.initialiser.sample()
        self.prev_obs = init_obs

        return static_obs.reshape((self.domain.static_in_dim)), init_obs.reshape(
            (self.domain.series_in_dim)
        )

    def render(self):

        obs = list(self.prev_obs.reshape((self.domain.series_in_dim)))

        for name, value in zip(self.domain.series_names, obs):
            print(f"{name}: {value}")


# FROM https://github.com/jvanvugt/pytorch-domain-adaptation/
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
