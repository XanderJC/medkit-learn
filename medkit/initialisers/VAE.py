from .__head__ import *


class Encoder(nn.Module):
    def __init__(self, domain):
        super(Encoder, self).__init__()
        self.hyper = domain.init_config
        self.in_dim = domain.static_in_dim + domain.series_in_dim

        self.linear1 = nn.Linear(self.in_dim, self.hyper["hidden_units"])
        self.linear2 = nn.Linear(self.hyper["hidden_units"], self.hyper["hidden_units"])
        self.mean_head = nn.Linear(
            self.hyper["hidden_units"], self.hyper["latent_size"]
        )
        self.lstd_head = nn.Linear(
            self.hyper["hidden_units"], self.hyper["latent_size"]
        )

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))
        means = self.mean_head(x)
        lstds = self.lstd_head(x)

        return means, lstds


class Decoder(nn.Module):
    def __init__(self, domain):
        super(Decoder, self).__init__()
        self.hyper = domain.init_config
        self.out_dim = domain.static_in_dim + domain.series_in_dim
        self.domain = domain

        self.linear1 = nn.Linear(self.hyper["latent_size"], self.hyper["hidden_units"])
        self.linear2 = nn.Linear(self.hyper["hidden_units"], self.hyper["hidden_units"])

        self.series_cont_mean = nn.Linear(
            self.hyper["hidden_units"], self.domain.con_out_dim
        )
        self.series_cont_lstd = nn.Linear(
            self.hyper["hidden_units"], self.domain.con_out_dim
        )
        self.series_bin = nn.Linear(self.hyper["hidden_units"], self.domain.bin_out_dim)
        self.static_cont_mean = nn.Linear(
            self.hyper["hidden_units"], self.domain.static_con_dim
        )
        self.static_cont_lstd = nn.Linear(
            self.hyper["hidden_units"], self.domain.static_con_dim
        )
        self.static_bin = nn.Linear(
            self.hyper["hidden_units"], self.domain.static_bin_dim
        )

    def forward(self, x):
        x = F.elu(self.linear1(x))
        x = F.elu(self.linear2(x))

        series_mean = self.series_cont_mean(x)
        series_lstd = self.series_cont_lstd(x)
        series_prob = torch.sigmoid(self.series_bin(x))

        static_mean = self.static_cont_mean(x)
        static_lstd = self.static_cont_lstd(x)
        static_prob = torch.sigmoid(self.static_bin(x))

        return (
            series_mean,
            series_lstd,
            series_prob,
            static_mean,
            static_lstd,
            static_prob,
        )


class VAE(BaseModel):
    def __init__(self, domain):
        super(VAE, self).__init__(domain, "initialisers")
        self.name = "VAE"
        self.hyper = domain.init_config

        self.encoder = Encoder(domain)
        self.decoder = Decoder(domain)

    def kl_loss(self, z_mean, z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def loss(self, batch):
        x_static, x_series, mask, y_series = batch
        batch_size = x_static.shape[0]

        inputs = torch.cat((x_static, x_series[:, 0, :]), 1)

        latent_means, latent_lstds = self.encoder(inputs)
        kl_loss = self.kl_loss(latent_means, torch.exp(latent_lstds))
        latent_dist = torch.distributions.normal.Normal(
            latent_means, torch.exp(latent_lstds)
        )
        latents = latent_dist.rsample()

        (
            series_mean,
            series_lstd,
            series_prob,
            static_mean,
            static_lstd,
            static_prob,
        ) = self.decoder(latents)

        series_cont_dist = torch.distributions.normal.Normal(
            series_mean, torch.exp(series_lstd)
        )
        series_cont_log_l = series_cont_dist.log_prob(
            x_series[:, 0, :][:, : self.domain.con_out_dim]
        )

        series_bin_dist = torch.distributions.bernoulli.Bernoulli(series_prob)
        series_bin_log_l = series_bin_dist.log_prob(
            x_series[:, 0, :][
                :,
                self.domain.con_out_dim : (
                    self.domain.con_out_dim + self.domain.bin_out_dim
                ),
            ]
        )

        static_cont_dist = torch.distributions.normal.Normal(
            static_mean, torch.exp(static_lstd)
        )
        static_cont_log_l = static_cont_dist.log_prob(
            x_static[:, : self.domain.static_con_dim]
        )

        static_bin_dist = torch.distributions.bernoulli.Bernoulli(static_prob)
        static_bin_log_l = static_bin_dist.log_prob(
            x_static[
                :,
                self.domain.static_con_dim : (
                    self.domain.static_con_dim + self.domain.static_bin_dim
                ),
            ]
        )

        ll_loss = (
            series_cont_log_l.sum()
            + series_bin_log_l.sum()
            + static_cont_log_l.sum()
            + static_bin_log_l.sum()
        )

        return kl_loss - (ll_loss / batch_size)

    def save_model(self):
        path = resource_filename(
            "medkit",
            f"{self.form}/saved_models/{self.domain.base_name}_{self.name}.pth",
        )
        torch.save(self.state_dict(), path)


class VAEInit(BaseInit):
    def __init__(self, domain, load=True):
        super(VAEInit, self).__init__(domain)

        self.domain = domain
        self.name = "VAE"
        self.model_config = domain.get_init_config(self.name)
        self.model = VAE(domain)
        if load:
            self.load_pretrained()

    def sample(self):

        prior = torch.distributions.normal.Normal(loc=0, scale=1)
        latent = prior.sample([self.model_config["latent_size"]])

        (
            series_mean,
            series_lstd,
            series_prob,
            static_mean,
            static_lstd,
            static_prob,
        ) = self.model.decoder(latent)

        series_cont_dist = torch.distributions.normal.Normal(
            series_mean, torch.exp(series_lstd)
        )
        series_cont = series_cont_dist.sample()

        series_bin_dist = torch.distributions.bernoulli.Bernoulli(series_prob)
        series_bin = series_bin_dist.sample()

        static_cont_dist = torch.distributions.normal.Normal(
            static_mean, torch.exp(static_lstd)
        )
        static_cont = static_cont_dist.sample()

        static_bin_dist = torch.distributions.bernoulli.Bernoulli(static_prob)
        static_bin = static_bin_dist.sample()

        init_obs = torch.zeros((1, 1, self.domain.series_in_dim))
        static_obs = torch.zeros((1, self.domain.static_in_dim))

        init_obs[:, :, : self.domain.con_out_dim] = series_cont
        init_obs[
            :,
            :,
            self.domain.con_out_dim : self.domain.con_out_dim + self.domain.bin_out_dim,
        ] = series_bin

        static_obs[:, : self.domain.static_con_dim] = static_cont
        static_obs[
            :,
            self.domain.static_con_dim : self.domain.static_con_dim
            + self.domain.static_bin_dim,
        ] = static_bin

        return init_obs, static_obs
