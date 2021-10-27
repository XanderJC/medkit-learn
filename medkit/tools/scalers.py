from .__head__ import *


class scaler(nn.Module):
    def __init__(self, domain):
        super(scaler, self).__init__()
        self.name = "scaler"
        self.static_mean = nn.Parameter(
            torch.zeros([domain.static_in_dim]), requires_grad=False
        )
        self.static_std = nn.Parameter(
            torch.ones([domain.static_in_dim]), requires_grad=False
        )
        self.series_mean = nn.Parameter(
            torch.zeros([domain.series_in_dim]), requires_grad=False
        )
        self.series_std = nn.Parameter(
            torch.ones([domain.series_in_dim]), requires_grad=False
        )

        self.domain = domain
        return

    def fit_static(self, x_static):

        try:
            self.load_params()
        except:
            mean = torch.zeros([self.domain.static_in_dim])
            std = torch.ones([self.domain.static_in_dim])

            mean[: self.domain.static_con_dim] = x_static.mean(axis=0)[
                : self.domain.static_con_dim
            ]
            std[: self.domain.static_con_dim] = x_static.std(axis=0)[
                : self.domain.static_con_dim
            ]

            self.static_mean = nn.Parameter(mean, requires_grad=False)
            self.static_std = nn.Parameter(std, requires_grad=False)

        normed_static = (x_static - self.static_mean) / self.static_std
        return normed_static

    def fit_series(self, x_series, mask):

        try:
            self.load_params()
        except:
            combined = torch.tensor([])

            for i, traj in enumerate(x_series):
                no_pad = traj[: mask[i].sum().int(), :]
                combined = torch.cat((combined, no_pad), axis=0)

            mean = torch.zeros([self.domain.series_in_dim])

            std = torch.ones([self.domain.series_in_dim])

            mean[: self.domain.con_out_dim] = combined.mean(axis=0)[
                : self.domain.con_out_dim
            ]
            std[: self.domain.con_out_dim] = combined.std(axis=0)[
                : self.domain.con_out_dim
            ]

            self.series_mean = nn.Parameter(mean, requires_grad=False)
            self.series_std = nn.Parameter(std, requires_grad=False)

        normed = (x_series - self.series_mean) / self.series_std
        N = x_series.shape[0]
        T = x_series.shape[1]
        normed = normed * mask.reshape((N, T, 1)).expand(
            (N, T, self.domain.series_in_dim)
        )

        return normed

    def rescale_static(self, normed_static):
        x_static = (normed_static * self.static_std) + self.static_mean
        return x_static

    def rescale_series(self, normed_series, mask):
        x_series = (normed_series * self.series_std) + self.series_mean
        N = x_series.shape[0]
        T = x_series.shape[1]
        x_series = x_series * mask.reshape((N, T, 1)).expand(
            (N, T, self.domain.series_in_dim)
        )
        return x_series

    def save_params(self):
        path = resource_filename(
            "medkit", f"domains/scalers/{self.domain.base_name}_{self.name}.pth"
        )
        torch.save(self.state_dict(), path)

    def load_params(self):
        path = resource_filename(
            "medkit", f"domains/scalers/{self.domain.base_name}_{self.name}.pth"
        )
        self.load_state_dict(torch.load(path))
        return
