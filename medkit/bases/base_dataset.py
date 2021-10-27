from .__head__ import *


class BaseDataset(torch.utils.data.Dataset):
    """
    Base Dataset class to be passed to torch Dataloader so train functions can be unified.
    Not particularly designed for end-user, but for pre-training models.
    """

    def __init__(self):
        super(BaseDataset, self).__init__()
        self.N = None  # No. of items in dataset
        self.X_static = None  # Static features [N,static_dim]
        self.X_series = None  # Series features [N,max_seq_length,series_dim]
        self.X_mask = None  # Mask [N,max_seq_length]
        self.y_series = None  # Static features [N,max_seq_length]

    def __len__(self):
        "Total number of samples"
        return self.N

    def __getitem__(self, index):
        "Generates one batch of data"
        return (
            self.X_static[index],
            self.X_series[index],
            self.X_mask[index],
            self.y_series[index],
        )

    def get_whole_batch(self):
        "Returns all data as a single batch"
        return self.X_static, self.X_series, self.X_mask, self.y_series


class standard_dataset(BaseDataset):
    """
    Dataset to be passed to a torch DataLoader
    """

    def __init__(self, domain, max_seq_length=50, test=False, save_scale=False):
        super(standard_dataset, self).__init__()
        scale = scaler(domain)
        fold = "train"
        if test:
            fold = "test"
        path_head = f"data/{domain.base_name}/{domain.base_name}_temporal_{fold}_data_eav.csv.gz"
        path = resource_filename("medkit", path_head)

        data = pd.read_csv(path)
        if domain.base_name == "icu":
            series_df = data
            series_df.fillna(series_df.mean(), inplace=True)
        else:
            series_df = pd.pivot_table(
                data, index=["id", "time"], columns="variable", values="value"
            ).reset_index(level=[0, 1])

            series_df.fillna(method="ffill", inplace=True)

            series_df.fillna(0, inplace=True)

        unique_ids = pd.unique(series_df["id"])

        self.N = len(unique_ids)

        series = torch.zeros((len(unique_ids), max_seq_length, domain.series_in_dim))
        y_series = torch.zeros((len(unique_ids), max_seq_length))

        for i, ids in enumerate(unique_ids):
            patient = series_df[series_df["id"] == ids].sort_values(by=["time"])
            cov = patient[domain.series_names].to_numpy()

            if domain.bin_out_dim > 0:
                cov[:, -domain.bin_out_dim :] = (
                    cov[:, -domain.bin_out_dim :] > 0
                ).astype(int)
            targets = patient[domain.action_names].to_numpy()
            targets = (targets > 0).astype(int)
            y = targets[:, 0]
            num_targets = targets.shape[1]
            for j in range(num_targets - 1):
                y += (2 ** (j + 1)) * targets[:, j + 1]
            seq_length = len(cov)
            cov = torch.tensor(cov)
            y = torch.tensor(y)
            if seq_length > max_seq_length:
                series[i, :, :] = cov[:max_seq_length, :]
                y_series[i, :] = y[:max_seq_length]
            else:
                series[i, :seq_length, :] = cov
                y_series[i, :seq_length] = y

        mask = (series[:, :, 0] != 0).float()

        path_head = (
            f"data/{domain.base_name}/{domain.base_name}_static_{fold}_data.csv.gz"
        )

        path = resource_filename("medkit", path_head)
        static_df = pd.read_csv(path)
        static_df.fillna(static_df.mean(), inplace=True)

        static = torch.zeros((len(unique_ids), domain.static_in_dim))

        for i, ids in enumerate(unique_ids):
            patient = static_df[static_df["id"] == ids]
            cov = patient[domain.static_names].to_numpy()
            cov = torch.tensor(cov)
            static[i, :] = cov

        normed_series = scale.fit_series(series, mask)
        normed_static = scale.fit_static(static)

        if save_scale:
            scale.save_params()

        self.X_static = normed_static
        self.X_series = normed_series
        self.X_mask = mask
        self.y_series = y_series
