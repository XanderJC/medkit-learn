from .__head__ import *


class WardDomain(BaseDomain):
    def __init__(self, y_dim=2):
        self.base_name = "ward"
        self.name = self.base_name + f"_{y_dim}"
        self.static_in_dim = 49
        self.series_in_dim = 35

        self.static_bin_dim = 48
        self.static_con_dim = 1
        self.out_dim = 35
        self.bin_out_dim = 15
        self.con_out_dim = 20

        self.terminate = 0.0291

        valid_y = [2, 4, 8]
        assert y_dim in valid_y

        self.y_dim = y_dim

        TForce_config = {
            "hidden_dim": 128,
            "lr": 1e-3,
            "hidden_layers": 3,
            "lstm_layers": 2,
            "dropout": 0.2,
            "adam_betas": (0.9, 0.99),
            "epochs": 30,
        }
        SS_config = {
            "state_space_size": 10,
            "encoder_hidden_dim": 64,
            "emitter_hidden_dim": 64,
            "hidden_dim": 64,
            "mix_components": 5,
            "markov_order": 5,
            "lr": 1e-3,
            "hidden_layers": 1,
            "adam_betas": (0.9, 0.99),
            "epochs": 30,
        }
        SVAE_config = {
            "latent_size": 10,
            "ae_hidden_dim": 128,
            "ae_hidden_layers": 1,
            "t_hidden_dim": 128,
            "t_lstm_layers": 2,
            "lr": 1e-3,
            "adam_betas": (0.9, 0.99),
            "epochs": 30,
        }
        CRN_config = {
            "hidden_dim": 128,
            "lr": 1e-3,
            "hidden_layers": 3,
            "lstm_layers": 2,
            "dropout": 0.2,
            "adam_betas": (0.9, 0.99),
            "epochs": 30,
        }

        LSTM_config = {
            "hidden_dim": 64,
            "lr": 1e-3,
            "hidden_layers": 2,
            "lstm_layers": 1,
            "dropout": 0.2,
            "adam_betas": (0.9, 0.99),
            "epochs": 30,
        }
        MLP_config = {
            "hidden_dim": 128,
            "lr": 1e-3,
            "hidden_layers": 3,
            "adam_betas": (0.9, 0.99),
            "epochs": 30,
        }
        Linear_config = {"lr": 1e-3, "adam_betas": (0.9, 0.99), "epochs": 30}

        VAE_config = {
            "latent_size": 10,
            "hidden_units": 100,
            "lr": 1e-4,
            "hidden_layers": 3,
            "adam_betas": (0.9, 0.9),
            "epochs": 200,
        }

        self.env_config_dict = {
            "tforce": TForce_config,
            "statespace": SS_config,
            "SVAE": SVAE_config,
            "CRN": CRN_config,
        }
        self.pol_config_dict = {
            "lstm": LSTM_config,
            "mlp": MLP_config,
            "linear": Linear_config,
        }
        self.init_config_dict = {"VAE": VAE_config}

        self.static_names = [
            "age",
            "floor: MP2 MPU PR",
            "floor: MP2 PERIOPERATIVE AREA",
            "floor: RR 3F",
            "floor: RR 5DR",
            "floor: RR 5EMS",
            "floor: RR 5EOB",
            "floor: RR 5FDU IOF",
            "floor: RR 5PICU",
            "floor: RR 5W",
            "floor: RR 6E",
            "floor: RR 6ICU",
            "floor: RR 6N",
            "floor: RR 6W",
            "floor: RR 7E",
            "floor: RR 7ICU",
            "floor: RR 7N",
            "floor: RR 7W",
            "floor: RR 8E",
            "floor: RR 8N",
            "floor: RR 8W",
            "floor: RR GOU",
            "floor: RR MPU",
            "floor: RR PACU BOARDING",
            "floor: RR PERIOPERATIVE AREA",
            "floor: RR TRU 2WEST",
            "floor: SM 4SW",
            "gender",
            "ethnicity: Hispanic or Latino",
            "ethnicity: Hispanic/Spanish origin Other",
            "ethnicity: Mexican, Mexican American, Chicano/a",
            "ethnicity: Not Hispanic or Latino",
            "ethnicity: Patient Refused",
            "ethnicity: Puerto Rican",
            "ethnicity: Unknown",
            "race: American Indian or Alaska Native",
            "race: Asian",
            "race: Black or African American",
            "race: Multiple Races",
            "race: Native Hawaiian or Other Pacific Islander",
            "race: Other",
            "race: Patient Refused",
            "race: Unknown",
            "race: White or Caucasian",
            "cpt: Allogeneic stem cell transplantation",
            "cpt: Autologous stem cell transplantation",
            "cpt: Chemo",
            "cpt: NA",
            "icu_admission",
        ]

        self.series_names = [
            "Best Motor Response",
            "Best Verbal Response",
            "CHLORIDE",
            "CREATINEINE",
            "DBP",
            "Eye Opening",
            "GLUCLOSE",
            "Glasgow Coma Scale Score",
            "HEMOGLOBIN",
            "PLATELET COUNT",
            "POTASSIUM",
            "Pulse",
            "Respiratory Rate",
            "SBP",
            "SODIUM",
            "SpO2",
            "TOTAL CO2",
            "Temperature",
            "UREA NITROGEN",
            "WHITE BLOOD CELL COUNT",
            "O2 Device: Aerosol mask",
            "O2 Device: Blow-by",
            "O2 Device: Continuous Inhaled Medication",
            "O2 Device: Face tent",
            "O2 Device: Heliox",
            "O2 Device: CPAP",
            "O2 Device: Trach Collar",
            "O2 Device: Non-rebreather mask",
            "O2 Device: None (Room air)",
            "O2 Device: Other (Comment)",
            "O2 Device: Partial rebreather mask",
            "O2 Device: Trach",
            "O2 Device: Bi-PAP",
            "O2 Device: Transtracheal catheter",
            "O2 Device: Venturi mask",
        ]
        self.action_names = ["O2 Device: Nasal cannula"]
        if y_dim > 2:
            self.action_names += ["O2 Device: Simple mask"]
        if y_dim > 4:
            self.action_names += ["O2 Device: High flow nasal cannula"]
        return


class ward_dataset(BaseDataset):
    """
    Dataset to be passed to a torch DataLoader
    """

    def __init__(self, max_seq_length=100):

        domain = WardsDomain()
        scale = scaler(domain)

        path = resource_filename("data", "ward/ward_temporal_train_data_eav.csv.gz")

        wards = pd.read_csv(path)
        series_df = pd.pivot_table(
            wards, index=["id", "time"], columns="variable", values="value"
        ).reset_index(level=[0, 1])
        series_df.fillna(method="ffill", inplace=True)

        unique_ids = pd.unique(series_df["id"])

        self.N = len(unique_ids)

        series = torch.zeros((len(unique_ids), max_seq_length, domain.series_in_dim))
        y_series = torch.zeros((len(unique_ids), max_seq_length))

        for i, ids in enumerate(unique_ids):
            patient = series_df[series_df["id"] == ids].sort_values(by=["time"])
            cov = patient[domain.series_names].to_numpy()
            targets = patient[domain.action_names].to_numpy()
            y = targets[:, 0] + (2 * targets[:, 1])
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

        path = resource_filename("data", "ward/ward_static_train_data.csv.gz")

        static_df = pd.read_csv(path)

        static = torch.zeros((len(unique_ids), domain.static_in_dim))

        for i, ids in enumerate(unique_ids):
            patient = static_df[static_df["id"] == ids]
            cov = patient[domain.static_names].to_numpy()
            cov = torch.tensor(cov)
            static[i, :] = cov

        normed_series = scale.fit_series(series, mask)
        normed_static = scale.fit_static(static)

        self.X_static = normed_static
        self.X_series = normed_series
        self.X_mask = mask
        self.y_series = y_series
