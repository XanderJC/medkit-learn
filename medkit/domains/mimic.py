from .__head__ import *


class MimicDomain(BaseDomain):
    def __init__(self, y_dim=2):
        self.base_name = "mimic"
        self.name = self.base_name + f"_{y_dim}"
        self.static_in_dim = 36
        self.series_in_dim = 24

        self.static_bin_dim = 30
        self.static_con_dim = 6
        self.out_dim = 24
        self.bin_out_dim = 1
        self.con_out_dim = 23

        self.terminate = 0.1119

        valid_y = [2, 4, 8]
        assert y_dim in valid_y
        self.y_dim = y_dim

        TForce_config = {
            "hidden_dim": 128,
            "lr": 1e-2,
            "hidden_layers": 3,
            "lstm_layers": 2,
            "dropout": 0.2,
            "adam_betas": (0.9, 0.99),
            "epochs": 100,
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
            "epochs": 100,
        }
        SVAE_config = {
            "latent_size": 10,
            "ae_hidden_dim": 128,
            "ae_hidden_layers": 1,
            "t_hidden_dim": 128,
            "t_lstm_layers": 2,
            "lr": 1e-4,
            "adam_betas": (0.9, 0.99),
            "epochs": 50,
        }
        CRN_config = {
            "hidden_dim": 128,
            "lr": 1e-3,
            "hidden_layers": 3,
            "lstm_layers": 2,
            "dropout": 0.2,
            "adam_betas": (0.9, 0.99),
            "epochs": 100,
        }

        LSTM_config = {
            "hidden_dim": 64,
            "lr": 1e-3,
            "hidden_layers": 2,
            "lstm_layers": 1,
            "dropout": 0.2,
            "adam_betas": (0.9, 0.99),
            "epochs": 200,
        }
        MLP_config = {
            "hidden_dim": 128,
            "lr": 1e-3,
            "hidden_layers": 3,
            "adam_betas": (0.9, 0.99),
            "epochs": 100,
        }
        Linear_config = {"lr": 1e-3, "adam_betas": (0.9, 0.99), "epochs": 100}

        VAE_config = {
            "latent_size": 10,
            "hidden_units": 100,
            "lr": 1e-3,
            "hidden_layers": 3,
            "adam_betas": (0.9, 0.9),
            "epochs": 500,
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
            "height_first",
            "height_min",
            "height_max",
            "weight_first",
            "weight_min",
            "weight_max",
            "congestive_heart_failure",
            "cardiac_arrhythmias",
            "valvular_disease",
            "pulmonary_circulation",
            "peripheral_vascular",
            "hypertension",
            "paralysis",
            "other_neurological",
            "chronic_pulmonary",
            "diabetes_uncomplicated",
            "diabetes_complicated",
            "hypothyroidism",
            "renal_failure",
            "liver_disease",
            "peptic_ulcer",
            "aids",
            "lymphoma",
            "metastatic_cancer",
            "solid_tumor",
            "rheumatoid_arthritis",
            "coagulopathy",
            "obesity",
            "weight_loss",
            "fluid_electrolyte",
            "blood_loss_anemia",
            "deficiency_anemias",
            "alcohol_abuse",
            "drug_abuse",
            "psychoses",
            "depression",
        ]

        self.series_names = [
            "bicarbonate",
            "bun",
            "chloride",
            "creatinine",
            "diasbp",
            "fio2",
            "glucosechart",
            "glucoselab",
            "heartratehigh",
            "hematocrit",
            "hemoglobin",
            "inr",
            "meanbp",
            "platelet",
            "potassium",
            "pt",
            "ptt",
            "resprate",
            "sodium",
            "spo2",
            "sysbp",
            "temperature",
            "wbc",
            "extubated",
        ]

        self.action_names = ["antibiotics"]
        if y_dim > 2:
            self.action_names += ["ventilator"]
        if y_dim > 4:
            self.action_names += ["oxygentherapy"]
        return
