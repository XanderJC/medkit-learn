from .__head__ import *


class ICUDomain(BaseDomain):
    def __init__(self, y_dim=2):
        self.base_name = "icu"
        self.name = self.base_name + f"_{y_dim}"
        self.static_in_dim = 32
        self.series_in_dim = 37

        self.static_bin_dim = 29
        self.static_con_dim = 3
        self.out_dim = 37
        self.bin_out_dim = 0
        self.con_out_dim = 37

        self.terminate = 0.0601

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
            "lr": 1e-5,
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
            "lr": 1e-5,
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
            "weight",
            "height",
            "urgency",
            "gender",
            "surgical",
            "sepsis_at_admission",
            "sepsis_antibiotics",
            "other_antibiotics",
            "sepsis_cultures",
            "General surgery",
            "Internal medicine",
            "Non-operative cardiovascular",
            "Non-operative gastro-intestinal",
            "Non-operative hematological",
            "Non-operative metabolic",
            "Non-operative neurologic",
            "Non-operative genitourinary",
            "Non-operative respiratory",
            "Non-operative musculo-skeletal",
            "Non-operative transplant",
            "Non-operative trauma",
            "Post-operative cardiovascular",
            "Post-operative Gastro-intestinal",
            "Post-operative hematological",
            "Post-operative metabolic",
            "Post-operative neurologic",
            "Post-operative genitourinary",
            "Post-operative respiratory",
            "Post-operative musculo-skeletal",
            "Post-operative transplant",
            "Post-operative trauma",
        ]

        self.series_names = [
            "Diastolic ABP",
            "Average ABP",
            "Systolic ABP",
            "ALAT (blood)",
            "APTT (blood)",
            "ASAT (blood)",
            "Act.HCO3 (blood)",
            "Breathing rate",
            "Alb.Chem (blood)",
            "Alk.Fosf. (blood)",
            "B.E. (blood)",
            "Bilirubine (blood)",
            "CRP (blood)",
            "Ca (alb.corr.) (blood)",
            "Calcium total (blood)",
            "Cl (blood)",
            "Exp. tidal volume",
            "FiO2 %",
            "Phosphate (blood)",
            "Glucose (blood)",
            "Heartrate",
            "Hb (blood)",
            "Potassium (blood)",
            "Creatinine (blood)",
            "Lactate (blood)",
            "Leukocytes (blood)",
            "Magnesium (blood)",
            "Sodium (blood)",
            "O2 concentration",
            "O2 l/min",
            "O2-Saturation (blood)",
            "PO2 (blood)",
            "Saturation (Monitor)",
            "Thrombo's (blood)",
            "Urea (blood)",
            "pCO2 (blood)",
            "pH (blood)",
        ]

        self.action_names = ["antibiotics"]
        if y_dim > 2:
            self.action_names += ["ventilation"]
        if y_dim > 4:
            self.action_names += ["vasopressors"]
        return
