from .__head__ import *


class CFDomain(BaseDomain):
    def __init__(self, y_dim=2):
        self.base_name = "cf"
        self.name = self.base_name + f"_{y_dim}"
        self.static_in_dim = 10
        self.series_in_dim = 76

        self.static_bin_dim = 10
        self.static_con_dim = 0
        self.out_dim = 76
        self.bin_out_dim = 66
        self.con_out_dim = 10

        self.terminate = 0.1890

        valid_y = [2, 4, 8]
        assert y_dim in valid_y
        self.y_dim = y_dim

        TForce_config = {
            "hidden_dim": 128,
            "lr": 1e-3,
            "hidden_layers": 3,
            "adam_betas": (0.9, 0.99),
            "epochs": 5,
        }
        SS_config = {
            "state_space_size": 6,
            "encoder_hidden_dim": 64,
            "emitter_hidden_dim": 64,
            "hidden_dim": 64,
            "mix_components": 3,
            "markov_order": 3,
            "lr": 1e-3,
            "hidden_layers": 1,
            "adam_betas": (0.9, 0.99),
            "epochs": 5,
        }
        SVAE_config = {
            "latent_size": 10,
            "ae_hidden_dim": 128,
            "ae_hidden_layers": 1,
            "t_hidden_dim": 128,
            "lr": 1e-4,
            "adam_betas": (0.9, 0.99),
            "epochs": 100,
        }
        CRN_config = {
            "hidden_dim": 128,
            "lr": 1e-2,
            "hidden_layers": 1,
            "adam_betas": (0.9, 0.99),
            "epochs": 50,
        }

        LSTM_config = {
            "hidden_dim": 64,
            "lr": 1e-3,
            "hidden_layers": 1,
            "adam_betas": (0.9, 0.99),
            "epochs": 200,
        }
        MLP_config = {
            "hidden_dim": 128,
            "lr": 1e-2,
            "hidden_layers": 3,
            "adam_betas": (0.9, 0.99),
            "epochs": 50,
        }
        Linear_config = {"lr": 1e-2, "adam_betas": (0.9, 0.99), "epochs": 500}

        VAE_config = {
            "latent_size": 10,
            "hidden_units": 100,
            "lr": 1e-3,
            "hidden_layers": 3,
            "adam_betas": (0.9, 0.9),
            "epochs": 300,
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
            "Gender",
            "Smoking Status",
            "Class I Mutation",
            "Class II Mutation",
            "Class III Mutation",
            "Class IV Mutation",
            "Class V Mutation",
            "Class VI Mutation",
            "DF508 Mutation",
            "G551D Mutation",
        ]

        self.series_names = [
            "BMI",
            "Best FEV1",
            "Best FEV1 Predicted",
            "FEV1",
            "FEV1 Predicted",
            "Height",
            "IV Antibiotic Days Home",
            "IV Antibiotic Days Hosp",
            "Non-IV Hospital Admission",
            "Weight",
            "ABPA",
            "ALCA",
            "Acetylcysteine",
            "Aminoglycoside",
            "Anti-fungals",
            "Arthropathy",
            "Aspergillus",
            "Asthma",
            "Bone fracture",
            "Burkholderia Cepacia",
            "Cancer",
            "Chronic Oral Antibiotic",
            "Cirrhosis",
            "Colistin",
            "Colonic structure",
            "Cortico Inhaled",
            "Cortico Oral",
            "Depression",
            "Diabetes",
            "Diabetes Inter Insulin",
            "Drug Dornase",
            "Ecoli",
            "GI bleeding non-var source",
            "GI bleeding var source",
            "Gall bladder",
            "Gram-Negative",
            "HDI Buprofen",
            "Haemophilus Influenza",
            "Hearing Loss",
            "Hemoptysis",
            "Heterozygous",
            "Homozygous",
            "HyperSaline",
            "Hypertension",
            "HypertonicSaline",
            "Inhaled Broncho BAAC",
            "Inhaled Broncho LAAC",
            "Inhaled Broncho LABA",
            "Inhaled Broncho SAAC",
            "Inhaled Bronchodilators",
            "Intestinal Obstruction",
            "Kidney Stones",
            "Klebsiella Pneumoniae",
            "Lab Liver Enzymes",
            "Leukotriene",
            "Liver Disease",
            "Liver Enzymes",
            "Macrolida Antibiotics",
            "NTM",
            "Noninvasive Ventilation",
            "O2 Cont",
            "O2 Exc",
            "O2 Noct",
            "O2 Prn",
            "Oral Broncho BA",
            "Oral Broncho THEOPH",
            "Oral Hypoglycemic Agents",
            "Osteopenia",
            "Osteoporosis",
            "Pancreatitus",
            "Pseudomonas Aeruginosa",
            "Staphylococcus Aureus",
            "Tobi Solution",
            "Tobramycin",
            "Xanthomonas",
            "iBuprofen",
        ]

        self.action_names = ["Cortico Combo"]
        if y_dim > 2:
            self.action_names += ["Oxygen Therapy"]
        if y_dim > 4:
            self.action_names += ["Dornase Alpha"]
        return
