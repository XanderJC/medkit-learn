from .__head__ import *
from .base_domain import BaseDomain,BaseDataset

class MIMICDomain(BaseDomain):

    def __init__(self,y_dim=2):
        self.base_name      = 'mimic_antibiotics'
        self.name           = self.base_name + f'_{y_dim}'
        self.static_in_dim  = 36
        self.series_in_dim  = 24

        self.static_bin_dim = 30
        self.static_con_dim = 6
        self.out_dim        = 24
        self.bin_out_dim    = 1
        self.con_out_dim    = 23

        valid_y = [2,4,8]
        assert y_dim in valid_y
        self.y_dim         = y_dim

        RNN_config = {'hidden_dim':128,'lr':1e-2,'hidden_layers':3,'adam_betas':(0.9,0.99),'epochs':50}
        RNN_p_config = {'hidden_dim':128,'lr':1e-4,'hidden_layers':3,'adam_betas':(0.9,0.99),'epochs':50}
        VAE_config = {'latent_size':10,'hidden_units':100,'lr':1e-3,
                'hidden_layers':3,'adam_betas':(0.9,0.9),'epochs':20}
        self.env_config_dict = {'RNN':RNN_config}
        self.pol_config_dict = {'RNN':RNN_p_config}
        self.init_config_dict = {'VAE':VAE_config}

        self.static_names = ['height_first', 'height_min', 'height_max', 'weight_first',
            'weight_min', 'weight_max', 'congestive_heart_failure',
            'cardiac_arrhythmias', 'valvular_disease', 'pulmonary_circulation',
            'peripheral_vascular', 'hypertension', 'paralysis',
            'other_neurological', 'chronic_pulmonary', 'diabetes_uncomplicated',
            'diabetes_complicated', 'hypothyroidism', 'renal_failure',
            'liver_disease', 'peptic_ulcer', 'aids', 'lymphoma',
            'metastatic_cancer', 'solid_tumor', 'rheumatoid_arthritis',
            'coagulopathy', 'obesity', 'weight_loss', 'fluid_electrolyte',
            'blood_loss_anemia', 'deficiency_anemias', 'alcohol_abuse',
            'drug_abuse', 'psychoses', 'depression']

        self.series_names = ['bicarbonate', 'bun', 'chloride', 'creatinine', 'diasbp', 'fio2',
            'glucosechart', 'glucoselab', 'heartratehigh', 'hematocrit', 'hemoglobin',
            'inr', 'meanbp', 'platelet', 'potassium', 'pt', 'ptt', 'resprate', 'sodium',
            'spo2', 'sysbp', 'temperature', 'wbc','extubated']

        self.action_names = ['antibiotics']
        if y_dim > 2:
            self.action_names += ['ventilator']
        if y_dim > 4:
            self.action_names += ['oxygentherapy']
        return