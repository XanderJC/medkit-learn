from .__head__ import *
from .base_domain import BaseDomain

class ICUDomain(BaseDomain):
    def __init__(self):
        self.name          = 'ICU'
        self.static_in_dim = 2 
        self.series_in_dim = 27

        self.out_dim       = 27
        self.bin_out_dim   = 2
        self.con_out_dim   = 25

        self.y_dim         = 2

        RNN_config = {'hidden_dim':128,'lr':1e-4,'hidden_layers':3,'adam_betas':(0.9,0.9)}
        self.env_config_dict = {'RNN':RNN_config}
        self.pol_config_dict = {'RNN':None}
        return


    def get_env_config(self,name):
        
        env_config = self.env_config_dict[name]
        self.env_config = env_config
        return env_config

    def get_pol_config(self,name):

        pol_config = self.pol_config_dict[name]
        self.pol_config = pol_config
        return pol_config


    def details(self):
        '''
        Print/return relevant metadata 
        '''
        return