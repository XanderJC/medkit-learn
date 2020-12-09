from .__head__ import *
from .base_domain import BaseDomain,BaseDataset

class ICUDomain(BaseDomain):
    '''
    This is an example domain based on MIMIC - just a dataset I had lying around for testing
    purposes, the labels are definitely incorrect. When it comes to actually building domains
    it will take some nice cleaning and selection of variables from our various real data sets.
    '''
    def __init__(self):
        self.name          = 'ICU'
        self.static_in_dim = 2 
        self.series_in_dim = 27

        self.static_bin_dim = 0
        self.static_con_dim = 2
        self.out_dim       = 27
        self.bin_out_dim   = 2
        self.con_out_dim   = 25

        self.y_dim         = 2

        RNN_config = {'hidden_dim':128,'lr':1e-2,'hidden_layers':3,'adam_betas':(0.9,0.99),'epochs':50}
        RNN_p_config = {'hidden_dim':128,'lr':1e-4,'hidden_layers':3,'adam_betas':(0.9,0.99),'epochs':50}
        VAE_config = {'latent_size':10,'hidden_units':100,'lr':1e-3,
                'hidden_layers':3,'adam_betas':(0.9,0.9),'epochs':20}
        self.env_config_dict = {'RNN':RNN_config}
        self.pol_config_dict = {'RNN':RNN_p_config}
        self.init_config_dict = {'VAE':VAE_config}

        self.static_names = ['age', 'weight']
        self.series_names  = ['temphigh', 'heartratehigh', 'sysbplow', 'diasbplow', 'meanbplow', 'spo2high',
                      'fio2high', 'respratelow', 'glucoselow', 'bicarbonatehigh',
                      'bicarbonatelow', 'creatininehigh', 'creatininelow', 'hematocrithigh',
                      'hematocritlow', 'hemoglobinhigh', 'hemoglobinlow', 'platelethigh',
                      'plateletlow', 'potassiumlow', 'potassiumhigh', 'bunhigh', 'bunlow',
                      'wbchigh', 'wbclow', 'antibiotics', 'norepinephrine', 'mechanical_ventilator']
        return


    def get_env_config(self,name):
        
        env_config = self.env_config_dict[name]
        self.env_config = env_config
        return env_config

    def get_pol_config(self,name):

        pol_config = self.pol_config_dict[name]
        self.pol_config = pol_config
        return pol_config
    
    def get_init_config(self,name):

        init_config = self.init_config_dict[name]
        self.init_config = init_config
        return init_config


    def details(self):
        '''
        Print/return relevant metadata 
        '''
        return


class icu_dataset(BaseDataset):
    '''
    Dataset to be passed to a torch DataLoader
    '''
    def __init__(self):
        '''
        path = resource_filename("data","mimic/mimic.p")
        with open(path, 'rb') as f:
            MIMIC_data = pickle.load(f)

        XX     = MIMIC_data["longitudinal"][:, :, :]
        scaler_static = StandardScaler()
        X_static = scaler_static.fit_transform(MIMIC_data['static'])

        self.N,T_max,_ = XX.shape
        self.X_mask = torch.zeros(self.N,T_max)
        self.X_mask[:,:] = torch.FloatTensor(XX[:,:,0] != 0)

        scaler = StandardScaler()
        X_unrolled  = XX.reshape((XX.shape[0] * XX.shape[1], XX.shape[2]))
        X_unrolled[:,:25]  = scaler.fit_transform(X_unrolled[:,:25])
        X_long  = X_unrolled.reshape((XX.shape[0], XX.shape[1], XX.shape[2]))  


        scaler = StandardScaler()
        X_unrolled  = XX.reshape((XX.shape[0] * XX.shape[1], XX.shape[2]))
        X_unrolled[:,:25]  = scaler.fit_transform(X_unrolled[:,:25])
        X_long  = X_unrolled.reshape((XX.shape[0], XX.shape[1], XX.shape[2]))

        self.X_static = torch.FloatTensor(X_static)
        self.X_series = torch.FloatTensor(X_long[:,:,:-1])
        self.y_series = torch.FloatTensor(X_long[:,:,-1])
        '''

        path = resource_filename("data","mimic/mimic.p")
        with open(path, 'rb') as f:
            MIMIC_data = pickle.load(f)

        X_series = MIMIC_data["longitudinal"][:, :, :]
        self.N = X_series.shape[0]
        X_static = MIMIC_data['static']

        x_static = torch.FloatTensor(X_static)
        y_series = torch.FloatTensor(X_series[:,:,-1])
        x_mask = torch.FloatTensor(X_series[:,:,0] != 0)
        x_series = torch.FloatTensor(X_series[:,:,:-1])

        domain = ICUDomain()
        scale = scaler(domain)

        normed_series = scale.fit_series(x_series,x_mask)
        normed_static = scale.fit_static(x_static)

        self.X_static = normed_static
        self.X_series = normed_series
        self.X_mask = x_mask
        self.y_series = y_series
