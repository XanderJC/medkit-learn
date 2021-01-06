from .__head__ import *

class BaseDomain(ABC):
    '''
    Domain objects contain the relevant meta-data to be passed to both environment
    and policy classes. In this way they act as essentially config files.
    '''
    def __init__(self):
        
        # Define key features of dataset
        self.name          = None #string
        self.static_in_dim = None #int
        self.series_in_dim = None #int

        self.static_bin_dim = None #int
        self.static_con_dim = None #int
        self.out_dim       = None #int
        self.bin_out_dim   = None #int number of binary variables
        self.con_out_dim   = None #int number of continuous variables

        self.y_dim         = None #int

        return

    '''
    Domains also contain config dictionaries for the different pre-trained policies
    and envrionmnets - load them here.
    '''
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

class BaseDataset(torch.utils.data.Dataset):
    '''
    Base Dataset class to be passed to torch Dataloader so train functions can be unified.
    Not particularly designed for end-user, but for pre-training models.
    '''
    def __init__(self):
        self.N = None #No. of items in dataset
        self.X_static = None # Static features [N,static_dim]
        self.X_series = None # Series features [N,max_seq_length,series_dim]
        self.X_mask = None # Mask [N,max_seq_length]
        self.y_series = None # Static features [N,max_seq_length]

    def __len__(self):
        'Total number of samples'
        return self.N

    def __getitem__(self, index):
        'Generates one batch of data'
        return self.X_static[index], self.X_series[index], self.X_mask[index], self.y_series[index]


class standard_dataset(BaseDataset):
    '''
    Dataset to be passed to a torch DataLoader
    '''
    def __init__(self,domain,max_seq_length=100,save_scale=False):


        scale = scaler(domain)

        path_head = f'{domain.base_name}/{domain.base_name}_temporal_train_data_eav.csv.gz'
        path = resource_filename("data",path_head)
    
        wards = pd.read_csv(path)
        series_df = pd.pivot_table(wards, index=['id', 'time'], columns='variable', 
                                    values='value').reset_index(level=[0, 1])
        series_df.fillna(method='ffill',inplace=True)

        unique_ids = pd.unique(series_df['id'])

        self.N = len(unique_ids)

        series = torch.zeros((len(unique_ids),max_seq_length,domain.series_in_dim))
        y_series = torch.zeros((len(unique_ids),max_seq_length))

        for i,ids in enumerate(unique_ids):
            patient = series_df[series_df['id'] == ids].sort_values(by=['time'])
            cov = patient[domain.series_names].to_numpy()
            cov[:,-domain.bin_out_dim:] = (cov[:,-domain.bin_out_dim:] > 0).astype(int)
            targets = patient[domain.action_names].to_numpy()
            targets = (targets > 0).astype(int)
            y = targets[:,0] 
            num_targets = targets.shape[1]
            for j in range(num_targets-1):
                y += ((2**(j+1)) * targets[:,j+1])
            seq_length = len(cov)
            cov = torch.tensor(cov)
            y = torch.tensor(y)
            if seq_length > max_seq_length:
                series[i,:,:] = cov[:max_seq_length,:]
                y_series[i,:] = y[:max_seq_length]
            else:
                series[i,:seq_length,:] = cov
                y_series[i,:seq_length] = y

        mask = (series[:,:,0] != 0).float()

        path_head = f'{domain.base_name}/{domain.base_name}_static_train_data.csv.gz'

        path = resource_filename("data",path_head)
        static_df = pd.read_csv(path)

        static = torch.zeros((len(unique_ids),domain.static_in_dim))

        for i,ids in enumerate(unique_ids):
            patient = static_df[static_df['id'] == ids]
            cov = patient[domain.static_names].to_numpy()
            cov = torch.tensor(cov)
            static[i,:] = cov

        normed_series = scale.fit_series(series,mask)
        normed_static = scale.fit_static(static)

        if save_scale:
            scale.save_params()

        self.X_static = normed_static
        self.X_series = normed_series
        self.X_mask   = mask
        self.y_series = y_series