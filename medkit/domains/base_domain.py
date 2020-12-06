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
        
        env_config = None
        self.env_config = env_config
        return

    def get_pol_config(self,name):

        pol_config = None
        self.pol_config = pol_config
        return


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