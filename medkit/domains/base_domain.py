
class BaseDomain():
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
