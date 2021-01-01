from .__head__ import *
from .base_env import BaseEnv
from medkit.initialisers.VAE import VAEInit

class RNN_env(nn.Module):
    def __init__(self, domain):
        super(RNN_env, self).__init__()
        self.name = 'RNN'
        self.hidden_size = domain.env_config['hidden_dim']
        self.num_layers = domain.env_config['hidden_layers']
        self.input_size = domain.series_in_dim + 1
        self.lstm = opacus.layers.DPLSTM(self.input_size, self.hidden_size, batch_first=True)
        self.fc_bin = nn.Linear(self.hidden_size, domain.bin_out_dim)
        self.fc_cont = nn.Linear(self.hidden_size, 2 * domain.con_out_dim)
        self.bin_out_dim = domain.bin_out_dim
        self.con_out_dim = domain.con_out_dim
        self.domain = domain
        self.hyper = domain.env_config
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        cont_params = self.fc_cont(out[:, :, :])
        bin_params = torch.sigmoid(self.fc_bin(out[:, :, :]))
        return cont_params,bin_params 

    def loss(self,batch):
        x_static,x_series,mask,y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]
        mask = mask[:,1:].reshape((batch_size,seq_length-1,1))
        concat = torch.cat((x_series,y_series.reshape((batch_size,seq_length,1))),2)
        inputs = concat[:,:-1,:]
        outputs = x_series[:,1:,:]

        cont_pars,bin_pars = self.forward(inputs)

        cont_dist = torch.distributions.normal.Normal(cont_pars[:,:,:self.con_out_dim],
                F.softplus(cont_pars[:,:,self.con_out_dim:]))
        cont_log_l = cont_dist.log_prob(outputs[:,:,:self.con_out_dim])

        bin_dist = torch.distributions.bernoulli.Bernoulli(bin_pars)
        bin_log_l = bin_dist.log_prob(outputs[:,:,self.con_out_dim:])

        log_l = (cont_log_l * mask.expand((-1,-1,self.con_out_dim))).sum() + \
                (bin_log_l * mask.expand((-1,-1,self.bin_out_dim))).sum()

        return -log_l / mask.sum()

    def train(self,dataset,batch_size=128):
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        optimizer = torch.optim.Adam(self.parameters(),lr=self.hyper['lr'],betas= self.hyper['adam_betas'])

        sample_size = len(dataset)
        privacy_engine = PrivacyEngine(
            self,
            batch_size,
            sample_size,
            alphas=[10, 100],
            noise_multiplier=0.1,
            max_grad_norm=1.0,
            secure_rng = True
            )
        privacy_engine.attach(optimizer)
        total_step = len(data_loader)
        for epoch in range(self.hyper['epochs']):
            running_loss = 0
            start = time.time()
            for i,batch in enumerate(data_loader):
                
                optimizer.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                optimizer.step()

                running_loss += loss
            end = time.time()
            average_loss = round((running_loss.detach().numpy()/(i+1)),5)
            print(f'Epoch {epoch+1} average loss: {average_loss} ({round(end-start,2)} seconds)')

        return

    def save_model(self):
        path = resource_filename("environments",f"saved_models/{self.domain.name}_{self.name}.pth")
        torch.save(self.state_dict(), path)

class RNNEnv(BaseEnv):
    def __init__(self,domain,load=True):
        
        self.name = 'RNN'
        
        self.domain = domain
        self.domain.get_env_config(self.name)

        self.model = RNN_env(domain)
        if load:
            self.load_pretrained()

        self.initialiser = VAEInit(domain)


    def step(self,action):

        action = action.reshape((1,1,1))
        x = torch.cat((self.prev_obs,action),2)

        out, (self.hn,self.cn) = self.model.lstm(x, (self.hn, self.cn))

        cont_pars = self.model.fc_cont(out[:, -1, :])
        bin_pars = torch.sigmoid(self.model.fc_bin(out[:, -1, :]))

        cont_dist = torch.distributions.normal.Normal(cont_pars[:,:self.domain.con_out_dim],
                F.softplus(cont_pars[:,self.domain.con_out_dim:]))
        cont_sample = cont_dist.sample()

        bin_dist = torch.distributions.bernoulli.Bernoulli(bin_pars)
        bin_sample = bin_dist.sample()

        observation = torch.cat((cont_sample,bin_sample),1)
        reward = None
        info = None
        done = torch.distributions.bernoulli.Bernoulli(0.1).sample().bool()

        self.prev_obs = observation.reshape((1,1,self.domain.series_in_dim))

        return observation.reshape((self.domain.series_in_dim)),reward,info,done

    def reset(self):

        # Initialise LSTM hidden states
        self.hn = torch.zeros(self.model.num_layers, 1, self.model.hidden_size)
        self.cn = torch.zeros(self.model.num_layers, 1, self.model.hidden_size)

        init_obs,static_obs = self.initialiser.sample()
        self.prev_obs = init_obs

        return static_obs.reshape((self.domain.static_in_dim)),init_obs.reshape((self.domain.series_in_dim))

    def render(self):

        obs = list(self.prev_obs.reshape((self.domain.series_in_dim)))

        for name,value in zip(self.domain.series_names,obs):
            print(f'{name}: {value}')
