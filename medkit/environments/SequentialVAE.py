from .__head__ import *
from .base_env import BaseEnv
from medkit.initialisers.VAE import VAEInit

class Encoder(nn.Module):
    def __init__(self,domain):
        super(Encoder, self).__init__()
        self.hyper = domain.env_config
        self.hidden_size = self.hyper['ae_hidden_dim']
        self.num_layers = self.hyper['ae_hidden_layers']

        self.latent_size = self.hyper['latent_size']

        self.in_dim = domain.series_in_dim

        self.linear1 = nn.Linear(self.in_dim,self.hidden_size)

        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) \
                                        for _ in range(self.num_layers)])

        self.mean_head = nn.Linear(self.hidden_size,self.latent_size)
        self.lstd_head = nn.Linear(self.hidden_size,self.latent_size)

    def forward(self,x):
        x = F.elu(self.linear1(x))
        for linear in self.linears:
            x = F.elu(linear(x))

        means = self.mean_head(x)
        lstds = self.lstd_head(x)

        return means,lstds


class Decoder(nn.Module):
    def __init__(self,domain):
        super(Decoder, self).__init__()

        self.hyper = domain.env_config
        self.out_dim = domain.static_in_dim + domain.series_in_dim
        self.hidden_size = self.hyper['ae_hidden_dim']
        self.num_layers = self.hyper['ae_hidden_layers']

        self.latent_size = self.hyper['latent_size']

        self.linear1 = nn.Linear(self.latent_size,self.hidden_size)

        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) \
                                        for _ in range(self.num_layers)])

        self.series_cont_mean = nn.Linear(self.hidden_size,domain.con_out_dim)
        self.series_cont_lstd = nn.Linear(self.hidden_size,domain.con_out_dim)
        self.series_bin = nn.Linear(self.hidden_size,domain.bin_out_dim)

    def forward(self,x):
        x = F.elu(self.linear1(x))
        for linear in self.linears:
            x = F.elu(linear(x))

        mean = self.series_cont_mean(x)
        lstd = self.series_cont_lstd(x)
        prob = torch.sigmoid(self.series_bin(x))

        return mean,lstd,prob


class SVAE_env(nn.Module):
    def __init__(self, domain):
        super(SVAE_env, self).__init__()
        self.name = 'SVAE'

        self.hyper = domain.env_config
        self.ae_hidden_size = self.hyper['ae_hidden_dim']
        self.ae_num_layers = self.hyper['ae_hidden_layers']

        self.latent_size = self.hyper['latent_size']

        self.encoder = Encoder(domain)
        self.decoder = Decoder(domain)

        self.t_hidden_size = self.hyper['t_hidden_dim']

        self.t_input_size = self.latent_size + domain.y_dim
        self.lstm = opacus.layers.DPLSTM(self.t_input_size, self.t_hidden_size, batch_first=True)
        self.fc_mean = nn.Linear(self.t_hidden_size, self.latent_size)
        self.fc_std = nn.Linear(self.t_hidden_size, self.latent_size)

        #self.h0 = nn.Parameter(torch.randn(1,1,self.t_hidden_size))
        #self.c0 = nn.Parameter(torch.randn(1,1,self.t_hidden_size))

        self.domain = domain
    
    def lstm_forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(1, x.size(0), self.t_hidden_size)#.to(device) 
        c0 = torch.zeros(1, x.size(0), self.t_hidden_size)#.to(device)
        #h0 = self.h0.expand((1, x.size(0), self.t_hidden_size))
        #c0 = self.c0.expand((1, x.size(0), self.t_hidden_size))
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state for all time steps step
        means = self.fc_mean(out)
        lstds = self.fc_std(out)
        return means,lstds 

    def kl_loss(self,z_mean,z_stddev):
        mean_sq = z_mean * z_mean
        stddev_sq = z_stddev * z_stddev
        return 0.5 * torch.mean((mean_sq + stddev_sq - torch.log(stddev_sq) - 1),2)

    def reconstruction_loss(self,x_series,emission_params,latents):
        
        mean,lstd,prob = emission_params

        cont_dist = torch.distributions.normal.Normal(mean,torch.exp(lstd))
        cont_log_l = cont_dist.log_prob(x_series[:,:,:self.domain.con_out_dim])

        bin_dist = torch.distributions.bernoulli.Bernoulli(prob)
        bin_log_l = bin_dist.log_prob(x_series[:,:,self.domain.con_out_dim:])

        return -(cont_log_l.sum(axis=2) + bin_log_l.sum(axis=2))

    def transition_loss(self,targets, pred_latent_params):

        mean,lstd = pred_latent_params
        dist = torch.distributions.normal.Normal(mean,torch.exp(lstd))
        log_l = dist.log_prob(targets)

        return -log_l.sum(axis=2)

    def loss(self,batch):
        x_static,x_series,mask,y_series = batch
        batch_size = x_series.shape[0]

        seq_length = x_series.shape[1]
        t_mask = mask[:,1:].reshape((batch_size,seq_length-1,1))

        latent_means,latent_lstds = self.encoder(x_series)
        kl_loss = self.kl_loss(latent_means,torch.exp(latent_lstds))

        latent_dist = torch.distributions.normal.Normal(latent_means,torch.exp(latent_lstds)) 
        latents = latent_dist.rsample()

        emission_params = self.decoder(latents)

        r_loss = self.reconstruction_loss(x_series,emission_params,latents)

        y_one_hot = F.one_hot(y_series.long(),self.domain.y_dim)

        concat = torch.cat((latents,y_one_hot),2)
        inputs = concat[:,:-1,:]
        targets = latents[:,1:,:]

        pred_latent_params = self.lstm_forward(inputs)

        t_loss = self.transition_loss(targets,pred_latent_params)

        return kl_loss.masked_select(mask.bool()).mean() + \
                r_loss.masked_select(mask.bool()).mean() + \
                t_loss.masked_select(t_mask.squeeze().bool()).mean()

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
        #privacy_engine.attach(optimizer)
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


class SVAEEnv(BaseEnv):
    def __init__(self,domain,load=True):
        
        self.name = 'SVAE'
        
        self.domain = domain
        self.domain.get_env_config(self.name)

        self.model = SVAE_env(domain)
        if load:
            self.load_pretrained()

        self.initialiser = VAEInit(domain)

        self.terminate = 0.1


    def step(self,action):

        action = action.reshape((1,1))
        action_one_hot = F.one_hot(action,self.domain.y_dim)

        x = torch.cat((self.prev_latent,action_one_hot),2)

        out, (self.hn,self.cn) = self.model.lstm(x, (self.hn, self.cn))

        #latent_means = out[0][:, -1, :]
        #latent_lstds = out[1][:, -1, :]
        latent_means = self.model.fc_mean(out)[:, -1, :]
        latent_lstds = self.model.fc_std(out)[:, -1, :]

        latent_dist = torch.distributions.normal.Normal(latent_means,torch.exp(latent_lstds))
        latents = latent_dist.sample()

        self.prev_latent = latents.unsqueeze(1)
        mean,lstd,prob = self.model.decoder(latents)

        cont_dist = torch.distributions.normal.Normal(mean,torch.exp(lstd))
        cont_sample = cont_dist.sample()

        bin_dist = torch.distributions.bernoulli.Bernoulli(prob)
        bin_sample = bin_dist.sample()

        observation = torch.cat((cont_sample,bin_sample),1)
        reward = None
        info = None
        done = torch.distributions.bernoulli.Bernoulli(self.terminate).sample().bool()

        self.prev_obs = observation.reshape((1,1,self.domain.series_in_dim))

        return observation.reshape((self.domain.series_in_dim)),reward,info,done

    def reset(self):

        # Initialise LSTM hidden states
        self.hn = torch.zeros(1, 1, self.model.t_hidden_size)
        self.cn = torch.zeros(1, 1, self.model.t_hidden_size)

        init_obs,static_obs = self.initialiser.sample()
        self.prev_obs = init_obs

        mean,lstd = self.model.encoder(self.prev_obs)

        latent_dist = torch.distributions.normal.Normal(mean,torch.exp(lstd))
        self.prev_latent = latent_dist.sample()

        return static_obs.reshape((self.domain.static_in_dim)),init_obs.reshape((self.domain.series_in_dim))

    def render(self):

        obs = list(self.prev_obs.reshape((self.domain.series_in_dim)))

        for name,value in zip(self.domain.series_names,obs):
            print(f'{name}: {value}')
