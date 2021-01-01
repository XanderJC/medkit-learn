from .__head__ import *
from .base_env import BaseEnv
from medkit.initialisers.VAE import VAEInit
from medkit.tools import reverse_sequence

class emitter(nn.Module):
    def __init__(self,domain):
        super(emitter, self).__init__()
        self.static_in_dim = domain.static_in_dim
        self.series_in_dim = domain.series_in_dim

        self.out_dim = domain.out_dim
        self.bin_out_dim = domain.bin_out_dim
        self.con_out_dim = domain.con_out_dim
        self.emission_dim = domain.env_config['emitter_hidden_dim']

        self.S = domain.env_config['state_space_size']

        self.in_dim = self.S + self.static_in_dim

        self.base = nn.Sequential(
            nn.Linear(self.in_dim, self.emission_dim),
            nn.Tanh(),
            nn.Linear(self.emission_dim, self.emission_dim),
            nn.Tanh()
        )

        self.cont_head = nn.Sequential(
            nn.Linear(self.emission_dim, self.con_out_dim*2)
        )

        self.bin_head = nn.Sequential(
            nn.Linear(self.emission_dim,self.bin_out_dim),
            nn.Sigmoid()
        )

        return

    def forward(self,z_prev,x_static):

        joined_in = torch.cat((x_static,z_prev),1)

        latent = self.base(joined_in)

        cont_params = self.cont_head(latent)
        bin_params = self.bin_head(latent)

        return cont_params,bin_params

class inference_net(nn.Module):
    def __init__(self,domain):
        super(inference_net, self).__init__()
        self.z_dim = domain.env_config['state_space_size']
        self.h_dim = domain.env_config['hidden_dim']
        self.z_to_h = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.Tanh()
        )
        self.h_to_p = nn.Linear(self.h_dim, self.z_dim)

    def forward(self, z_t_1, h_x):

        h_combined = 0.5*(self.z_to_h(z_t_1) + h_x)# combine the rnn hidden state with a transformed version of z_t_1
        
        z_t_p = F.softmax(self.h_to_p(h_combined),1)

        dist = torch.distributions.categorical.Categorical(probs=z_t_p)
        z_t = F.one_hot(dist.sample(),self.z_dim).float()

        return z_t, z_t_p 

class encoder_net(nn.Module):
    def __init__(self,domain):
        super(encoder_net, self).__init__()
        self.input_size = domain.series_in_dim
        self.hidden_size = domain.env_config['encoder_hidden_dim']
        self.lstm = opacus.layers.DPLSTM(self.input_size, self.hidden_size, batch_first=True)
        self.num_layers = 1
        return

    def forward(self, x, mask):

        # Reverse ordering of x 
        rev_x = reverse_sequence(x, mask)

        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Reverse hidden state again
        re_rev_out = reverse_sequence(out, mask)
        return re_rev_out


class state_space_model(nn.Module):
    def __init__(self,domain):
        super(state_space_model, self).__init__()
        self.name = 'statespace'

        self.static_in_dim = domain.static_in_dim
        self.series_in_dim = domain.series_in_dim

        self.out_dim = domain.out_dim
        self.bin_out_dim = domain.bin_out_dim
        self.con_out_dim = domain.con_out_dim

        self.y_dim = domain.y_dim

        self.S = domain.env_config['state_space_size']

        self.inf_net = inference_net(domain)
        self.encoder = encoder_net(domain)
        self.emitter = emitter(domain)

        self.T = nn.Parameter(torch.randn(self.S,self.S,self.y_dim))
        self.z_init_p = nn.Parameter(torch.ones(self.S))

        self.hyper = domain.env_config

        return


    def transition(self,z_prevs,transitions,alpha):

        batch_size = transitions.shape[0]
        distribution = torch.matmul(z_prevs.unsqueeze(2).float(),transitions)
        m_order = len(alpha)
        prob = torch.matmul(alpha.reshape(1,1,m_order).expand(batch_size,1,m_order),distribution.squeeze())

        dist = torch.distributions.categorical.Categorical(probs=prob.squeeze())
        sample = F.one_hot(dist.sample(),self.S).float()

        return sample, prob.squeeze()

    def ll_eval(self,x_series,emission_params):

        cont_pars,bin_pars = emission_params

        cont_dist = torch.distributions.normal.Normal(cont_pars[:,:self.con_out_dim],
                F.softplus(cont_pars[:,self.con_out_dim:]))
        cont_log_l = cont_dist.log_prob(x_series[:,:self.con_out_dim]).sum(axis=1)

        bin_dist = torch.distributions.bernoulli.Bernoulli(bin_pars)
        bin_log_l = bin_dist.log_prob(x_series[:,self.con_out_dim:]).sum(axis=1)

        return -(cont_log_l + bin_log_l)

    @staticmethod
    def kl_div(p,q):
        # p,q both normalised catagorical distirbutions size [batch_size,self.S]

        kl = (p*torch.log(p)) - (p*torch.log(q))

        return kl.sum(axis=1)

    def loss(self,batch,alpha=None):

        x_static,x_series,mask,y_series = batch
        batch_size, T_max, _ = x_series.size()

        nll_losses = torch.zeros((batch_size, T_max)) 
        kl_losses = torch.zeros((batch_size, T_max)) 

        probs=F.softmax(self.z_init_p.unsqueeze(1),0).squeeze()
        dist = torch.distributions.categorical.Categorical(probs=probs)

        z_prev = F.one_hot(dist.sample((batch_size,1)).squeeze(),self.S).float()

        #z_prev = self.z_sample(self.z_init_p)
        #z_prev = F.gumbel_softmax(self.z_init_p.expand((batch_size,self.S)),tau=0.25)

        future_code = self.encoder(x_series, mask)


        alpha = torch.ones((2,1))
        alpha = F.softmax(alpha,dim=0)

        m_order = len(alpha)

        t_norm = F.softmax(self.T,dim=1)

        action_transition = torch.index_select(t_norm,dim=2,index=y_series.reshape(-1,1).squeeze().to(torch.int64))
        action_transition = action_transition.reshape((self.S,self.S,batch_size,-1))
        action_transition = torch.cat((action_transition[:,:,:,:m_order],action_transition),3)

        z_prevs = z_prev.unsqueeze(2).expand(batch_size,self.S,m_order)

        for t in range(T_max):
            z_prior, z_prior_p = self.transition(z_prevs.permute(0,2,1),
                                action_transition[:,:,:,t:t+m_order].permute((2,3,0,1)),alpha)

            z_t, z_t_p = self.inf_net(z_prev,future_code[:,t,:])

            kl = self.kl_div(z_t_p,z_prior_p)

            emission_params = self.emitter(z_t,x_static)

            nll = self.ll_eval(x_series[:,t],emission_params)
            
            kl_losses[:,t] = kl
            nll_losses[:,t] = nll

            new_z_prevs = torch.zeros(z_prevs.shape)
            new_z_prevs[:,:,:-1] = z_prevs[:,:,1:]
            new_z_prevs[:,:,-1] = z_prior
            z_prevs = new_z_prevs


        neg_ll = nll_losses.masked_select(mask.bool()).mean()
        kl_loss = kl_losses.masked_select(mask.bool()).mean()

        return neg_ll + kl_loss
    
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
                print(loss)
                running_loss += loss
            end = time.time()
            average_loss = round((running_loss.detach().numpy()/(i+1)),5)
            print(f'Epoch {epoch+1} average loss: {average_loss} ({round(end-start,2)} seconds)')

        return


class StateSpaceEnv(BaseEnv):
    def __init__(self,load=True):
        self.name = 'statespace'
        
        self.domain = domain
        self.domain.get_env_config(self.name)

        self.model = state_space_model(domain)
        if load:
            self.load_pretrained()

        self.initialiser = VAEInit(domain)


    def step(self,action):

        T_mat = self.mode.T[:,:,action.squeeze().long()]

        z_prime_p = torch.matmul(self.z,self.T)
        dist = torch.distributions.categorical.Categorical(probs=z_prime_p)
        self.z = F.one_hot(dist.sample().squeeze(),self.model.S).float()

        cont_pars,bin_pars = self.model.emitter(self.z,self.static_obs)

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
        
        probs =F.softmax(self.model.z_init_p.unsqueeze(1),0).squeeze()
        dist = torch.distributions.categorical.Categorical(probs=probs)

        self.z = F.one_hot(dist.sample().squeeze(),self.model.S).float()

        init_obs, static_obs = self.initialiser.sample()
        self.static_obs = static_obs

        cont_pars,bin_pars = self.model.emitter(self.z,static_obs)

        cont_dist = torch.distributions.normal.Normal(cont_pars[:,:self.domain.con_out_dim],
                F.softplus(cont_pars[:,self.domain.con_out_dim:]))
        cont_sample = cont_dist.sample()

        bin_dist = torch.distributions.bernoulli.Bernoulli(bin_pars)
        bin_sample = bin_dist.sample()

        init_obs = torch.cat((cont_sample,bin_sample),1)

        return static_obs.reshape((self.domain.static_in_dim)),init_obs.reshape((self.domain.series_in_dim))

    def render(self):

        obs = list(self.prev_obs.reshape((self.domain.series_in_dim)))

        for name,value in zip(self.domain.series_names,obs):
            print(f'{name}: {value}')