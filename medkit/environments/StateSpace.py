from .__head__ import *
from .base_env import BaseEnv
from medkit.initialisers.VAE import VAEInit


class emitter(nn.Module):
    def __init__(self,domain):
        self.static_in_dim = domain.static_in_dim
        self.series_in_dim = domain.series_in_dim

        self.out_dim = domain.out_dim
        self.bin_out_dim = domain.bin_out_dim
        self.con_out_dim = domain.con_out_dim
        self.emission_dim = domain.emission_dim

        self.S = domain.model_config['state_space_size']

        self.in_dim = self.S + self.static_in_dim

        self.base = nn.Sequential(
            nn.Linear(self.in_dim, self.emission_dim),
            nn.ReLU()
        )

        self.cont_head = nn.Sequential(
            nn.Linear(self.emission_dim, self.cont_out_dim_par)
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

        return

class encoder_net(nn.Module):
    def __init__(self,domain):
        
        self.input_size = 
        self.hidden_size = 
        self.lstm = opacus.layers.DPLSTM(self.input_size, self.hidden_size, batch_first=True)
        return


class state_space_model(nn.Module):
    def __init__(self,domain):

        self.static_in_dim = domain.static_in_dim
        self.series_in_dim = domain.series_in_dim

        self.out_dim = domain.out_dim
        self.bin_out_dim = domain.bin_out_dim
        self.con_out_dim = domain.con_out_dim

        self.y_dim = domain.y_dim

        self.rnn_dim = domain.model_config['rnn_dim']
        self.inf_net = inference_net()

        self.encoder = encoder_net()

        self.num_mixes = domain.model_config['num_mixes']
        self.S = domain.model_config['state_space_size']

        self.emitter = emitter(domain)
        self.T = nn.Parameter(torch.zeros(self.S,self.S,self.y_dim))
        self.z_init_p = nn.Parameter(torch.zeros(self.S))

        return

    def transition(self,z_prev,y_series,alpha):

        t_norm = F.softmax(self.T,dim=1)    
        t_matrix = t_norm.expand((len(alpha),self.S,self.S,self.y_dim))
        alpha_e = alpha.reshape((len(alpha),1,1,1)).expand((-1,self.S,self.S,self.y_dim))

        attentive_transition = (t_matrix * alpha_e).sum(axis=0)
        action_transtion = torch.gather(attentive_transition,dim=2,index=y_series)
        distribution = torch.matmul(z_prev,action_transtion)

        #sample = F.gumbel_softmax(distribution,tau=0.25)
        dist = = torch.distributions.categorical.Categorical(probs=distribution)
        sample = F.one_hot(dist.sample(),self.S)

        return sample, distribution

    def ll_eval(self,x_series,emission_params):

        cont_pars,bin_pars = emission_params

        cont_dist = torch.distributions.normal.Normal(cont_pars[:,:self.con_out_dim],
                F.softplus(cont_pars[:,self.con_out_dim:]))
        cont_log_l = cont_dist.log_prob(outputs[:,:self.con_out_dim])

        bin_dist = torch.distributions.bernoulli.Bernoulli(bin_pars)
        bin_log_l = bin_dist.log_prob(outputs[:,self.con_out_dim:])

        return cont_log_l + bin_log_l

    @staticmethod
    def kl_div(p,q):
        # p,q both normalised catagorical distirbutions size [batch_size,self.S]
        kl = (p*torch.log(p)) - (p*torch.log(q))

        return kl.sum(axis=1)

    def forward(self,x_static,x_series,y_series,mask,alpha):

        batch_size, T_max, _ = x_series.size()

        nll_losses = torch.zeros((batch_size, T_max)) 
        kl_losses = torch.zeros((batch_size, T_max)) 

        dist = = torch.distributions.categorical.Categorical(probs=F.softmax(self.z_init_p))
        z_prev = F.one_hot(dist.sample(batch_size),self.S)

        #z_prev = self.z_sample(self.z_init_p)
        #z_prev = F.gumbel_softmax(self.z_init_p.expand((batch_size,self.S)),tau=0.25)

        future_code = self.encoder(x_series, mask)


        for t in range(T_max):
            z_prior, z_prior_p = self.transition(z_prev,y_series[t],alpha)

            z_t, z_t_p = self.inf_net(z_prev,future_code[:,t,:])

            kl = self.kl_div(z_t_p,z_prior_p)

            emission_params = self.emitter(z_t,x_static)

            likelihood = self.ll_eval(x_series[:,t],emission_params)
            
            kl_losses[:,t] = kl
            nll_losses[:,t] = likelihood

            z_prev = z_prior


        neg_ll = nll_losses.masked_select(mask.bool()).mean()
        kl_loss = kl_losses.masked_select(mask.bool()).mean()

        return neg_ll + kl_loss


class StateSpaceEnv(BaseEnv):
    def __init__(self):
        return