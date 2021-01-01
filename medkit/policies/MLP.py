from .__head__ import *
from .base_policy import BasePol

class mlp_pol(nn.Module):
    def __init__(self, domain):
        super(mlp_pol, self).__init__()
        self.name = 'mlp'
        self.input_size = domain.series_in_dim
        self.hidden_size = domain.pol_config['hidden_dim']
        self.hidden_layers = domain.pol_config['hidden_layers']
        self.in_layer = nn.Linear(self.input_size, self.hidden_size)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) \
                                        for _ in range(self.hidden_layers)])
        self.out_layer = nn.Linear(self.hidden_size, domain.y_dim)                               
        self.domain = domain
        self.hyper = domain.pol_config

    def forward(self,x):

        x = self.in_layer(x)
        for layer in self.linears:
            x = layer(x)
            x = F.elu(x)
        x = self.out_layer(x)

        return x

    def loss(self,batch):
        x_static,x_series,mask,y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]

        pred = self.forward(x_series)
        pred_flat = pred.reshape((pred.shape[0]*pred.shape[1],pred.shape[2]))
        y_flat = y_series.reshape((pred.shape[0]*pred.shape[1])).long()
        mask_flat = mask.reshape((pred.shape[0]*pred.shape[1]))
        nll = nn.CrossEntropyLoss(reduction='none')
        flat_loss = nll(pred_flat,y_flat)

        return (flat_loss * mask_flat).sum() / mask.sum()

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

    def save_model(self):
        path = resource_filename("policies",f"saved_models/{self.domain.name}_{self.name}.pth")
        torch.save(self.state_dict(), path)


class MLPPol(BasePol):
    def __init__(self,domain,load=True):
        
        self.name = 'mlp'
        
        self.domain = domain
        self.domain.get_pol_config(self.name)

        self.model = mlp_pol(domain)
        if load:
            self.load_pretrained()


    def select_action(self,history,stochastic=False,temperature=1.0):

        prev_obs,prev_acts = history
        pred = F.softmax(self.model.forward(prev_obs)[:,-1] /temperature, 1)

        if stochastic:
            act = torch.distributions.categorical.Categorical(probs=pred)
            action = act.sample()
        else:
            action = torch.argmax(pred)

        return action