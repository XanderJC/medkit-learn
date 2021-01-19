from .__head__ import *
from .base_policy import BasePol

class RNN_pol(nn.Module):
    def __init__(self, domain):
        super(RNN_pol, self).__init__()
        self.name = 'lstm'
        self.hidden_size = domain.pol_config['hidden_dim']
        self.num_layers = domain.pol_config['hidden_layers']
        self.input_size = domain.series_in_dim
        self.lstm = opacus.layers.DPLSTM(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, domain.y_dim)
        self.linears = nn.ModuleList([nn.Linear(self.hidden_size, self.hidden_size) \
                                        for _ in range(self.num_layers)])
        self.domain = domain
        self.hyper = domain.pol_config
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(1, x.size(0), self.hidden_size)#.to(device) 
        c0 = torch.zeros(1, x.size(0), self.hidden_size)#.to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        for layer in self.linears:
            out = layer(out)
            out = F.elu(out)
        # Decode the hidden state
        pred = self.fc(out)
        return pred

    def loss(self,batch):
        x_static,x_series,mask,y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]

        pred = F.softmax(self.forward(x_series),2)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.masked_select(mask.bool()).mean()

    def train(self,dataset,batch_size=128,private=False):
        data_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)
        optimizer = torch.optim.Adam(self.parameters(),lr=self.hyper['lr'],betas= self.hyper['adam_betas'])

        sample_size = len(dataset)
        privacy_engine = PrivacyEngine(self,batch_size, sample_size, alphas=[10, 100], noise_multiplier=0.1,
                                        max_grad_norm=1.0, secure_rng = True)
        if private:
            privacy_engine.attach(optimizer)

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
        path = resource_filename("policies",f"saved_models/{self.domain.name}_{self.name}.pth")
        torch.save(self.state_dict(), path)


class LSTMPol(BasePol):
    def __init__(self,domain,load=True):
        
        self.name = 'lstm'
        
        self.domain = domain
        self.domain.get_pol_config(self.name)

        self.model = RNN_pol(domain)
        if load:
            self.load_pretrained()


    def select_action(self,history,stochastic=False,temperature=1.0):

        prev_obs,prev_acts = history

        pred = F.softmax(self.model.forward(prev_obs)[:,-1] / temperature, 1)

        if stochastic:
            act = torch.distributions.categorical.Categorical(probs=pred)
            action = act.sample()
        else:
            action = torch.argmax(pred)

        return action