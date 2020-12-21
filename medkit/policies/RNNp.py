from .__head__ import *
from .base_policy import BasePol

class RNN_pol(nn.Module):
    def __init__(self, domain):
        super(RNN_pol, self).__init__()
        self.name = 'RNN'
        self.hidden_size = domain.pol_config['hidden_dim']
        self.num_layers = domain.pol_config['hidden_layers']
        self.input_size = domain.series_in_dim
        self.lstm = opacus.layers.DPLSTM(self.input_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, domain.y_dim)
        self.domain = domain
        self.hyper = domain.pol_config
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)#.to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state
        pred = F.softmax(self.fc(out[:, :, :]),2)
        return pred

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
        print((pred_flat))
        return (flat_loss * mask_flat).sum() / batch_size

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
            for i,batch in enumerate(data_loader):
                
                optimizer.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                optimizer.step()
        
                print(loss.item())
        return

    def save_model(self):
        path = resource_filename("policies",f"saved_models/{self.domain.name}_{self.name}.pth")
        torch.save(self.state_dict(), path)


class RNNPol(BasePol):
    def __init__(self,domain,load=True):
        
        self.name = 'RNN'
        
        self.domain = domain
        self.domain.get_pol_config(self.name)

        self.model = RNN_pol(domain)
        if load:
            self.load_pretrained()


    def select_action(self,history):

        prev_obs,prev_acts = history
        pred = self.model.forward(prev_obs)[:,-1]
        action = torch.argmax(pred)

        return action