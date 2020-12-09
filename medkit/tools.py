import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pkg_resources import resource_filename
#from medkit.domains.ICU import ICUDomain, icu_dataset

class scaler(nn.Module):
    def __init__(self,domain):
        super(scaler, self).__init__()
        self.name = 'scaler'
        self.static_mean = nn.Parameter(torch.zeros([domain.static_in_dim]),requires_grad=False)
        self.static_std  = nn.Parameter(torch.ones([domain.static_in_dim]),requires_grad=False)
        self.series_mean = nn.Parameter(torch.zeros([domain.series_in_dim]),requires_grad=False)
        self.series_std  = nn.Parameter(torch.ones([domain.series_in_dim]),requires_grad=False)

        self.domain = domain
        return

    def fit_static(self,x_static):

        mean = torch.zeros([self.domain.static_in_dim])
        std = torch.ones([self.domain.static_in_dim])

        mean[:self.domain.static_con_dim] = x_static.mean(axis=0)[:self.domain.static_con_dim]
        std[:self.domain.static_con_dim] = x_static.std(axis=0)[:self.domain.static_con_dim]

        self.static_mean = nn.Parameter(mean,requires_grad=False)
        self.static_std = nn.Parameter(std,requires_grad=False)

        normed_static = (x_static - self.static_mean) / self.static_std
        return normed_static

    def fit_series(self,x_series,mask):
        try:
            self.load_params()
        except:
            combined = torch.tensor([])

            for i,traj in enumerate(x_series):
                no_pad = traj[:mask[i].sum().int(),:]
                combined = torch.cat((combined,no_pad),axis=0)

            mean = torch.zeros([self.domain.series_in_dim])
            std = torch.ones([self.domain.series_in_dim])

            mean[:self.domain.con_out_dim] = combined.mean(axis=0)[:self.domain.con_out_dim]
            std[:self.domain.con_out_dim] = combined.std(axis=0)[:self.domain.con_out_dim]

            self.series_mean = nn.Parameter(mean,requires_grad=False)
            self.series_std  = nn.Parameter(std,requires_grad=False)

        normed = (x_series - self.series_mean) / self.series_std
        N = x_series.shape[0]
        T = x_series.shape[1]
        normed = normed * mask.reshape((N,T,1)).expand((N,T,self.domain.series_in_dim))

        return normed

    def rescale_static(self,normed_static):
        x_static = (normed_static * self.static_std) + self.static_mean
        return x_static

    def rescale_series(self,normed_series,mask):
        x_series = (normed_series * self.series_std) + self.series_mean
        N = x_series.shape[0]
        T = x_series.shape[1]
        x_series = x_series * mask.reshape((N,T,1)).expand((N,T,self.domain.series_in_dim))
        return x_series

    def save_params(self):
        path = resource_filename("domains",f"scalers/{self.domain.name}_{self.name}.pth")
        torch.save(self.state_dict(), path)

    def load_params(self):
        path = resource_filename("domains",f"scalers/{self.domain.name}_{self.name}.pth")
        self.load_state_dict(torch.load(path))
        return

if __name__ == '__main__':

    domain = ICUDomain()
    test = scaler(domain)
    test2 = scaler(domain)
    test2.load_params()
    '''
    data = icu_dataset()
    training_generator = torch.utils.data.DataLoader(data,batch_size=64,shuffle=True)
    list_train = list(training_generator)
    x_static,x_series,mask,y_series = list_train[0]
    '''

    path = resource_filename("data","mimic/mimic.p")
    with open(path, 'rb') as f:
        MIMIC_data = pickle.load(f)

    X_series = MIMIC_data["longitudinal"][:, :, :]

    X_static = MIMIC_data['static']

    x_static = torch.tensor(X_static)
    y_series = torch.tensor(X_series[:,:,-1])
    x_mask = torch.FloatTensor(X_series[:,:,0] != 0)
    x_series = torch.tensor(X_series[:,:,:-1])

    normed = test.fit_series(x_series,x_mask)

    print(x_series[0])
    print(normed.shape)
    print(normed[0])

    unnormed = test2.rescale_series(normed,x_mask)

    print(unnormed[0])

    normed = test.fit_static(x_static)

    unnormed = test2.rescale_static(normed)

    print(x_static)
    print(normed)
    print(unnormed)

    #test.save_params()

    
