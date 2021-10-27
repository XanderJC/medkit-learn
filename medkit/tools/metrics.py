import opacus
from sklearn.metrics import accuracy_score, roc_auc_score

from .__head__ import *


class discriminator(nn.Module):
    def __init__(self, input_size):
        super(discriminator, self).__init__()

        self.input_size = input_size
        self.hidden = 64
        self.lstm = opacus.layers.DPLSTM(self.input_size, self.hidden, batch_first=True)
        self.num_layers = 0
        self.linears = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.num_layers)]
        )
        self.fc = nn.Linear(self.hidden, 2)
        self.hyper = {"lr": 1e-3, "adam_betas": (0.9, 0.99), "epochs": 250}

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden)
        c0 = torch.zeros(1, x.size(0), self.hidden)

        # Forward propagate LSTM
        out, _ = self.lstm(x.float(), (h0, c0))
        for layer in self.linears:
            out = layer(out)
            out = F.elu(out)
        # Decode the hidden state
        pred = self.fc(out[:, -1, :])
        return F.softmax(pred, 1)

    def loss(self, batch):
        x_series, y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]

        pred = self.forward(x_series)
        print(pred)
        print(y_series)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.mean()

    def train(self, dataset, batch_size=128):
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyper["lr"], betas=self.hyper["adam_betas"]
        )

        sample_size = len(dataset)

        total_step = len(data_loader)
        for epoch in range(self.hyper["epochs"]):
            running_loss = 0
            start = time.time()
            for i, batch in enumerate(data_loader):

                optimizer.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                optimizer.step()
                print(loss)
                running_loss += loss
            end = time.time()
            average_loss = round((running_loss.detach().numpy() / (i + 1)), 5)
            print(
                f"Epoch {epoch+1} average loss: {average_loss} ({round(end-start,2)} seconds)"
            )

        return


class discriminator2(nn.Module):
    def __init__(self, input_size):
        super(discriminator2, self).__init__()

        self.input_size = input_size
        self.hidden = 64
        self.num_layers = 2
        self.in_layer = nn.Linear(self.input_size, self.hidden)
        self.linears = nn.ModuleList(
            [nn.Linear(self.hidden, self.hidden) for _ in range(self.num_layers)]
        )
        self.fc = nn.Linear(self.hidden, 2)
        self.hyper = {"lr": 1e-3, "adam_betas": (0.9, 0.99), "epochs": 250}

    def forward(self, x):

        x_flat = x.reshape((x.shape[0], x.shape[1] * x.shape[2]))

        out = self.in_layer(x_flat)

        for layer in self.linears:
            out = layer(out)
            out = F.elu(out)
        # Decode the hidden state
        pred = self.fc(out)
        return F.softmax(pred, 1)

    def loss(self, batch):
        x_series, y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]

        pred = self.forward(x_series)
        print(pred)
        print(y_series)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.mean()

    def train(self, dataset, batch_size=128):
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyper["lr"], betas=self.hyper["adam_betas"]
        )

        sample_size = len(dataset)

        total_step = len(data_loader)
        for epoch in range(self.hyper["epochs"]):
            running_loss = 0
            start = time.time()
            for i, batch in enumerate(data_loader):

                optimizer.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                optimizer.step()
                print(loss)
                running_loss += loss
            end = time.time()
            average_loss = round((running_loss.detach().numpy() / (i + 1)), 5)
            print(
                f"Epoch {epoch+1} average loss: {average_loss} ({round(end-start,2)} seconds)"
            )

        return


class DiscDataset(torch.utils.data.Dataset):
    def __init__(self, s_data, r_data):
        self.N = None  # No. of items in dataset

        self.x_series = torch.cat((s_data, r_data), 0)

        self.N = len(self.x_series)

        synt = torch.ones((len(s_data), 1))
        real = torch.zeros((len(r_data), 1))
        self.y_series = torch.cat((synt, real), 0)

        print(self.x_series.shape)
        print(self.y_series.shape)

    def __len__(self):
        "Total number of samples"
        return self.N

    def __getitem__(self, index):
        "Generates one batch of data"
        return self.x_series[index], self.y_series[index]


class PredDataset(torch.utils.data.Dataset):
    def __init__(self, x_series, y_series):

        self.x_series = x_series

        self.mask = (x_series[:, :, 0] != 0).float()

        self.y_series = y_series.squeeze()

        self.N = len(x_series)
        print(self.x_series.shape)
        print(self.y_series.shape)

    def __len__(self):
        "Total number of samples"
        return self.N

    def __getitem__(self, index):
        "Generates one batch of data"
        return self.x_series[index], self.mask[index], self.y_series[index]


class predictor(nn.Module):
    def __init__(self, input_size, target_size):
        super(predictor, self).__init__()

        self.hidden_size = 128
        self.num_layers = 2
        self.input_size = input_size
        self.lstm = opacus.layers.DPLSTM(
            self.input_size, self.hidden_size, batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, target_size)
        self.linears = nn.ModuleList(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(self.num_layers)
            ]
        )
        self.hyper = {"lr": 1e-2, "adam_betas": (0.9, 0.99), "epochs": 50}

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        c0 = torch.zeros(1, x.size(0), self.hidden_size)

        # Forward propagate LSTM
        out, _ = self.lstm(x.float(), (h0, c0))

        for layer in self.linears:
            out = layer(out)
            out = F.elu(out)
        # Decode the hidden state
        pred = self.fc(out)
        return F.softmax(pred, 2)

    def loss(self, batch):
        x_series, mask, y_series = batch
        batch_size = x_series.shape[0]
        seq_length = x_series.shape[1]

        pred = self.forward(x_series)
        dist = torch.distributions.categorical.Categorical(probs=pred)
        ll = dist.log_prob(y_series)

        return -ll.masked_select(mask.bool()).mean()

    def train(self, dataset, batch_size=128):
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyper["lr"], betas=self.hyper["adam_betas"]
        )

        sample_size = len(dataset)

        for epoch in range(self.hyper["epochs"]):
            running_loss = 0
            start = time.time()
            for i, batch in enumerate(data_loader):

                optimizer.zero_grad()
                loss = self.loss(batch)
                loss.backward()
                optimizer.step()

                running_loss += loss
            end = time.time()
            average_loss = round((running_loss.detach().numpy() / (i + 1)), 5)
            print(
                f"Epoch {epoch+1} average loss: {average_loss} ({round(end-start,2)} seconds)"
            )

        return


from sklearn.metrics import accuracy_score


def predictive_score(s_x, s_y, r_x, r_y, r_mask, y_dim):

    no, seq_len, dim = s_x.shape

    pred_model = predictor(dim, y_dim)

    train_data = PredDataset(s_x, s_y)

    pred_model.train(train_data, batch_size=train_data.N)

    r_preds = pred_model.forward(r_x)
    num_test, _, _ = r_x.shape

    """
    r_preds = r_preds.masked_select(r_mask.bool()).detach().numpy()
    print(r_preds.shape)
    r_y = r_y.masked_select(r_mask.bool()).detach().numpy()
    print(r_y.shape)
    print(r_preds)
    """

    r_preds = r_preds.detach().numpy().reshape(num_test * seq_len, y_dim)
    r_y = r_y.detach().numpy().reshape(num_test * seq_len)
    r_mask = r_mask.bool().detach().numpy().reshape(num_test * seq_len)

    if y_dim == 2:
        r_preds = r_preds[:, 1]
    auc = roc_auc_score(r_y[r_mask], r_preds[r_mask], multi_class="ovr")
    # auc = accuracy_score(r_y[r_mask],np.argmax(r_preds[r_mask],axis=1))

    return auc


def discriminative_score(s_data, r_data):

    no, seq_len, dim = r_data.shape

    split = int(np.ceil(0.5 * no))

    disc_model = discriminator(input_size=dim)

    train_data = DiscDataset(s_data[:split], r_data[:split])

    optimizer = torch.optim.Adam(disc_model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    for i in range(30):
        optimizer.zero_grad()

        synt_pred = disc_model.forward(s_data[:split])
        real_pred = disc_model.forward(r_data[:split])

        loss = (
            -torch.log(synt_pred[:, 1]).mean() - torch.log(real_pred[:, 0]).mean()
        ) / 2
        loss.backward()
        optimizer.step()
        print(loss)

    # Testing time

    synt_pred = disc_model.forward(s_data[split:])
    real_pred = disc_model.forward(r_data[split:])

    synt_corr = synt_pred[:, 1] > 0.5
    real_corr = real_pred[:, 0] > 0.5

    acc = (synt_corr.sum() + real_corr.sum()) / (len(synt_corr) + len(real_corr))

    return np.abs(0.5 - acc)
