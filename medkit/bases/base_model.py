from .__head__ import *


class BaseModel(nn.Module):
    def __init__(self, domain, form):
        super(BaseModel, self).__init__()

        assert form in ["environments", "policies", "initialisers"]
        self.form = form

        self.domain = domain

        if form == "environments":
            self.hyper = domain.env_config
        elif form == "policies":
            self.hyper = domain.pol_config
        elif form == "initialisers":
            self.hyper = domain.init_config

    def forward(self, x):

        return NotImplementedError

    def loss(self, batch):

        return NotImplementedError

    def fit(self, dataset, batch_size=128, validation_set=None, private=False):
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hyper["lr"], betas=self.hyper["adam_betas"]
        )

        sample_size = len(dataset)

        if private:

            privacy_engine = PrivacyEngine(
                self,
                batch_size=batch_size,
                sample_size=sample_size,
                alphas=[1, 10, 100],
                noise_multiplier=0.1,
                max_grad_norm=1.0,
                secure_rng=False,
            )
            privacy_engine.attach(optimizer)

        for epoch in range(self.hyper["epochs"]):
            self.train()
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

            if validation_set is not None:
                self.eval()
                validation_loss = round(float(self.loss(validation_set).detach()), 5)
                print(f"Epoch {epoch+1} validation loss: {validation_loss}")

        return

    def save_model(self):
        path = resource_filename(
            "medkit", f"{self.form}/saved_models/{self.domain.name}_{self.name}.pth"
        )
        torch.save(self.state_dict(), path)
