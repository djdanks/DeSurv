import numpy as np
import torch
import torch.nn as nn

from torch.nn.functional import softplus, softmax


class FCNet(nn.Module):
    """Standard Neural Network"""
    def __init__(self, input_dim, hidden_dim, output_dim,
                 nonlinearity=nn.ReLU, output_act=nn.Identity,
                 device="cpu"):
        super().__init__()

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"FCNet: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"FCNet: {device} specified, {self.device} used")

        self.mapping = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity(),
            nn.Linear(hidden_dim, output_dim),
            output_act()
        )

    def forward(self,x):
        return self.mapping(x)

class CondODENet(nn.Module):
    def __init__(self, cov_dim, hidden_dim, output_dim,
                 nonlinearity=nn.ReLU,
                 device="cpu", n=15):
        super().__init__()

        self.output_dim = output_dim

        if device == "gpu":
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"CondODENet: {device} specified, {self.device} used")
        else:
            self.device = torch.device("cpu")
            print(f"CondODENet: {device} specified, {self.device} used")

        self.dudt = nn.Sequential(
            nn.Linear(cov_dim+1, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, hidden_dim),
            nonlinearity(),

            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()
        )

        self.n = n
        u_n, w_n = np.polynomial.legendre.leggauss(n)
        self.u_n = nn.Parameter(torch.tensor(u_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)
        self.w_n = nn.Parameter(torch.tensor(w_n,device=self.device,dtype=torch.float32)[None,:],requires_grad=False)

    def mapping(self, x_):
        t = x_[:,0][:,None].to(self.device)
        x = x_[:,1:].to(self.device)
        tau = torch.matmul(t/2, 1+self.u_n) # N x n
        tau_ = torch.flatten(tau)[:,None] # Nn x 1. Think of as N n-dim vectors stacked on top of each other
        reppedx = torch.repeat_interleave(x, torch.tensor([self.n]*t.shape[0], dtype=torch.long, device=self.device), dim=0)
        taux = torch.cat((tau_, reppedx),1) # Nn x (d+1)
        f_n = self.dudt(taux).reshape((*tau.shape, self.output_dim)) # N x n x d_out
        pred = t/2 * ((self.w_n[:,:,None] * f_n).sum(dim=1))
        return pred

    def forward(self, x_):
        return torch.tanh(self.mapping(x_))

class ODESurvSingle(nn.Module):
    def __init__(self, lr, cov_dim, hidden_dim,
                 nonlinearity=nn.ReLU,
                 device="cpu", n=15):
        super().__init__()

        self.net = CondODENet(cov_dim, hidden_dim, 1, nonlinearity, device, n)
        self.net = self.net.to(self.net.device)

        self.lr = lr
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

    def predict(self, x, t):
        t = t[:,None]
        N = t.shape[0]

        z = torch.cat((t,x),1)

        return self.net.forward(z).squeeze()

    def forward(self, x, t, k):
        t = t[:,None]
        N = t.size()[0]

        cens_ids = torch.nonzero(torch.eq(k,0))[:,0]
        ncens = cens_ids.size()[0]
        uncens_ids = torch.nonzero(torch.eq(k,1))[:,0]

        z = torch.cat((t,x),1)

        eps = 1e-8

        censterm = 0
        if torch.numel(cens_ids) != 0:
            cdf_cens = self.net.forward(z[cens_ids,:]).squeeze()
            s_cens = 1 - cdf_cens
            censterm = torch.log(s_cens + eps).sum()

        uncensterm = 0
        if torch.numel(uncens_ids) != 0:
            cdf_uncens = self.net.forward(z[uncens_ids,:]).squeeze()
            dudt_uncens = self.net.dudt(z[uncens_ids,:]).squeeze()
            uncensterm = (torch.log(1 - cdf_uncens**2 + eps) + torch.log(dudt_uncens + eps)).sum()

        return -(censterm + uncensterm)

    def optimize(self, data_loader, n_epochs, logging_freq=10, data_loader_val=None,
                 max_wait=20):
        batch_size = data_loader.batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (x, t, k) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t,:]
                t_ = t[argsort_t]
                k_ = k[argsort_t]

                self.optimizer.zero_grad()
                loss = self.forward(x_,t_,k_)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                if data_loader_val is not None:
                    val_loss = 0
                    for batch_idx, (x, t, k) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t,:]
                        t_ = t[argsort_t]
                        k_ = k[argsort_t]

                        loss = self.forward(x_,t_,k_)
                        val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        print(f"best_epoch: {epoch}")
                        torch.save(self.state_dict(), "low")
                    else:
                        wait += 1

                    if wait > max_wait:
                        state_dict = torch.load("low")
                        self.load_state_dict(state_dict)
                        return

                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_loss:11.2f}")
        if data_loader_val is not None:
            state_dict = torch.load("low")
            self.load_state_dict(state_dict)

class ODESurvMultiple(nn.Module):
    def __init__(self, lr, cov_dim, hidden_dim, num_risks,
                 nonlinearity=nn.ReLU, device="cpu", n=15):
        super().__init__()

        self.K = num_risks

        self.odenet = CondODENet(cov_dim, hidden_dim, num_risks, nonlinearity, device, n)
        self.odenet = self.odenet.to(self.odenet.device)

        self.pinet = FCNet(cov_dim, hidden_dim, self.K, device=device, output_act=nn.Softmax)
        self.pinet = self.pinet.to(self.pinet.device)

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def get_pi(self, x):
        return self.pinet(x)

    def predict(self, x, t):
        t = t[:,None]
        N = t.size()[0]

        z = torch.cat((t,x),1)

        pi = self.get_pi(x)

        preds = pi * self.odenet.forward(z)

        return preds, pi

    def forward(self, x, t, k):
        t = t[:,None]

        eps = 1e-8

        censterm = 0
        cens_ids = torch.nonzero(torch.eq(k,0))[:,0]
        if torch.numel(cens_ids) != 0:
            cif_cens = self.predict(x[cens_ids,:], t[cens_ids,0])[0]
            cdf_cens = cif_cens.sum(dim=1)
            censterm = torch.log(1 - cdf_cens + eps).sum()

        uncensterm = 0
        for i in range(self.K):
            ids = torch.nonzero(torch.eq(k,i+1))[:,0]
            if torch.numel(ids) != 0:
                tanhu = self.odenet.forward(torch.cat((t[ids,:], x[ids,:]),1))[:,i]
                pi = self.predict(x[ids,:],t[ids,0])[1][:,i]
                dudt = self.odenet.dudt(torch.cat((t[ids,:], x[ids,:]),1))[:,i]
                l = (torch.log(1 - tanhu**2 + eps) + torch.log(dudt + eps) + torch.log(pi + eps)).sum()
                uncensterm = uncensterm + l
        return -(censterm + uncensterm)

    def optimize(self, data_loader, n_epochs, logging_freq=10, data_loader_val=None,
                 max_wait=20):
        batch_size = data_loader.batch_size

        if data_loader_val is not None:
            best_val_loss = np.inf
            wait = 0

        for epoch in range(n_epochs):

            train_loss = 0.0

            for batch_idx, (x, t, k) in enumerate(data_loader):
                argsort_t = torch.argsort(t)
                x_ = x[argsort_t,:].to(self.odenet.device)
                t_ = t[argsort_t].to(self.odenet.device)
                k_ = k[argsort_t].to(self.odenet.device)

                self.optimizer.zero_grad()
                loss = self.forward(x_,t_,k_)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            if epoch % logging_freq == 0:
                print(f"\tEpoch: {epoch:2}. Total loss: {train_loss:11.2f}")
                if data_loader_val is not None:
                    val_loss = 0
                    for batch_idx, (x, t, k) in enumerate(data_loader_val):
                        argsort_t = torch.argsort(t)
                        x_ = x[argsort_t,:].to(self.odenet.device)
                        t_ = t[argsort_t].to(self.odenet.device)
                        k_ = k[argsort_t].to(self.odenet.device)

                        loss = self.forward(x_,t_,k_)
                        val_loss += loss.item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        wait = 0
                        print(f"best_epoch: {epoch}")
                        torch.save(self.state_dict(), "low_")
                    else:
                        wait += 1

                    if wait > max_wait:
                        state_dict = torch.load("low_")
                        self.load_state_dict(state_dict)
                        return

                    print(f"\tEpoch: {epoch:2}. Total val loss: {val_loss:11.2f}")
        if data_loader_val is not None:
            print("loading low_")
            state_dict = torch.load("low_")
            self.load_state_dict(state_dict)
            return
