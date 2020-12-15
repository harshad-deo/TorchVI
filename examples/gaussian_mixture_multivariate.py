import numpy as np
from tqdm import tqdm
import torch
from torch import distributions, nn, optim

from torchvi.vdistributions import Normal, HalfNormal
from torchvi.vtensor import Cholesky, Simplex, CholeskyLKJ


class Model(nn.Module):
    def __init__(self, n_components: int, ndim: int):
        super().__init__()
        self.loc = Normal(size=[n_components, ndim], loc=0.0, scale=2.0, name='loc')
        self.scale = HalfNormal(size=[n_components, ndim], scale=1.0, name='scale')
        self.corr = CholeskyLKJ(size=[n_components, ndim], name='corr')
        self.phi = Simplex(size=n_components, name='phi').log()

    def forward(self, xs):
        loc, loc_constraint = self.loc(None)
        scale, scale_constraint = self.scale(None)
        corr, corr_constraint = self.corr(None)
        phi, phi_constraint = self.phi(None)

        tril = torch.matmul(torch.diag_embed(scale), corr)

        dist = distributions.MultivariateNormal(loc=loc, scale_tril=tril)
        lp = dist.log_prob(xs.unsqueeze(1))
        lp = lp + phi
        lp = torch.logsumexp(lp, dim=1).sum()

        constraint = loc_constraint + scale_constraint + phi_constraint + corr_constraint
        return constraint.add_tensor(lp)

    def sample(self, size):
        loc_samples = self.loc.sample(None, size).cpu()
        scale_samples = self.scale.sample(None, size).cpu()
        corr_samples = self.corr.sample(None, size).cpu()
        phi_samples = self.phi.sample(None, size).cpu()

        return {'loc': loc_samples, 'scale': scale_samples, 'phi': phi_samples, 'corr': corr_samples}


def fit(n_components, ndim, xs, num_epochs, num_samples):
    model = Model(n_components, ndim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = model.to(device)
    xs = xs.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    losses = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = -model(xs)
        losses[i] = loss.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)
    samples = model.sample(num_samples)

    return model.cpu(), losses, samples


if __name__ == "__main__":
    import logging
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed
    fix_seed(42)
    logging.basicConfig(format='%(asctime)s - [%(name)25s]:[%(lineno)4d]:[%(levelname)5s] - %(message)s',
                        level=logging.INFO)

    mu_x0 = 2.0
    mu_y0 = 4.0
    mu_x1 = -2.0
    mu_y1 = 2.0

    sd_x0 = 0.4
    sd_y0 = 0.5
    sd_x1 = 0.7
    sd_y1 = 0.5

    rho_0 = torch.tensor([[1.0, 0.0], [0.3, 1.0]])
    rho_1 = torch.tensor([[1.0, 0.0], [-0.5, 1.0]])

    rho_0_chol = torch.cholesky(rho_0, upper=False)
    rho_1_chol = torch.cholesky(rho_1, upper=False)

    sd_0 = torch.matmul(torch.diag(torch.tensor([sd_x0, sd_y0])), rho_0_chol)
    sd_1 = torch.matmul(torch.diag(torch.tensor([sd_x1, sd_y1])), rho_1_chol)

    locs = torch.tensor([[mu_x0, mu_y0], [mu_x1, mu_y1]])
    scale_tril = torch.cat([sd_0.unsqueeze(0), sd_1.unsqueeze(0)])

    dist = distributions.MultivariateNormal(loc=locs, scale_tril=scale_tril)
    num_samples = 100
    xs = dist.sample([num_samples])
    xs = xs.view(-1, 2)

    plt.scatter(xs[:, 0], xs[:, 1])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    plt.clf()
    plt.close()

    num_epochs = 10000
    model, losses, samples = fit(2, 2, xs, num_epochs, num_samples * 10)

    print(model)

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Log loss')
    plt.show()
    plt.clf()
    plt.close()

    samples_loc = samples['loc'].numpy()
    samples_scale = samples['scale'].numpy()
    samples_corr = samples['corr'].numpy()

    fig, axs = plt.subplots(2, 2)

    ax = axs[0][0]
    ax.hist(samples_loc[:, 0, 0])
    ax.set_xlabel('$\mu_{x0}$')
    ax.set_ylabel('$p(\mu_{x0})$')

    ax = axs[0][1]
    ax.hist(samples_loc[:, 0, 1])
    ax.set_xlabel('$\mu_{y0}$')
    ax.set_ylabel('$p(\mu_{y0})$')

    ax = axs[1][0]
    ax.hist(samples_loc[:, 1, 0])
    ax.set_xlabel('$\mu_{x1}$')
    ax.set_ylabel('$p(\mu_{x1})$')

    ax = axs[1][1]
    ax.hist(samples_loc[:, 1, 1])
    ax.set_xlabel('$\mu_{y1}$')
    ax.set_ylabel('$p(\mu_{y1})$')
    plt.show()
    plt.clf()
    plt.close()

    fig, axs = plt.subplots(2, 2)

    ax = axs[0][0]
    ax.hist(samples_scale[:, 0, 0])
    ax.set_xlabel('$\sigma_{x0}$')
    ax.set_ylabel('$p(\sigma_{x0})$')

    ax = axs[0][1]
    ax.hist(samples_scale[:, 0, 1])
    ax.set_xlabel('$\sigma_{y0}$')
    ax.set_ylabel('$p(\sigma_{y0})$')

    ax = axs[1][0]
    ax.hist(samples_scale[:, 1, 0])
    ax.set_xlabel('$\sigma_{x1}$')
    ax.set_ylabel('$p(\sigma_{x1})$')

    ax = axs[1][1]
    ax.hist(samples_scale[:, 1, 1])
    ax.set_xlabel('$\sigma_{y1}$')
    ax.set_ylabel('$p(\sigma_{y1})$')
    plt.show()
    plt.clf()
    plt.close()

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    ax.hist(samples_corr[:, 0, 1, 0])
    ax.set_xlabel('$\\rho_{x0}$')
    ax.set_ylabel('$p(\\rho_{x0})$')

    ax = axs[1]
    ax.hist(samples_corr[:, 1, 1, 0])
    ax.set_xlabel('$\\rho_{y0}$')
    ax.set_ylabel('$p(\\rho_{y0})$')

    plt.show()

    print('ho gaya')
