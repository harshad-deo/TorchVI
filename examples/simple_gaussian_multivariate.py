import numpy as np
from tqdm import tqdm
import torch
from torch import distributions, nn, optim

from torchvi.vtensor import Cholesky, Unconstrained


class Model(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.mu = Unconstrained(size, name='mu')
        self.chol = Cholesky(size, name='chol')

    def forward(self, xs):
        mu, mu_contrib = self.mu.forward(None)
        chol, chol_contrib = self.chol.forward(None)
        dist = distributions.MultivariateNormal(loc=mu, scale_tril=chol.double())
        data_lp = dist.log_prob(xs).sum()
        constraint_contrib = mu_contrib + chol_contrib
        return constraint_contrib.add_tensor(data_lp)

    def sample(self, size):
        mu_sample = self.mu.sample(None, size).squeeze()
        chol_sample = self.chol.sample(None, size).squeeze()
        return {'mu': mu_sample.cpu().numpy(), 'chol': chol_sample.cpu().numpy()}


def fit(size, xs, num_epochs, num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(size)
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
    samples = model.sample(num_samples * 10)

    return model.cpu(), losses, samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed
    fix_seed(42)

    loc = torch.tensor([-1, 2], dtype=torch.float64)
    scale_tril = torch.tensor([[1, 0], [-1, 2]], dtype=torch.float64)

    num_samples = 100
    dist = distributions.MultivariateNormal(loc=loc, scale_tril=scale_tril)
    xs = dist.sample([num_samples])

    num_epochs = 20000

    model, losses, samples = fit(2, xs, num_epochs, num_samples)
    print(model)

    fig, axs = plt.subplots(2, 3)

    ax = axs[0][0]
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log Loss')

    ax = axs[0][1]
    ax.hist(samples['mu'][:, 0], density=True)
    ax.set_xlabel('$\mu_0$')
    ax.set_ylabel('$p(\mu_0)$')

    ax = axs[0][2]
    ax.hist(samples['mu'][:, 1], density=True)
    ax.set_xlabel('$\mu_1$')
    ax.set_ylabel('$p(\mu_1)$')

    ax = axs[1][0]
    ax.hist(samples['chol'][:, 0, 0], density=True)
    ax.set_xlabel('$\sigma_{00}$')
    ax.set_ylabel('$p(\sigma_{00})$')

    ax = axs[1][1]
    ax.hist(samples['chol'][:, 1, 0], density=True)
    ax.set_xlabel('$\sigma_{10}$')
    ax.set_ylabel('$p(\sigma_{10})$')

    ax = axs[1][2]
    ax.hist(samples['chol'][:, 1, 1], density=True)
    ax.set_xlabel('$\sigma_{11}$')
    ax.set_ylabel('$p(\sigma_{11})$')

    plt.show()

    print("ho gaya")
