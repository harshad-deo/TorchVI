import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch import nn, distributions, optim

from torchvi.vdistributions import Exponential, Beta
from torchvi.vtensor import LowerUpperBound


class Model(nn.Module):
    def __init__(self, rate: float, num_players: int):
        super().__init__()
        kappa_log = Exponential(size=1, rate=rate, name='kappa_log')
        kappa = kappa_log.exp()
        phi = LowerUpperBound(size=1, lower_bound=0.0, upper_bound=1.0, name='phi')
        alpha = phi * kappa
        beta = (1 - phi) * kappa
        self.thetas = Beta(size=num_players, alpha=alpha, beta=beta, name='theta')

    def forward(self, at_bats, hits):
        theta, theta_constraint = self.thetas(None)
        dist = distributions.Binomial(total_count=at_bats, probs=theta)
        lp = dist.log_prob(hits)
        return theta_constraint.add_tensor(lp.sum())

    def sample(self, size):
        return self.thetas.sample(None, size).cpu().numpy()


def fit(rate, at_bats, hits, num_epochs, num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    num_players = at_bats.shape[0]

    model = Model(rate=rate, num_players=num_players)
    model = model.to(device)
    at_bats = at_bats.to(device)
    hits = hits.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    losses = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = -model(at_bats, hits)
        losses[i] = loss.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)
    samples = model.sample(num_samples)

    return model.cpu(), losses, samples


def load_data():
    raw = pd.read_csv('examples/EfronMorrisBaseball.tsv', sep='\t')
    at_bats = torch.from_numpy(raw['At-Bats'].to_numpy())
    hits = torch.from_numpy(raw['Hits'].to_numpy(dtype=np.float32))

    return at_bats, hits


if __name__ == "__main__":
    import logging
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed
    fix_seed(42)
    logging.basicConfig(format='%(asctime)s - [%(name)25s]:[%(lineno)4d]:[%(levelname)5s] - %(message)s',
                        level=logging.INFO)

    at_bats, hits = load_data()
    num_players = at_bats.shape[0]
    rate = 1.5

    num_epochs = 5000
    num_samples = 1000

    model, losses, samples = fit(rate=rate, at_bats=at_bats, hits=hits, num_epochs=num_epochs, num_samples=num_samples)
    print(model)

    xs = np.arange(num_players)
    xticks = np.arange(start=0, stop=num_players)

    fig, axs = plt.subplots(2, 1)

    ax = axs[0]
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log loss')

    print(hits)

    ax = axs[1]
    ax.boxplot(samples)
    ax.set_xlabel('$\\theta_{i}$')
    ax.set_ylabel('$p(\\theta_{i})$')

    plt.show()

    print('ho gaya')
