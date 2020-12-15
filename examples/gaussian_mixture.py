import numpy as np
from tqdm import tqdm
import torch
from torch import distributions, nn, optim

from torchvi.vdistributions import Normal, HalfNormal
from torchvi.vtensor import Simplex


class Model(nn.Module):
    def __init__(self, n_components, loc_mean_prior, loc_scale_prior, scale_prior):
        super().__init__()
        self.loc = Normal(size=n_components, loc=loc_mean_prior, scale=loc_scale_prior, name='mean')
        self.scale = HalfNormal(size=n_components, scale=scale_prior, name='scale')
        self.phi = Simplex(size=n_components, name='phi').log()

    def forward(self, xs):
        loc, loc_constraint = self.loc(None)
        scale, scale_constraint = self.scale(None)
        phi, phi_constraint = self.phi(None)

        dist = distributions.Normal(loc=loc, scale=scale)
        lp = dist.log_prob(xs.unsqueeze(1))
        lp = lp + phi
        lp = torch.logsumexp(lp, dim=1)

        constraint_contrib = loc_constraint + scale_constraint + phi_constraint
        return constraint_contrib.add_tensor(lp.sum())

    def sample(self, size):
        loc_samples = self.loc.sample(None, size).cpu().numpy()
        scale_samples = self.scale.sample(None, size).cpu().numpy()
        phi_samples = self.phi.sample(None, size).exp().cpu().numpy()

        return {'loc': loc_samples, 'scale': scale_samples, 'phi': phi_samples}


def fit(n_components, loc_mean_prior, loc_scale_prior, scale_prior, xs, num_epochs, num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(n_components=n_components,
                  loc_mean_prior=loc_mean_prior,
                  loc_scale_prior=loc_scale_prior,
                  scale_prior=scale_prior)
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

    num_samples = 50
    loc = torch.tensor([-1.0, 2.0])
    scale = torch.tensor([0.9, 0.5])
    dist = distributions.Normal(loc=loc, scale=scale)
    xs = dist.sample([num_samples])
    xs = xs.flatten()

    num_epochs = 5000
    model, losses, samples = fit(2, 0.0, 1.0, 1.0, xs, num_epochs, num_samples * 10)

    fig, axs = plt.subplots(2, 2)

    ax = axs[0][0]
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log Loss')

    ax = axs[0][1]
    ax.boxplot(samples['loc'])
    ax.set_xlabel('loc')

    ax = axs[1][0]
    ax.boxplot(samples['scale'])
    ax.set_xlabel('scale')

    ax = axs[1][1]
    ax.boxplot(samples['phi'])
    ax.set_xlabel('phi')

    plt.show()

    print('ho gaya')
