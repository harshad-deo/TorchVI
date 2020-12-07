import math
import numpy as np
import torch
from torch import nn, distributions, optim
from tqdm import tqdm

from torchvi import vtensor


class Model(nn.Module):
    def __init__(self, scale_known):
        super().__init__()
        self.mu = vtensor.Unconstrained(1)
        self.scale_known = scale_known

    def forward(self, xs):
        mu, constraint_contrib = self.mu(None)
        dist = distributions.Normal(mu, self.scale_known)
        lp = dist.log_prob(xs).sum()

        return lp + constraint_contrib

    def sample(self, size):
        return torch.squeeze(self.mu.sample(None, size))


def fit(sigma_known, xs, num_epochs, num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(sigma_known)
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
    samples = model.sample(num_samples * 10).cpu().numpy()

    return model.cpu(), losses, samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed
    fix_seed(42)

    num_samples = 100

    mu_known = -4
    sigma_known = 3

    xs = torch.normal(mu_known, sigma_known, size=(num_samples, ))

    num_epochs = 200
    model, losses, actual_samples = fit(sigma_known, xs, num_epochs, num_samples)

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(losses)
    ax1.set_ylabel('Log Loss')
    ax1.set_xlabel('Epoch')

    expected_samples = np.random.normal(loc=mu_known, scale=sigma_known / math.sqrt(num_samples), size=num_samples * 10)

    ax2.hist(actual_samples, label='actual', alpha=0.5)
    ax2.hist(expected_samples, label='expected', alpha=0.5)
    ax2.set_ylabel('$p(\\theta$)')
    ax2.set_xlabel('$\\theta$')
    ax2.legend()

    plt.show()

    print('ho gaya')
