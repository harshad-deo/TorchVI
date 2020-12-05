import numpy as np
import torch
from torch import distributions, optim
from tqdm import tqdm


class Model:
    def __init__(self, sigma_known):
        self.mu = torch.randn(1)
        self.omega = torch.randn(1)
        self.sigma_known = sigma_known

        self.mu.requires_grad = True
        self.omega.requires_grad = True

    def __call__(self, x):
        eta = torch.randn((1, ))
        loc = self.mu + eta * self.omega.exp()
        dist = distributions.Normal(loc=loc, scale=self.sigma_known)
        lp = dist.log_prob(x).sum()
        return lp + self.omega

    def parameters(self):
        return [self.mu, self.omega]

    def __repr__(self):
        return f'mu: {self.mu}, omega: {self.omega}'


def fit(sigma_known, xs, num_epochs):
    model = Model(sigma_known)
    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    losses = np.zeros(num_epochs)
    mus = np.zeros(num_epochs)
    omegas = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = -model(xs)
        losses[i] = loss.item()
        mus[i] = model.mu.item()
        omegas[i] = model.omega.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)
    sds = np.exp(omegas)

    return losses, mus, sds


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed

    fix_seed(42)

    num_samples = 100

    mu_known = -4
    sigma_known = 3

    xs = torch.normal(mu_known, sigma_known, size=(num_samples, ))

    num_epochs = 200
    losses, mus, sds = fit(sigma_known, xs, num_epochs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    ax1.plot(losses)
    ax1.set_ylabel('Log Loss')

    mu_expected = torch.mean(xs)
    ax2.plot(mus, label='calc')
    ax2.hlines(mu_expected, 0, num_epochs, label='expected', linestyle='dashed', color='red')
    ax2.legend()
    ax2.set_ylabel('$\mu$')

    std_expected = torch.std(xs) / np.sqrt(num_samples)
    ax3.plot(sds, label='calc')
    ax3.hlines(std_expected, 0, num_epochs, label='expected', linestyle='dashed', color='red')
    ax3.legend()
    ax3.set_ylabel('$\sigma$')
    ax3.set_xlabel('Epoch')

    plt.show()

    print('ho gaya')
