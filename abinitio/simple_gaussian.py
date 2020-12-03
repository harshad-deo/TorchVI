import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import distributions, optim
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

MU_KNOWN = -4
SIGMA_KNOWN = 3

class Model:
  def __init__(self):
    self.mu = torch.randn(1)
    self.omega = torch.randn(1)

    self.mu.requires_grad = True
    self.omega.requires_grad = True

  def __call__(self, x):
    eta = torch.randn((1,))
    loc = self.mu + eta * self.omega.exp()
    dist = distributions.Normal(loc=loc, scale=SIGMA_KNOWN)
    lp = dist.log_prob(x).sum()
    return lp + self.omega

  def parameters(self):
    return [self.mu, self.omega]

  def __repr__(self):
    return f'mu: {self.mu}, omega: {self.omega}'


def fit(xs, num_epochs):
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    losses = []
    mus = []
    omegas = []

    for _ in tqdm(range(num_epochs)):
      optimizer.zero_grad()
      loss = -model(xs)
      losses.append(loss.item())
      mus.append(model.mu.item())
      omegas.append(model.omega.item())
      loss.backward()
      optimizer.step()

    losses = np.log(np.array(losses))
    sds = np.exp(np.array(omegas))

    return losses, mus, sds


if __name__ == "__main__":
    num_samples = 100
    xs = torch.normal(MU_KNOWN, SIGMA_KNOWN, size=(num_samples,))
    
    num_epochs = 200
    losses, mus, sds = fit(xs, num_epochs)

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

    plt.show()

    print('ho gaya')