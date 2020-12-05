import numpy as np
import torch
from torch import distributions, optim
from tqdm import tqdm


class Model:
    def __init__(self):
        self.mu = torch.randn(1)
        self.omega = torch.randn(1)

        self.mu.requires_grad = True
        self.omega.requires_grad = True

    def __call__(self, x):
        eta = torch.randn((1, ))
        zeta = self.mu + eta * self.omega.exp()
        p = 1 / (1 + (-zeta).exp())

        dist = distributions.Bernoulli(logits=zeta)
        lp = dist.log_prob(x).sum()

        jac = p.log() + (1 - p).log()

        return lp + jac + self.omega

    def parameters(self):
        return [self.mu, self.omega]

    def __repr__(self):
        return f'mu: {self.mu}, omega: {self.omega}'


def fit(xs, num_epochs):
    model = Model()
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
    p_known = 0.65

    xs = torch.bernoulli(torch.ones((num_samples, )), p=p_known)
    num_epochs = 1000

    losses, mus, sds = fit(xs, num_epochs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(losses)
    ax1.set_ylabel('Log Loss')
    ax1.set_xlabel('Epoch')

    zeta = np.random.normal(loc=mus[-1], scale=sds[-1], size=num_samples * 10)
    actual = 1 / (1 + np.exp(-zeta))

    sample_success = torch.sum(xs)
    alpha = 1 + sample_success
    beta = 1 + num_samples - sample_success
    expected = np.random.beta(a=alpha, b=beta, size=num_samples * 10)

    ax2.hist(actual, label='actual', alpha=0.5)
    ax2.hist(expected, label='expected', alpha=0.5)
    ax2.set_ylabel('$p(\\theta$)')
    ax2.set_xlabel('$\\theta$')
    ax2.legend()

    plt.show()

    print('ho gaya')
