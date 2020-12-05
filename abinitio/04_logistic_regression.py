import math
import numpy as np
import torch
from torch import optim, distributions
from tqdm import tqdm


class Model:
    def __init__(self, prior_mu, prior_sd):
        self.mu_0 = torch.randn(1)
        self.mu_1 = torch.randn(1)
        self.omega_0 = torch.randn(1)
        self.omega_1 = torch.randn(1)

        self.mu_0.requires_grad = True
        self.mu_1.requires_grad = True
        self.omega_0.requires_grad = True
        self.omega_1.requires_grad = True

        self.prior = distributions.Normal(loc=prior_mu, scale=prior_sd)

    def __call__(self, xs, ys):
        eta_0 = torch.randn(1)
        eta_1 = torch.randn(1)

        zeta_0 = self.mu_0 + eta_0 * self.omega_0.exp()
        zeta_1 = self.mu_1 + eta_1 * self.omega_1.exp()

        logit = zeta_0 + xs * zeta_1
        dist = distributions.Bernoulli(logits=logit)

        data_lp = dist.log_prob(ys).sum()
        theta_lp = self.prior.log_prob(zeta_0) + self.prior.log_prob(zeta_1)

        return data_lp + theta_lp + self.omega_0 + self.omega_1

    def parameters(self):
        return [self.mu_0, self.mu_1, self.omega_0, self.omega_1]

    def __repr__(self):
        mu_rep = f'{[self.mu_0, self.mu_1]}'
        omega_rep = f'{[self.omega_0, self.omega_1]}'
        return f'mu: {mu_rep}, omega: {omega_rep}'


def fit(prior_mu, prior_sd, xs, ys, num_epochs):
    model = Model(prior_mu, prior_sd)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    losses = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = -model(xs, ys)
        losses[i] = loss.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)

    return model, losses


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed

    fix_seed(42)

    num_samples = 100
    slope = -0.5
    intercept = 0.5

    xmin = -5
    xmax = 5
    xs = xmin + (xmax - xmin) * torch.rand(num_samples)
    logits = intercept + slope * xs
    y_dist = distributions.Bernoulli(logits=logits)
    ys = y_dist.sample()

    num_epochs = 5000
    model, losses = fit(0, 0.5, xs, ys, num_epochs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    mu_0 = model.mu_0.item()
    mu_1 = model.mu_1.item()

    sd_0 = math.exp(model.omega_0.item())
    sd_1 = math.exp(model.omega_1.item())

    samples_theta_0 = np.random.normal(loc=mu_0, scale=sd_0, size=num_samples * 10)
    samples_theta_1 = np.random.normal(loc=mu_1, scale=sd_1, size=num_samples * 10)

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Log Loss')

    ax2.hist(samples_theta_0)
    ax2.set_xlabel('$\\theta_0$')
    ax2.set_ylabel('$p(\\theta_0)$')

    ax3.hist(samples_theta_1)
    ax3.set_xlabel('$\\theta_1$')
    ax3.set_ylabel('$p(\\theta_1)$')

    plt.show()

    print('ho gaya')
