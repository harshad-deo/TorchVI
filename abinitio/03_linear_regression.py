import math
import numpy as np
import torch
from torch import optim, distributions
from tqdm import tqdm


class Model:
    def __init__(self, reg_prior_mu, reg_prior_sd, noise_prior_scale):
        self.mu_0 = torch.randn(1)
        self.mu_1 = torch.randn(1)
        self.mu_2 = torch.randn(1)
        self.omega_0 = torch.randn(1)
        self.omega_1 = torch.randn(1)
        self.omega_2 = torch.randn(1)

        self.mu_0.requires_grad = True
        self.mu_1.requires_grad = True
        self.mu_2.requires_grad = True
        self.omega_0.requires_grad = True
        self.omega_1.requires_grad = True
        self.omega_2.requires_grad = True

        self.norm_prior = distributions.Normal(loc=reg_prior_mu, scale=reg_prior_sd)
        self.half_norm_prior = distributions.HalfNormal(scale=noise_prior_scale)

    def __call__(self, xs, ys):
        eta_0 = torch.randn(1)
        eta_1 = torch.randn(1)
        eta_2 = torch.randn(1)

        zeta_0 = self.mu_0 + eta_0 * self.omega_0.exp()
        zeta_1 = self.mu_1 + eta_1 * self.omega_1.exp()
        zeta_2 = self.mu_2 + eta_2 * self.omega_2.exp()

        y_mu = zeta_0 + zeta_1 * xs
        theta_2 = zeta_2.exp()
        dist = distributions.Normal(loc=y_mu, scale=theta_2)

        data_lp = dist.log_prob(ys).sum()

        theta_lp = self.norm_prior.log_prob(zeta_0)
        theta_lp += self.norm_prior.log_prob(zeta_1)
        theta_lp += self.half_norm_prior.log_prob(theta_2)

        jac = zeta_2

        return data_lp + theta_lp + jac + self.omega_0 + self.omega_1 + self.omega_2

    def parameters(self):
        return [self.mu_0, self.mu_1, self.mu_2, self.omega_0, self.omega_1, self.omega_2]

    def __repr__(self):
        mu_rep = f'{[self.mu_0, self.mu_1, self.mu_2]}'
        omega_rep = f'{[self.omega_0, self.omega_1, self.omega_2]}'
        return f'mu: {mu_rep}, omega: {omega_rep}'


def fit(reg_prior_mu, reg_prior_sd, noise_prior_scale, xs, ys, num_epochs):
    model = Model(reg_prior_mu, reg_prior_sd, noise_prior_scale)
    optimizer = optim.Adam(model.parameters(), lr=7e-2)

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
    slope = -0.8
    intercept = 0.3
    noise_scale = 0.5

    xmin = -5
    xmax = 5
    xs = xmin + (xmax - xmin) * torch.rand(num_samples)
    ys = intercept + slope * xs + noise_scale * torch.randn(num_samples)

    num_epochs = 5000
    model, losses = fit(0, 1, 1, xs, ys, num_epochs)

    fig, axs = plt.subplots(2, 2)

    mu_0 = model.mu_0.item()
    mu_1 = model.mu_1.item()
    mu_2 = model.mu_2.item()

    sd_0 = math.exp(model.omega_0.item())
    sd_1 = math.exp(model.omega_1.item())
    sd_2 = math.exp(model.omega_2.item())

    samples_theta_0 = np.random.normal(loc=mu_0, scale=sd_0, size=num_samples * 10)
    samples_theta_1 = np.random.normal(loc=mu_1, scale=sd_1, size=num_samples * 10)
    samples_theta_2 = np.exp(np.random.normal(loc=mu_2, scale=sd_2, size=num_samples * 10))

    ax = axs[0][0]
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log Loss')

    ax = axs[0][1]
    ax.hist(samples_theta_0)
    ax.set_xlabel('$\\theta_0$')
    ax.set_ylabel('$p(\\theta_0)$')

    ax = axs[1][0]
    ax.hist(samples_theta_1)
    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$p(\\theta_1)$')

    ax = axs[1][1]
    ax.hist(samples_theta_2)
    ax.set_xlabel('$\\theta_2$')
    ax.set_ylabel('$p(\\theta_2)$')

    plt.show()

    print('ho gaya')
