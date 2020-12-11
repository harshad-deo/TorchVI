import numpy as np
import torch
from torch import optim, distributions, nn
from tqdm import tqdm

from torchvi.vdistributions import Normal, HalfNormal


class Model(nn.Module):
    def __init__(self, reg_prior_mu, reg_prior_sd, noise_prior_scale):
        super().__init__()
        self.theta_0 = Normal(1, reg_prior_mu, reg_prior_sd, name='theta_0')
        self.theta_1 = Normal(1, reg_prior_mu, reg_prior_sd, name='theta_1')
        self.theta_2 = HalfNormal(1, noise_prior_scale, name='theta_2')

    def forward(self, xs, ys):
        theta_0, theta_0_contrib = self.theta_0(None)
        theta_1, theta_1_contrib = self.theta_1(None)
        theta_2, theta_2_contrib = self.theta_2(None)

        y_mu = theta_0 + theta_1 * xs
        dist = distributions.Normal(loc=y_mu, scale=theta_2)

        data_lp = dist.log_prob(ys).sum()
        constraint_contrib = theta_0_contrib + theta_1_contrib + theta_2_contrib

        return constraint_contrib.add_tensor(data_lp)

    def sample(self, xs, size):
        theta_0 = torch.squeeze(self.theta_0.sample(None, size))
        theta_1 = torch.squeeze(self.theta_1.sample(None, size))
        theta_2 = torch.squeeze(self.theta_2.sample(None, size))

        return {'theta_0': theta_0, 'theta_1': theta_1, 'theta_2': theta_2}


def fit(reg_prior_mu, reg_prior_sd, noise_prior_scale, xs, ys, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(reg_prior_mu, reg_prior_sd, noise_prior_scale)

    model = model.to(device)
    xs = xs.to(device)
    ys = ys.to(device)

    optimizer = optim.Adam(model.parameters(), lr=7e-2)

    losses = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = -model(xs, ys)
        losses[i] = loss.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)
    samples = model.sample(None, num_samples * 10)

    return model, losses, samples


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
    model, losses, samples = fit(0.0, 1.0, 1.0, xs, ys, num_epochs)
    print(model)

    fig, axs = plt.subplots(2, 2)

    ax = axs[0][0]
    ax.plot(losses)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Log Loss')

    ax = axs[0][1]
    ax.hist(samples['theta_0'].cpu().numpy())
    ax.set_xlabel('$\\theta_0$')
    ax.set_ylabel('$p(\\theta_0)$')

    ax = axs[1][0]
    ax.hist(samples['theta_1'].cpu().numpy())
    ax.set_xlabel('$\\theta_1$')
    ax.set_ylabel('$p(\\theta_1)$')

    ax = axs[1][1]
    ax.hist(samples['theta_2'].cpu().numpy())
    ax.set_xlabel('$\\theta_2$')
    ax.set_ylabel('$p(\\theta_2)$')

    plt.show()

    print('ho gaya')
