import numpy as np
from tqdm import tqdm
import torch
from torch import nn, distributions, optim

from torchvi.vdistributions import Normal, HalfNormal, Exponential


class StandardRegression(nn.Module):
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
        theta_0 = self.theta_0.sample(None, size)
        theta_1 = self.theta_1.sample(None, size)

        xs = xs.unsqueeze(0)
        ys = theta_0 + xs * theta_1

        theta_2 = torch.squeeze(self.theta_2.sample(None, size))

        return {'ys': ys.cpu().numpy(), 'theta_2': theta_2.cpu().numpy()}


class RobustRegression(nn.Module):
    def __init__(self, reg_prior_mu, reg_prior_sd, noise_prior_scale, rate_prior):
        super().__init__()
        self.theta_0 = Normal(1, reg_prior_mu, reg_prior_sd, name='theta_0')
        self.theta_1 = Normal(1, reg_prior_mu, reg_prior_sd, name='theta_1')
        self.theta_2 = HalfNormal(1, noise_prior_scale, name='theta_2')
        self.theta_3 = Exponential(1, rate_prior, name='theta_3')

    def forward(self, xs, ys):
        theta_0, theta_0_contrib = self.theta_0(None)
        theta_1, theta_1_contrib = self.theta_1(None)
        theta_2, theta_2_contrib = self.theta_2(None)
        theta_3, theta_3_contrib = self.theta_3(None)

        y_mu = theta_0 + theta_1 * xs
        dist = distributions.StudentT(df=theta_3, loc=y_mu, scale=theta_2)

        data_lp = dist.log_prob(ys).sum()
        constraint_contrib = theta_0_contrib + theta_1_contrib + theta_2_contrib + theta_3_contrib

        return constraint_contrib.add_tensor(data_lp)

    def sample(self, xs, size):
        theta_0 = self.theta_0.sample(None, size)
        theta_1 = self.theta_1.sample(None, size)

        xs = xs.unsqueeze(0)
        ys = theta_0 + xs * theta_1

        theta_2 = torch.squeeze(self.theta_2.sample(None, size))
        theta_3 = torch.squeeze(self.theta_3.sample(None, size))

        return {'ys': ys.cpu().numpy(), 'theta_2': theta_2.cpu().numpy(), 'theta_3': theta_3.cpu().numpy()}


def fit(desc, model, lr, xs, ys, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = model.to(device)
    xs = xs.to(device)
    ys = ys.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    losses = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs), desc=desc):
        optimizer.zero_grad()
        loss = -model(xs, ys)
        losses[i] = loss.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)
    samples = model.sample(xs, num_samples * 10)

    return model.cpu(), losses, samples


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
    xs = torch.linspace(xmin, xmax, num_samples)
    ys_true = intercept + slope * xs
    ys = ys_true + noise_scale * torch.randn(num_samples)

    ys[65] = 3.0
    ys[85] = 2.0
    ys[95] = 1.0

    num_epochs = 5000

    standard_model = StandardRegression(0.0, 1.0, 1.0)
    standard_model, standard_loss, standard_samples = fit('standard', standard_model, 7e-2, xs, ys, num_epochs)
    print('------- standard model -------')
    print(standard_model)

    robust_model = RobustRegression(0.0, 1.0, 1.0, 1.0)
    robust_model, robust_loss, robust_samples = fit('robust', robust_model, 7e-2, xs, ys, num_epochs)
    print('------- robust model -------')
    print(robust_model)

    ys_standard = standard_samples['ys']
    ys_standard_quantiles = np.quantile(ys_standard, [0.025, 0.5, 0.975], axis=0)

    ys_robust = robust_samples['ys']
    ys_robust_quantiles = np.quantile(ys_robust, [0.025, 0.5, 0.975], axis=0)

    plt.scatter(xs, ys, label='Data')
    plt.plot(xs, ys_true, label='True Regression', color='black')

    plt.plot(xs, ys_standard_quantiles[1, :], label='Standard regression median', color='red', linestyle='dashed')
    plt.fill_between(xs,
                     ys_standard_quantiles[0, :],
                     ys_standard_quantiles[2, :],
                     color='red',
                     alpha=0.3,
                     label='95% CI for standard regression line')

    plt.plot(xs, ys_robust_quantiles[1, :], label='Robust regression median', color='green', linestyle='dashed')
    plt.fill_between(xs,
                     ys_robust_quantiles[0, :],
                     ys_robust_quantiles[2, :],
                     color='green',
                     alpha=0.3,
                     label='95% CI for robust regression line')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    print('ho gaya')
