import numpy as np
import torch
from torch import optim, distributions, nn
from tqdm import tqdm

from torchvi.vdistributions import Normal


class Model(nn.Module):
    def __init__(self, prior_mu, prior_sd):
        super().__init__()
        self.theta_0 = Normal(1, prior_mu, prior_sd)
        self.theta_1 = Normal(1, prior_mu, prior_sd)

    def forward(self, xs, ys):
        theta_0, theta_0_contrib = self.theta_0(None)
        theta_1, theta_1_contrib = self.theta_1(None)

        logit = theta_0 + xs * theta_1
        dist = distributions.Bernoulli(logits=logit)

        data_lp = dist.log_prob(ys).sum()
        constraint_contrib = theta_0_contrib + theta_1_contrib

        return data_lp + constraint_contrib

    def sample(self, xs, size):
        theta_0 = torch.squeeze(self.theta_0.sample(None, size))
        theta_1 = torch.squeeze(self.theta_1.sample(None, size))

        return {'theta_0': theta_0, 'theta_1': theta_1}


def fit(prior_mu, prior_sd, xs, ys, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(prior_mu, prior_sd)

    model = model.to(device)
    xs = xs.to(device)
    ys = ys.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-2)

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
    slope = -0.5
    intercept = 0.5

    xmin = -5
    xmax = 5
    xs = xmin + (xmax - xmin) * torch.rand(num_samples)
    logits = intercept + slope * xs
    y_dist = distributions.Bernoulli(logits=logits)
    ys = y_dist.sample()

    num_epochs = 5000
    model, losses, samples = fit(0, 0.5, xs, ys, num_epochs)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Log Loss')

    ax2.hist(samples['theta_0'].cpu().numpy())
    ax2.set_xlabel('$\\theta_0$')
    ax2.set_ylabel('$p(\\theta_0)$')

    ax3.hist(samples['theta_1'].cpu().numpy())
    ax3.set_xlabel('$\\theta_1$')
    ax3.set_ylabel('$p(\\theta_1)$')

    plt.show()

    print('ho gaya')
