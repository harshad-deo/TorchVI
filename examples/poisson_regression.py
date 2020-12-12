import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from torch import nn, distributions, optim

from torchvi.vdistributions import Normal


class Model(nn.Module):
    def __init__(self, size, loc, scale):
        super().__init__()
        self.theta_0 = Normal(size=size, loc=loc, scale=scale, name='theta_0')
        self.theta_1 = Normal(size=size, loc=loc, scale=scale, name='theta_1')

    def forward(self, xs, xs_math, ys):
        theta_0, theta_0_contrib = self.theta_0.forward(None)
        theta_1, theta_1_contrib = self.theta_1.forward(None)
        intercept = torch.matmul(xs, theta_0)
        slope = torch.matmul(xs, theta_1)

        log_rate = intercept + slope * xs_math
        dist = distributions.Poisson(rate=log_rate.exp())
        data_lp = dist.log_prob(ys)
        constraint_contrib = theta_0_contrib + theta_1_contrib
        return constraint_contrib.add_tensor(data_lp.sum())

    def sample(self, size):
        theta_0 = self.theta_0.sample(None, size).cpu()
        theta_1 = self.theta_1.sample(None, size).cpu()
        return {'theta_0': theta_0, 'theta_1': theta_1}


def fit(size, loc, scale, xs, xs_math, ys, num_epochs, num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(size, loc, scale)
    model = model.to(device)
    xs = xs.to(device)
    xs_math = xs_math.to(device)
    ys = ys.to(device)

    model(xs, xs_math, ys)

    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    losses = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = -model(xs, xs_math, ys)
        losses[i] = loss.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)
    samples = model.sample(num_samples)

    return model.cpu(), losses, samples


def load_data():
    raw = pd.read_csv('external/poisson_sim/file/poisson_sim.csv')

    xs_prog = pd.get_dummies(raw['prog'], prefix='prog')
    xs_math = raw['math']
    xs_math = (xs_math - xs_math.mean()) / xs_math.std()  # standardising

    ys = torch.from_numpy(raw['num_awards'].to_numpy(dtype=np.float32))

    return xs_prog, xs_math, ys


if __name__ == "__main__":
    import logging
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed
    fix_seed(42)
    logging.basicConfig(format='%(asctime)s - [%(name)25s]:[%(lineno)4d]:[%(levelname)5s] - %(message)s',
                        level=logging.INFO)

    xs_prog, xs_math, ys = load_data()
    xs_stacked = [ys[xs_prog['prog_1'] == 1], ys[xs_prog['prog_2'] == 1], ys[xs_prog['prog_3'] == 1]]

    plt.hist(xs_stacked, label=['prog_1', 'prog_2', 'prog_3'], align='left', bins=[0, 1, 2, 3, 4, 5, 6, 7])
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

    xs = torch.from_numpy(xs_prog.to_numpy(dtype=np.float32))
    xs_math = torch.from_numpy(xs_math.to_numpy(dtype=np.float32))
    num_samples = 1000
    num_epochs = 5000

    model, losses, samples = fit(size=3,
                                 loc=0.0,
                                 scale=1.0,
                                 xs=xs,
                                 xs_math=xs_math,
                                 ys=ys,
                                 num_epochs=num_epochs,
                                 num_samples=num_samples)
    print(model)

    plt.plot(losses)
    plt.show()
    plt.clf()
    plt.close()

    samples_theta_0 = samples['theta_0']
    samples_theta_1 = samples['theta_1']

    fig, axs = plt.subplots(2, 3)

    ax = axs[0][0]
    ax.hist(samples_theta_0[:, 0].numpy())
    ax.set_xlabel('$\\alpha_0$')
    ax.set_ylabel('$p(\\alpha_0)$')

    ax = axs[0][1]
    ax.hist(samples_theta_0[:, 1].numpy())
    ax.set_xlabel('$\\alpha_1$')
    ax.set_ylabel('$p(\\alpha_1)$')

    ax = axs[0][2]
    ax.hist(samples_theta_0[:, 2].numpy())
    ax.set_xlabel('$\\alpha_2$')
    ax.set_ylabel('$p(\\alpha_2)$')

    ax = axs[1][0]
    ax.hist(samples_theta_1[:, 0].numpy())
    ax.set_xlabel('$\\mu_0$')
    ax.set_ylabel('$p(\\mu_0)$')

    ax = axs[1][1]
    ax.hist(samples_theta_1[:, 1].numpy())
    ax.set_xlabel('$\\mu_1$')
    ax.set_ylabel('$p(\\mu_1)$')

    ax = axs[1][2]
    ax.hist(samples_theta_1[:, 2].numpy())
    ax.set_xlabel('$\\mu_2$')
    ax.set_ylabel('$p(\\mu_2)$')

    plt.show()
    plt.show()
    plt.clf()
    plt.close()

    intercept = torch.matmul(xs, samples_theta_0.t())
    slope = torch.matmul(xs, samples_theta_1.t())

    log_rate = intercept + slope * xs_math.unsqueeze(1)
    dist = distributions.Poisson(rate=log_rate.exp())
    pois_samples = dist.sample()
    plt.clf()
    plt.close()

    xs_samples_stacked = [
        pois_samples[[xs_prog['prog_1'] == 1]], pois_samples[[xs_prog['prog_2'] == 1]],
        pois_samples[[xs_prog['prog_3'] == 1]]
    ]

    xs_samples_stacked = [torch.flatten(x) for x in xs_samples_stacked]

    plt.hist(xs_samples_stacked, label=['prog_1', 'prog_2', 'prog_3'], align='left', bins=[0, 1, 2, 3, 4, 5, 6, 7])
    plt.legend()
    plt.show()

    print("ho gaya")
