import numpy as np
from tqdm import tqdm
import torch
from torch import nn, distributions, optim

from torchvi.vdistributions import Dirichlet


class Model(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.theta = Dirichlet(alpha=alpha)

    def forward(self, xs):
        theta, theta_contrib = self.theta(None)
        dist = distributions.Categorical(probs=theta)
        lp = dist.log_prob(xs).sum()
        return theta_contrib.add_tensor(lp)

    def sample(self, size):
        return torch.squeeze(self.theta.sample(None, size))


def fit(alpha, xs, num_epochs, num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(alpha)
    model = model.to(device)
    xs = xs.to(device)

    optimizer = optim.Adam(model.parameters(), lr=5e-2)

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

    num_samples = 200

    probs_known = torch.tensor([0.2, 0.15, 0.25, 0.1, 0.18, 0.12])
    dist = distributions.Categorical(probs=probs_known)
    xs = dist.sample([num_samples])

    num_epochs = 2000
    alpha = [1, 2, 3, 4, 5, 50.0]
    model, losses, samples = fit(alpha, xs, num_epochs, num_samples)
    print(model)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.plot(losses)
    ax1.set_ylabel('Log Loss')
    ax1.set_xlabel('Epoch')

    ax2.hist(xs.numpy() + 1, bins=[1, 2, 3, 4, 5, 6, 7], rwidth=0.5, align='left')
    ax2.set_xlabel('y')
    ax2.set_ylabel('Observed y')

    ax3.boxplot(samples)
    ax3.set_xlabel('y')
    ax3.set_ylabel('p(y)')

    plt.show()

    print('ho gaya')
