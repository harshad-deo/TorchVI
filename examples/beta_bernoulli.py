import numpy as np
import torch
from torch import nn, distributions, optim
from tqdm import tqdm

from torchvi import vdistributions


class Model(nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.theta = vdistributions.Beta(1, alpha, beta)

    def forward(self, xs, device):
        theta, constraint_contrib = self.theta(None, device)
        dist = distributions.Bernoulli(probs=theta)
        lp = dist.log_prob(xs).sum()

        return lp + constraint_contrib

    def sample(self, size, device):
        return torch.squeeze(self.theta.sample(None, size, device))


def fit(alpha, beta, xs, num_epochs, num_samples):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Fitting on: {device}')

    model = Model(alpha, beta)
    model = model.to(device)
    xs = xs.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-1)

    losses = np.zeros(num_epochs)

    for i in tqdm(range(num_epochs)):
        optimizer.zero_grad()
        loss = -model(xs, device)
        losses[i] = loss.item()
        loss.backward()
        optimizer.step()

    losses = np.log(losses)
    samples = model.sample(num_samples * 10, device).cpu().numpy()

    return model.cpu(), losses, samples


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from utils.fix_seed import fix_seed

    fix_seed(42)

    num_samples = 100
    p_known = 0.65
    alpha_0 = 20.0
    beta_0 = 30.0

    xs = torch.bernoulli(torch.ones((num_samples, )), p=p_known)
    num_epochs = 1000

    model, losses, samples_actual = fit(alpha_0, beta_0, xs, num_epochs, num_samples)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    ax1.plot(losses)
    ax1.set_ylabel('Log Loss')
    ax1.set_xlabel('Epoch')

    sample_success = torch.sum(xs)
    alpha = alpha_0 + sample_success
    beta = beta_0 + num_samples - sample_success
    samples_expected = np.random.beta(a=alpha, b=beta, size=num_samples * 10)

    ax2.hist(samples_actual, label='actual', alpha=0.5)
    ax2.hist(samples_expected, label='expected', alpha=0.5)
    ax2.set_ylabel('$p(\\theta$)')
    ax2.set_xlabel('$\\theta$')
    ax2.legend()

    plt.show()

    print('ho gaya')
