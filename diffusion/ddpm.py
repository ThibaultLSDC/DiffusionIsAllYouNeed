from torch.optim import Adam
from torch.nn import MSELoss

import torch

from tqdm import tqdm

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

class BaseDiffusion:
    def __init__(self, model: torch.nn.Module, T: int = 1000, beta_scales: list[float] = [1e-4, 0.02]) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.optim = Adam(model.parameters(), lr=2e-4)

        self.mse = MSELoss()

        self.T = T

        slope = (beta_scales[1] - beta_scales[0]) / T
        self.beta = torch.tensor([beta_scales[0] + slope * i for i in range(T)]).to(self.device)
        self.alpha = 1 - self.beta

        self.alphas = torch.cumprod(self.alpha, dim=0)

        self.sigmas = torch.sqrt(self.beta)

    def train(self, loader : 'DataLoader', epoch: int):
        counter = tqdm(loader)
        total_loss = 0
        for i, (x, _) in enumerate(counter):
            x = x.to(self.device)
            
            t = torch.randint(0, self.T, size=(x.size(0),), device=self.device)
            alpha = self.alphas[t][..., None, None, None]

            eps = torch.randn_like(x)

            obj = (torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * eps)

            x_t = self.model(obj, t)

            loss = self.mse(x_t, eps)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            total_loss += loss.cpu().detach().item()

            counter.set_description(f"Epoch {epoch+1}, Loss is {total_loss / (i+1):4f}")
        
        return total_loss / (i+1)
        
    def sample(self, example_batch):

        x = torch.randn_like(example_batch)

        x_0 = torch.clone(x)

        samples = [x_0]

        for t in tqdm(range(self.T - 1, -1, -1), desc='Sampling...'):
            z = torch.randn_like(x)

            with torch.no_grad():
                eps = self.model(x, t)

            mean = (x - eps * (1 - self.alpha[t]) / (1 - self.alphas[t])**.5) / self.alpha[t]**.5 

            x = mean + self.sigmas[t] * z

            if t % 100 == 0:
                samples.append(torch.clone(x))
        
        return x, samples