from torch.utils.data import Dataset
from sklearn.datasets import make_moons

class MoonDataset(Dataset):
    def __init__(self, n_samples=10_000, noise=0., random_state=42):
        self.X, self.y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]