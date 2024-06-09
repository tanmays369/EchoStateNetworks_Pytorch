import torch

class MackeyGlassDataset(torch.utils.data.Dataset):
    def __init__(self, n=2000, tau=30, dtype=torch.float32):
        self.n= n
        self.tau= tau
        self.dtype= dtype
        self.data= self._generate_data()

    def __len__(self):
        return self.n-self.tau

    def __getitem__(self, idx):
        return self.data[idx: idx + self.tau], self.data[idx + self.tau]

    def _generate_data(self):
        data= torch.zeros(self.n, dtype=self.dtype)
        data[0]= 0.969 * torch.sin(2 * torch.pi * 0.125 * 0.5) / (1 + 0.25 * (torch.sin(2 * torch.pi * 0.25 * 0.5))**2)
        for t in range(1, self.n):
            if t<25:
                data[t]= data[t-1] * (1-0.25 * data[t-1]) + 0.1 * data[t-1]
            else:
                data[t]= data[t-1] * (1-0.25 * data[t-25]) + 0.1 * data[t-1]
        return data
