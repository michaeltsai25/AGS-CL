import torch
from dataloaders.wrapper import Storage

class Memory(Storage):
    def reduce(self, m):
        self.storage = self.storage[:m]

    def get_tensor(self):
        storage = [x.unsqueeze(-1) for x in self.storage]
        return torch.cat(storage, axis=1)