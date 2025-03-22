from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

class MNISTData:
    def __init__(self, mode="train"):
        if mode == "train":
            dataset = MNIST(root='./data', train=True, download=True)
            self.dataset = [(ToTensor()(img), target) for img, target in dataset]
        else:
            dataset = MNIST(root='./data', train=False, download=True)
            self.dataset = [(ToTensor()(img), target) for img, target in dataset]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    @classmethod
    def get_dataloader(cls, dataset_kwargs, dataloader_kwargs):
        return DataLoader(cls(**dataset_kwargs), **dataloader_kwargs)