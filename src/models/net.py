import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=2)
        self.fc1 = nn.Linear(7*7*dim, 10)
        self.dim = dim
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(-1, 7*7*self.dim)
        x = self.fc1(x)
        return x