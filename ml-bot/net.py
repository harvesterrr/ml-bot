import torch.nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(len(arr), 32) # input layer
        self.fc2 = nn.Linear(32, 32) # hidden layer
        self.fc3 = nn.Linear(32, len(tags)) # output layer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
