import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super(ResNet, self).__init__()
        self.device = device
        self.start_block = nn.Sequential(
              nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
              nn.BatchNorm2d(num_hidden),
              nn.ReLU()
         )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.start_block(x)
        for ResBlock in self.backBone:
            x = ResBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x
