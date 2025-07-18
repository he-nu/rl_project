import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction_ratio=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction_ratio, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        # Ensure padding matches kernel size for same dimensions
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                  padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class EfficientResBlock(nn.Module):
    def __init__(self, num_hidden, reduction_ratio=16):
        super(EfficientResBlock, self).__init__()
        self.conv1 = DepthwiseSeparableConv(num_hidden, num_hidden)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = DepthwiseSeparableConv(num_hidden, num_hidden)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        self.se = SqueezeExcitation(num_hidden, reduction_ratio)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x += residual
        return F.relu(x, inplace=True)

class EfficientResNet(nn.Module):
    def __init__(self, game, num_res_blocks, num_hidden, device):
        super(EfficientResNet, self).__init__()
        self.device = device
        self.game = game
        
        # Initial convolution with same padding
        self.start_block = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(inplace=True)
        )
        
        # Backbone with efficient residual blocks
        self.backbone = nn.Sequential(*[
            EfficientResBlock(num_hidden)
            for _ in range(num_res_blocks)
        ])
        
        # Policy head with dimension preservation
        self.policy_head = nn.Sequential(
            DepthwiseSeparableConv(num_hidden, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(16 * game.row_count * game.column_count, game.action_size)
        )
        
        # Value head with global pooling
        self.value_head = nn.Sequential(
            DepthwiseSeparableConv(num_hidden, 1, kernel_size=1, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Tanh()
        )

        self.to(device)

    def forward(self, x):
        x = self.start_block(x)
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value