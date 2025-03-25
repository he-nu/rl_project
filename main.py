import torch
from torch.optim import Adam

from modules.games import ConnectFour
from modules.net import ResNet
from modules.alphazero import AlphaZeroParallel

game = ConnectFour()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet(game, 9, 128, device=device)
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


args = {
    'C': 2,
    'num_searches': 600,
    'num_iterations': 8,
    'num_self_play_iterations': 500,
    "num_parallel_games": 100,
    'num_epochs': 5,
    'batch_size': 128,
    'temperature': 1.25,
    'dirichlet_epsilon': 0.25,
    'dirichlet_alpha': 0.314159265358979
}

alpha_zero = AlphaZeroParallel(model, optimizer, game, args)


if __name__ == "__main__":
    alpha_zero.learn()