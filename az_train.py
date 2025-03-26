import torch
from torch.optim import Adam

from modules.alphazero import AlphaZeroParallel
from modules.games import ConnectFour
from modules.net import ResNet


# Hyperparameters
lr = 0.001
weight_decay = 0.0001
resnet_blocks = 12
num_hidden = 256
num_epochs = 20
batch_size = 512

num_searches = 800
num_parallel_games = 100
num_self_play_iterations = 600
num_iterations = 15

temperature = 0.5
dirichlet_epsilon = 0.25
dirichlet_alpha = 0.314159265358979
C = 2
# ----------------------------------------------------------------------

def train():
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model = ResNet(game, resnet_blocks, num_hidden, device=device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    args = {
        'C': C,
        'num_searches': num_searches,
        'num_iterations': num_iterations,
        'num_self_play_iterations': num_self_play_iterations,
        "num_parallel_games": num_parallel_games,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'temperature': temperature,
        'dirichlet_epsilon': dirichlet_epsilon,
        'dirichlet_alpha': dirichlet_alpha
    }

    alpha_zero = AlphaZeroParallel(model, optimizer, game, args)
    alpha_zero.learn()


if __name__ == "__main__":
    train()
