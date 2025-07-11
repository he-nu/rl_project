import torch
from torch.optim import Adam

from modules.alphazero import AlphaZeroParallel
from modules.games import ConnectFour
from modules.net import ResNet

from config.AlphaZeroConfig import AlphaZeroConfig



def train(config: AlphaZeroConfig):
    game = ConnectFour()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    model = ResNet(game, config.resnet_blocks, config.num_hidden, device=device)
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


    args = {
        'C': config.C,
        'num_searches': config.num_searches,
        'num_iterations': config.num_iterations,
        'num_self_play_iterations': config.num_self_play_iterations,
        "num_parallel_games": config.num_parallel_games,
        'num_epochs': config.num_epochs,
        'batch_size': config.batch_size,
        'temperature': config.temperature,
        'dirichlet_epsilon': config.dirichlet_epsilon,
        'dirichlet_alpha': config.dirichlet_alpha
    }

    alpha_zero = AlphaZeroParallel(model, optimizer, game, args)
    alpha_zero.learn()


if __name__ == "__main__":
    train()
