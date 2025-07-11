from dataclasses import dataclass


@dataclass
class AlphaZeroConfig:
    # Training hyperparameters
    lr: float = 0
    weight_decay: float = 0.0001
    resnet_blocks: int = 12
    num_hidden: int = 256
    num_epochs: int = 20
    batch_size: int = 512

    # Model parameters
    num_searches: int = 800
    num_parallel_games: int = 100
    num_self_play_iterations: int = 600
    num_iterations: int = 15

    # MCTS parameters
    temperature = 0.5
    dirichlet_epsilon = 0.25
    dirichlet_alpha = 0.314159265358979
    C = 2
