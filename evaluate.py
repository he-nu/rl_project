"""
Pits two models against each other and calculates wins, losses and draws.
"""

import torch
import numpy as np

from tqdm import tqdm

from modules.net import ResNet
from modules.games import ConnectFour
from modules.alphazero import MCTS


class SelfPlay:
    def __init__(self, path_to_weights1, path_to_weights2):
        self.path_to_weights1 = path_to_weights1
        self.path_to_weights2 = path_to_weights2

    def play_game(self, model1, model2, args):
        game = ConnectFour()
        state = game.get_initial_state()
        
        # Model1 starts first
        player = 1

        mcts1 = MCTS(game, args, model1)
        mcts2 = MCTS(game, args, model2)

        while True:
            current_mcts = mcts1 if player == 1 else mcts2
            neutral_state = game.change_perspective(state, player)
            mcts_probs = current_mcts.search(neutral_state)
            action = np.argmax(mcts_probs)

            state = game.get_next_state(state, action, player)
            value, is_terminal = game.get_value_and_terminated(state, action)

            if is_terminal:
                winner = player if value == 1 else 0
                break
            
            player = game.get_opponent(player)
        
        return winner

    def init_model(self, model, game, path_to_weights, device):
        model = ResNet(game, num_res_blocks=12, num_hidden=256, device=device)
        model.load_state_dict(torch.load(path_to_weights, map_location=device))
        model.eval()
        return model
    
    
    def main(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        game = ConnectFour()

        model1 = self.init_model(
            game=game,
            path_to_weights=self.path_to_weights1,
            device=device
            )

        model2 = self.init_model(
            game=game,
            path_to_weights=self.path_to_weights2,
            device=device
        )


        args = {
            'C': 2,
            'num_searches': 1000,
            'dirichlet_epsilon': 0.25,
            'dirichlet_alpha': 0.3
        }

        results = []
        for _ in tqdm(range(100)):
            results.append(self.play_game(model1, model2, args))

        model1_wins = results.count(1)
        model2_wins = results.count(-1)
        draws = results.count(0)

        print("\nResults after 100 matches:")
        print(f"Model 1 wins: {model1_wins}")
        print(f"Model 2 wins: {model2_wins}")
        print(f"Draws: {draws}")