import torch
import numpy as np

from modules.net import ResNet
from modules.games import ConnectFour
from modules.alphazero import MCTS


class AlphaZeroPlay:

    def __init__(self, path_to_weights=""):
        self.path_to_weights = path_to_weights
    

    def __validate_path(self):
        if (not self.path_to_weights 
            or not isinstance(self.path_to_weights, str)):
            exit("Path cannot be empty.")


    def play(self):
        game = ConnectFour()
        player = 1
        state = game.get_initial_state()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        args = {
            'C': 2,
            'num_searches': 1000,
            'dirichlet_epsilon': 0.25,
            'dirichlet_alpha': 0.3
        }

        torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = ResNet(game, num_res_blocks=9, num_hidden=128, device=device)
        self.__validate_path()
            
        model.load_state_dict(torch.load(self.path_to_weights, map_location=device))
        model.eval()

        mcts = MCTS(game, args, model)
        while True:
            print(state)
            if player == 1:

                valid_moves = game.get_valid_moves(state)
                print("valid moves", [i for i in range(
                    game.action_size) if valid_moves[i] == 1])
                action = int(input(f"{player}: "))

                if valid_moves[action] == 0:
                    print("Action not valid")
                    continue

            else:
                neutral_state = game.change_perspective(state, player)
                mcts_probs = mcts.search(neutral_state)
                action = np.argmax(mcts_probs)

            state = game.get_next_state(state, action, player)

            value, is_terminal = game.get_value_and_terminated(
                state, 
                action)

            if is_terminal:
                print(state)
                if value == 1:
                    print(f"{player} won")
                else:
                    print("draw")
                break    
        
            player = game.get_opponent(player)


if __name__ == "__main__":
    ...