import random
import math


import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch


class Node:
    def __init__(
            self, game, args, state,
            parent=None,
            action_taken=None,
            prior=0,
            visit_count=0
            ):
        
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0

    def is_fully_expanded(self):
        return len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_upper_confidence_bound(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child
    
    def get_upper_confidence_bound(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * math.sqrt(
            self.visit_count / (child.visit_count + 1)) * child.prior
    

    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                player = 1
                child_state = self.game.get_next_state(
                    child_state,
                    action,
                    player=player
                )
                child_state = self.game.change_perspective(
                    child_state,
                    player=-player
                )
                child = Node(
                    self.game,
                    self.args,
                    child_state,
                    self,
                    action,
                    prob
                )
                self.children.append(child)

        return child

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        value = self.game.get_opponent_value(value)
        if self.parent is not None:
           self.parent.backpropagate(value)
        return None

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model


    @torch.no_grad()
    def search(self, state):
        root = Node(self.game, self.args, state, visit_count=1)

        policy, _ = self.model(
            torch.tensor(
                self.game.get_encoded_state(state),
                device=self.model.device
            ).unsqueeze(0)
        )
        policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state)
        policy *= valid_moves
        policy /= np.sum(policy)

        root.expand(policy)
        
        for _ in range(self.args['num_searches']):
            # selection
            node = root

            # expansion
            while node.is_fully_expanded():
                node = node.select()

            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken)
            value = self.game.get_opponent_value(value)

            if not is_terminal:
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0)
                )
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state)
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)
                
            # backprop
            node.backpropagate(value)

        # return visit_counts
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs
    


class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(game, args, model)
        
    def self_play(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)

            # changed to temp_probs, but video has action_probs
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)
            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                return_memory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = (value if hist_player == player else self.game.get_opponent_value(value))
                    return_memory.append((
                        self.game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome
                    ))
                return return_memory
            
            player = self.game.get_opponent(player)

    def train(self, memory):
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batch_idx:min(len(memory) - 1, batch_idx + self.args['batch_size'])]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)

            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return None


    def learn(self):
        for i in range(self.args['num_iterations']):
            memory = []
            self.model.eval()
            for self_play_iteration in tqdm(range(self.args['num_self_play_iterations'])):
                memory += self.self_play()

            self.model.train()
            for _ in tqdm(range(self.args['num_epochs'])):
                self.train(memory)

            torch.save(self.model.state_dict(), f"weights/model_{i}_{self.game}.pt")
            torch.save(self.optimizer.state_dict(), f"weights/optimizer_{i}_{self.game}.pt")
        return None

class MCTSParallel:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model

    @torch.no_grad()
    def search(self, states, self_play_games):
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
        )
        policy = torch.softmax(policy, dim=1).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size, size=policy.shape[0])

        for i, spg in enumerate(self_play_games):
            spg_policy = policy[i]    
            valid_moves = self.game.get_valid_moves(states[i])
            spg_policy *= valid_moves
            spg_policy /= np.sum(spg_policy )

            

            spg.root = Node(self.game, self.args, states[i], visit_count=1)
            spg.root.expand(spg_policy)
         
        for _ in range(self.args['num_searches']):
            for spg in self_play_games:
                spg.node = None
                # selection
                node = spg.root

                # expansion
                while node.is_fully_expanded():
                    node = node.select()

                value, is_terminal = self.game.get_value_and_terminated(
                    node.state, node.action_taken)
                value = self.game.get_opponent_value(value)

                if is_terminal:
                    # backprop
                    node.backpropagate(value)
                else:
                    spg.node = node

            expandable_self_play_games = [mapping_idx for mapping_idx in range(len(self_play_games)) if self_play_games[mapping_idx].node is not None]

            # truthy not empty list
            if expandable_self_play_games:
                states = np.stack([self_play_games[mapping_idx].node.state for mapping_idx in expandable_self_play_games])
                policy, value = self.model(
                    torch.tensor(self.game.get_encoded_state(states), device=self.model.device)
                )
                policy = torch.softmax(policy, dim=1).cpu().numpy()
                value = value.cpu().numpy()

            for i, mapping_idx in enumerate(expandable_self_play_games):
                node = self_play_games[mapping_idx].node
                spg_policy, spg_value = policy[i], value[i]

                valid_moves = self.game.get_valid_moves(node.state)
                spg_policy *= valid_moves
                spg_policy /= np.sum(spg_policy)


                node.expand(spg_policy)
                node.backpropagate(spg_value)
        return None


class AlphaZeroParallel:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTSParallel(game, args, model)

    def self_play(self):
        return_memory = []
        player = 1
        self_play_games = [SPG(self.game) for _ in range(self.args['num_parallel_games'])]

        while len(self_play_games) > 0:
            states = np.stack([spg.state for spg in self_play_games])

            neutral_states = self.game.change_perspective(states, player)
            
            self.mcts.search(neutral_states, self_play_games)

            for i in range(len(self_play_games))[::-1]:
                spg = self_play_games[i]
                
                # return visit_counts
                action_probs = np.zeros(self.game.action_size)
                for child in spg.root.children:
                    action_probs[child.action_taken] = child.visit_count
                action_probs /= np.sum(action_probs)

                spg.memory.append((spg.root.state, action_probs, player))

                temperature_action_probs = action_probs ** (1 / self.args['temperature'])
                temperature_action_probs /= np.sum(temperature_action_probs)

                action = np.random.choice(self.game.action_size, p=temperature_action_probs)
                
                spg.state = self.game.get_next_state(spg.state, action, player)

                value, is_terminal = self.game.get_value_and_terminated(spg.state, action)

                if is_terminal:
                    for hist_neutral_state, hist_action_probs, hist_player in spg.memory:
                        hist_outcome = (
                            value if hist_player == player 
                            else self.game.get_opponent_value(value)
                            )
                        return_memory.append((
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome
                        ))
                    del self_play_games[i]
            
            player = self.game.get_opponent(player)
        return return_memory


    def train(self, memory):
        random.shuffle(memory)
        for batch_idx in range(0, len(memory), self.args['batch_size']):
            sample = memory[
                batch_idx:min(
                    len(memory) - 1,
                    batch_idx + self.args['batch_size']
                    )
                ]
            state, policy_targets, value_targets = zip(*sample)
            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1)
                )
            
            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=self.model.device)
            policy_targets = torch.tensor(
                policy_targets,
                dtype=torch.float32,
                device=self.model.device
                )
            value_targets = torch.tensor(
                value_targets,
                dtype=torch.float32,
                device=self.model.device
                )

            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return None


    def learn(self):
        for i in range(self.args['num_iterations']):
            memory = []
            self.model.eval()
            for _ in tqdm(range(
                self.args['num_self_play_iterations'] 
                // self.args['num_parallel_games']
                )):
                
                memory += self.self_play()

            self.model.train()
            for eopch in tqdm(range(self.args['num_epochs'])):
                self.train(memory)

            torch.save(
                self.model.state_dict(),
                f"weights/model_{i}_{self.game}_run2.pt")
            # torch.save(self.optimizer.state_dict(), f"weights/optimizer_{i}_{self.game}.pt")


class SPG: # self play game
    def __init__(self, game):
        self.state = game.get_initial_state()
        self.memory = []
        self.root = None
        self.node = None
