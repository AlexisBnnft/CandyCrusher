import torch
from mcts_complex import MCTS_CandyCrush_Complex
from board import Board
import random
from nn import *
from board import Action


strategy_params = {
    'random': {},
    'mcts_base': {
        'exploration_param': 2000,
        'N_rollout': 4,
        'n_simulation': 1500,
        'no_log': True,
        'write_log_file': False,
        'model': None
    },
    'mcts_fixed_depth': {
        'exploration_param': 2000,
        'N_rollout': 4,
        'n_simulation': 1500,
        'fixed_depth': 4,
        'no_log': True,
        'write_log_file': False,
        'model': None
    },
    'offline': {
        'model_path': 'model.pth'
    },
    'combined': {
        'exploration_param': 2000,
        'N_rollout': 4,
        'n_simulation': 1500,
        'fixed_depth': 4,
        'no_log': True,
        'write_log_file': False,
        'model': 'model.pth'
    }
}

class Master:
    def __init__(self, strategy, board, n_moves=10):
        self.strategy = strategy
        self.board = board
        self.params = strategy_params.get(strategy, {})
        self.model = None
        if 'model_path' in self.params:
            self.model = load_model(self.params['model_path'],"cpu")
        self.n_moves = n_moves
    
    def run_one_move(self, custom_params=None):
        params = self.params.copy()
        if custom_params:
            params.update(custom_params)
        
        if self.strategy == 'random':
            return self.random_strategy()
        elif self.strategy == 'mcts_base':
            return self.mcts_strategy(params)
        elif self.strategy == 'mcts_fixed_depth':
            return self.mcts_strategy(params, fixed_depth=True)
        elif self.strategy == 'offline':
            return self.offline_strategy()
        elif self.strategy == 'combined':
            return self.combined_strategy(params)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
    def run_simulation(self):
        for _ in range(self.n_moves):
            move = self.run_one_move()
            if not move:
                break
            Action(self.board).swap(*move[0], *move[1])
            self.board.update()
        return self.board.score
    
    def random_strategy(self):
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)
    
    def mcts_strategy(self, params, fixed_depth=False):
        if fixed_depth:
            params['fixed_depth'] = params.get('fixed_depth', 5)
        mcts = MCTS_CandyCrush_Complex(self.board, **params)
        return mcts.best_move()
    
    def offline_strategy(self):
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return None
        best_move = None
        best_score = float('-inf')
        for move in legal_moves:
            new_board = self.board.copy()
            a=Action(new_board)
            a.swap(*move[0], *move[1])
            score = predict(new_board.board, self.model)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
    
    def combined_strategy(self, params):
        params['model'] = self.model
        mcts = MCTS_CandyCrush_Complex(self.board, **params)
        return mcts.best_move()