import math
import random
import logging
from copy import deepcopy
from board import Action, Board
import numpy as np
from tqdm import tqdm

class MCTS_CandyCrush_Final:
    def __init__(self, board, exploration_param=1.4, N_rollout=5, n_simulation=100, no_log = True, write_log_file = False):
        """
        Initialize the MCTS with the given parameters.

        Args:
            board (Board): The initial board state.
            exploration_param (float): The exploration parameter for the UCB1 formula.
            N_rollout (int): The maximum depth to run the simulation.
            n_simulation (int): The number of simulations to run.
            write_log_file (bool): Whether to write logs to a file.
            no_log (bool): Whether to disable logging. 
        """
        
        self.root_board = deepcopy(board)
        self.exploration_param = exploration_param
        self.N_rollout = N_rollout
        self.n_simulation = n_simulation

        # Set up logging
        if no_log:
            self.logger = logging.getLogger('MCTS_CandyCrush')
            self.logger.disabled = True
        else:
            self.logger = self.setup_logger(write_log_file)

        self.Q = {}  # Stores cumulative rewards for state-action pairs
        self.N = {}  # Stores visit counts for state-action pairs
        self.N_state = {self.root_board.state() : 0}  # Stores visit counts for each state

    def setup_logger(self, write_log_file):
        """Sets up the logger."""
        logger = logging.getLogger('MCTS_CandyCrush')
        logger.setLevel(logging.DEBUG)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # Formatter without timestamp and log level
        formatter = logging.Formatter('%(message)s')
        ch.setFormatter(formatter)

        logger.addHandler(ch)

        # File handler in write mode to clear previous contents
        if write_log_file:
            fh = logging.FileHandler('mcts_log.txt', mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger


    def best_move(self, return_all = False):
        """Runs simulations to find the best move from the root state."""
        for i in tqdm(range(self.n_simulation)):
            self.logger.info(f"\n--- Simulation {i + 1} ---")
            self.run_simulation()

        # Choose the action with the highest average Q value from the root state
        root_state = self.root_board.state()
        legal_moves = self.root_board.get_legal_moves()
        if not legal_moves:
            return None
        
        best_action = max(
            legal_moves,
            key=lambda move: self.Q.get((root_state, move), 0)
        )
        self.logger.info(f"\nBest move selected: {best_action}")
        self.logger.info("\n--- Summary of States Visited ---")
        for state, visit_count in self.N_state.items():
            # Convert the state hash to hexadecimal (first 4 characters)
            self.logger.info(f"State {hex(state)} visited {visit_count} times")

        if not return_all:
            return best_action
        else:
            all_moves_info = []
            for move in legal_moves:
                visits = self.N.get((root_state, move), 0)
                mean_reward = self.Q.get((root_state, move), 0)
                all_moves_info.append((move, visits, mean_reward))
            return best_action, all_moves_info

    def run_simulation(self, current_board=None):
        """
        Simulates a game sequence from the root state using MCTS.
        
        Args:
            current_board (Board, optional): Current game board state. If None, starts from root.
            
        Returns:
            float: The final reward (score) from this simulation path
        """

        
        # Initialize board state
        simulation_board = deepcopy(self.root_board if current_board is None else current_board)
        current_state = simulation_board.state()
        
        # Check for legal moves
        legal_moves = simulation_board.get_legal_moves()
        if not legal_moves:
            return 0
        
        # Select and apply move
        selected_move = self.select_move(current_state, legal_moves)
        Action(simulation_board).swap(*selected_move[0], *selected_move[1])
        simulation_board.update()
        
        new_state = simulation_board.state()
        self.logger.info(f"New board state after move: {hex(new_state)}, Score: {simulation_board.score}")
        
        # Handle unvisited states
        if new_state not in self.N_state:
            self.logger.info("State has not been visited before. Running random rollout.")
            reward = self._run_random_rollout(simulation_board)
            #self._update_statistics(current_state, new_state, selected_move, reward)
            self._update_state_action_stats(current_state, selected_move, reward)
            return reward
        
        # Handle visited states
        else:
            self.logger.info("State has been visited before. Running simulation.")
            current_score = simulation_board.score
            future_reward = self.run_simulation(current_board=simulation_board)
            incremental_reward = future_reward - current_score
            
            # Update statistics with the incremental reward for the current state
            self._update_state_action_stats(new_state, selected_move, incremental_reward)
            # Update statistics with the total reward for the parent state
            self._update_state_action_stats(current_state, selected_move, future_reward)
            
            # Update state visit counts
            self.N_state[new_state] = self.N_state.get(new_state, 0) + 1
            self.N_state[current_state] = self.N_state.get(current_state, 0) + 1
            
            return future_reward
        
    def _update_state_action_stats(self, state, action, reward):
        """
        Updates the Q-value and visit count for a state-action pair.
        
        Args:
            state (int): The board state hash
            action (tuple): The selected move
            reward (float): The reward to incorporate
        """
        # Update visit count
        self.N[(state, action)] = self.N.get((state, action), 0) + 1
        visit_count = self.N[(state, action)]
        
        # Update Q-value using incremental mean formula
        current_q = self.Q.get((state, action), 0)
        self.Q[(state, action)] = (current_q * (visit_count - 1) + reward) / visit_count
    

    def _run_random_rollout(self, board):
        """
        Performs a random rollout from the given board state.
        
        Args:
            board (Board): The current board state
            
        Returns:
            float: Final score after rollout
        """
        simulation_board = deepcopy(board)
        starting_state = simulation_board.state()
        self.N_state[simulation_board.state()] = 1
        if starting_state not in self.N_state:
            self.N_state[starting_state] = 0
        
        # Increment visit count
        self.N_state[starting_state] += 1
        
        for depth in range(self.N_rollout):
            legal_moves = simulation_board.get_legal_moves()
            if not legal_moves:
                break
                
            move = random.choice(legal_moves)
            Action(simulation_board).swap(*move[0], *move[1])
            simulation_board.update()
        
        final_reward = simulation_board.score
        self.logger.info(f"End of rollout at depth {depth + 1}. Reward (Score): {final_reward}")
        return final_reward
            
    def select_move(self, state, legal_moves):
        """Selects a move using the UCB1 exploration strategy."""
        total_visits = self.N_state.get(state, 0)
        
        def ucb1(move):
            q_value = self.Q.get((state, move), 0)
            visit_count = self.N.get((state, move), 0)
            if visit_count == 0:
                return float('inf')  # Encourage exploration
            return q_value + self.exploration_param * math.sqrt(
                math.log(total_visits) / visit_count
            )
        
        move = max(legal_moves, key=ucb1)
        self.logger.info(f"UCB1 values for moves at state {hex(state)}: {[ucb1(m) for m in legal_moves]}")
        return move

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    b = Board(7, 7)
    b.fill_random()
    b.update()
    
    # Initialize the MCTS with the given board `b` and log output to a file
    mcts = MCTS_CandyCrush_Final(b, exploration_param=100000, N_rollout=5, n_simulation=10000, no_log = False, write_log_file=True)
    
    # Run MCTS to find the best move with step-by-step logs
    best_move = mcts.best_move()  # Adjust number of simulations if needed
    print(f"\Possible moves: {b.get_legal_moves()}")
    # Display the mean reward for each possible move
    reward_list = []
    for move in b.get_legal_moves():
        state = b.state()
        print(f"Mean reward for move {move}: {mcts.Q.get((state, move), 0)}")
        reward_list.append(mcts.Q.get((state, move), 0))
    print(f"\nFinal best move from MCTS: {best_move}")
    b.display_move(best_move)
    # get the Second best move
    second_best_move = b.get_legal_moves()[np.argsort(reward_list)[-2]]
    print(f"\nSecond best move from MCTS: {second_best_move}")
    b.display_move(second_best_move)
    # get the Third best move
    third_best_move = b.get_legal_moves()[np.argsort(reward_list)[-3]]
    print(f"\nThird best move from MCTS: {third_best_move}")
    b.display_move(third_best_move)


    

