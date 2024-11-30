import math
import random
import logging
from copy import deepcopy
from board import Action, Board
import numpy as np
from tqdm import tqdm
from nn import predict

class MCTS_CandyCrush_Complex:
    def __init__(self, board, fixed_depth=False, exploration_param=1.4, N_rollout=5, n_simulation=100, no_log = True, write_log_file = False, model = None):
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
        self.state_action_to_board = {}
        self.fixed_depth = fixed_depth
        self.model = model

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

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Formatter without timestamp and log level
        formatter = logging.Formatter('%(message)s')

        # File handler in write mode to clear previous contents
        if write_log_file:
            fh = logging.FileHandler('log/mcts_log.txt', mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        # Console handler with ERROR level
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(formatter)
        logger.addHandler(ch)


        return logger


    def best_move(self, return_all = False, N_random = 3):
        """Runs simulations to find the best move from the root state."""
        for i in range(self.n_simulation):
            self.logger.info(f"\n--- Simulation {i + 1} ---")
            self.run_simulation(N_random=N_random)

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

    def run_simulation(self, current_board=None, N_random = 10, depth = 1, dist_from_bottom=False):
        """Simulates a game sequence from the root state."""
        if current_board is None:
            current_board = deepcopy(self.root_board)  # Start with the root board for the first call
        init_board_score = current_board.score
        state = current_board.state()
        legal_moves = current_board.get_legal_moves()
        if not legal_moves:
            return 0
        init_move = self.select_move(state, legal_moves)
        if (state, init_move) not in self.state_action_to_board.keys():
            self.logger.info(f"Initializing state-action pair: {(hex(state), init_move)}")
            self.state_action_to_board[(state, init_move)] = []
        if len(self.state_action_to_board[state, init_move]) > N_random-1:
            self.logger.info(f"We have enough randomness on this {(state,init_move)}")
            # Choose a random state that we saw before
            self.logger.info(f"Our self.state_action_to_board is {self.state_action_to_board[(state, init_move)]}")
            # Choose a random board with state, action pair
            current_board = random.choice(self.state_action_to_board[state, init_move])
            self.logger.info(f"New board state after forced randomness: {hex(current_board.state())}, Score: {current_board.score}")
        else:
            # Simulate the move
            Action(current_board).swap(*init_move[0], *init_move[1])
            current_board.update()
            self.logger.info(f"New board state after move: {hex(current_board.state())}, Score: {current_board.score}")
            self.state_action_to_board[state, init_move].append(current_board.copy())
            # self.logger.info(f"Updating state-action pair: {(hex(state), init_move)} to add a new state -> {hex(current_board.state())}")
        # If the new state has not been visited before, run a simulation
        if current_board.state() not in self.N_state:
            rollout_board = deepcopy(current_board)
            ## Random rollout
            self.logger.info("State has not been visited before. Running random rollout, but saving it for future.")
            # Expansion
            self.N_state[current_board.state()] = 1
            depth_rollout = self.N_rollout if self.fixed_depth==False else self.fixed_depth-depth
            if self.model == None:
                for iter in range(depth_rollout):
                    legal_moves = rollout_board.get_legal_moves()
                    if not legal_moves:
                        break
                    move = random.choice(legal_moves)
                    Action(rollout_board).swap(*move[0], *move[1])
                    rollout_board.update()
                dist_from_bottom=depth_rollout
                # Calculate the reward at the end of the simulation
                reward = (rollout_board.score - init_board_score)/(depth_rollout + 1) # We divide by the number of move made in the rollout + the initial
                self.logger.info(f"End of Random Rollout. Reward of this random rollout ({depth_rollout}) (Score): {reward}, gone from {init_board_score} to {rollout_board.score}")
            else:
                rb = rollout_board.copy()
                reward = predict(rb.board, self.model)
                self.logger.info(f"NO RANDOM ROLLOUT. NN Predicted reward: {reward}")
            # Backpropagate reward
            self.N[(state, init_move)] = self.N.get((state, init_move), 0) + 1
            temp = self.Q.get((state, init_move), 0) # Exclusively for logging
            self.Q[(state, init_move)] = (self.Q.get((state, init_move), 0) * (self.N[(state, init_move)] - 1) + reward) / self.N[(state, init_move)]
            self.logger.info(f"Q value was {temp} and now {self.Q[(state, init_move)]}")
            self.N_state[state] = self.N_state.get(state, 0) + 1
            return reward, dist_from_bottom
        
        # If the new state has been visited before, run a simulation
        else:
            self.logger.info("State has been visited before. Running simulation.")

            current_score = current_board.score
            depth += 1 # We increase the depth as we go deeper in the simulation
            if self.fixed_depth!=False and depth > self.fixed_depth:
                self.logger.info("Maximum depth reached. We stop here.")
                return current_score - init_board_score,0
            reward, dist_from_bottom = self.run_simulation(current_board=current_board.copy(), depth = depth, N_random = N_random, dist_from_bottom = dist_from_bottom)
            dist_from_bottom += 1
            # dist_from_bottom is the distance from current_board to the bottom of the tree
            # We divide by the number of move made in the rollout + the depth in.
            true_reward = (reward * dist_from_bottom + (current_score - init_board_score) )/ (dist_from_bottom+1)

            self.logger.info(f"Reward of deeper {reward}, and of init move ({current_score - init_board_score}).")
            self.logger.info(f"The distance from the bottom is {dist_from_bottom} and the depth {depth}.")
            self.logger.info(f"We backpropagate {true_reward} to ({hex(state), init_move}).")
            # Backpropagate reward
            self.logger.info(f"Updated state visit count: {hex(current_board.state())} = {self.N_state[current_board.state()]}")
            self.N[(state, init_move)] = self.N.get((state, init_move), 0) + 1
            temp = self.Q.get((state, init_move), 0)
            self.Q[(state, init_move)] = (self.Q.get((state, init_move), 0) * (self.N[(state, init_move)] - 1) + true_reward) / self.N[(state, init_move)]
            self.logger.info(f"We had {temp} and now {self.Q[(state, init_move)]}.")
            self.N_state[state] = self.N_state.get(state, 0) + 1
            return true_reward, dist_from_bottom
        
            
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
    b = Board(6, 6)
    b.fill_random()
    b.update()
    b.score = 0
    # Initialize the MCTS with the given board `b` and log output to a file
    mcts = MCTS_CandyCrush_Complex(b, exploration_param=1000, N_rollout=4, n_simulation=3000, no_log = False, write_log_file=True, fixed_depth=5)
    
    # Run MCTS to find the best move with step-by-step logs
    best_move = mcts.best_move(N_random=3)  # Adjust number of simulations if needed
    print(f"Possible moves: {b.get_legal_moves()}")
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


    

