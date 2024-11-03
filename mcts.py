import math
import random
import logging
from copy import deepcopy
from board import Action, Board

class MCTS_CandyCrush:
    def __init__(self, board, exploration_param=1.4, max_depth=5, n_simulation=100, no_log = True, write_log_file = False):
        """
        Initialize the MCTS with the given parameters.

        Args:
            board (Board): The initial board state.
            exploration_param (float): The exploration parameter for the UCB1 formula.
            max_depth (int): The maximum depth to run the simulation.
            n_simulation (int): The number of simulations to run.
            write_log_file (bool): Whether to write logs to a file.
            no_log (bool): Whether to disable logging. 
        """
        
        self.root_board = deepcopy(board)
        self.exploration_param = exploration_param
        self.max_depth = max_depth
        self.n_simulation = n_simulation

        # Set up logging
        if no_log:
            self.logger = logging.getLogger('MCTS_CandyCrush')
            self.logger.disabled = True
        else:
            self.logger = self.setup_logger(write_log_file)

        self.Q = {}  # Stores cumulative rewards for state-action pairs
        self.N = {}  # Stores visit counts for state-action pairs
        self.N_state = {}  # Stores visit counts for each state

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


    def best_move(self):
        """Runs simulations to find the best move from the root state."""
        for i in range(self.n_simulation):
            self.logger.info(f"\n--- Simulation {i + 1} ---")
            self.run_simulation()

        # Choose the action with the highest average Q value from the root state
        root_state = self.root_board.state()
        legal_moves = self.root_board.get_legal_moves()
        best_action = max(
            legal_moves,
            key=lambda move: self.Q.get((root_state, move), 0) / (self.N.get((root_state, move), 1))
        )
        self.logger.info(f"\nBest move selected: {best_action}")
        self.logger.info("\n--- Summary of States Visited ---")
        for state, visit_count in self.N_state.items():
            # Convert the state hash to hexadecimal (first 4 characters)
            self.logger.info(f"State {hex(state)[:6]} visited {visit_count} times")

        return best_action
        return best_action

    def run_simulation(self):
        """Simulates a game sequence from the root state."""
        current_board = deepcopy(self.root_board)
        visited_state_actions = []
        depth = 0
        
        while depth < self.max_depth:
            state = current_board.state()
            legal_moves = current_board.get_legal_moves()
            self.logger.info(f"Depth {depth}: State = {hex(state)[:6]}, Legal Moves = {legal_moves}")
            
            # If there are no legal moves, end the simulation early
            if not legal_moves:
                self.logger.info("No more legal moves. Ending simulation.")
                break
            
            # Choose the move with the highest UCB1 value
            move = self.select_move(state, legal_moves)
            self.logger.info(f"Selected move: {move}")
            visited_state_actions.append((state, move))
            
            # Apply the move
            Action(current_board).swap(*move[0], *move[1])
            current_board.update()
            self.logger.info(f"New board state after move: {hex(current_board.state())[:6]}, Score: {current_board.score}")
            
            # Increase depth
            depth += 1
        
        # Calculate the reward at the end of the simulation
        reward = current_board.score
        self.logger.info(f"End of simulation with depth {depth}. Reward (Score): {reward}")
        
        # Backpropagate reward
        for state, move in visited_state_actions:
            self.N[(state, move)] = self.N.get((state, move), 0) + 1
            self.Q[(state, move)] = self.Q.get((state, move), 0) + reward
            self.N_state[state] = self.N_state.get(state, 0) + 1
            self.logger.info(f"Updated Q[{hex(state)[:6]}, {move}] = {self.Q[(state, move)]}, N[{hex(state)[:6]}, {move}] = {self.N[(state, move)]}")

    def select_move(self, state, legal_moves):
        """Selects a move using the UCB1 exploration strategy."""
        total_visits = self.N_state.get(state, 0)
        
        def ucb1(move):
            q_value = self.Q.get((state, move), 0)
            visit_count = self.N.get((state, move), 0)
            if visit_count == 0:
                return float('inf')  # Encourage exploration
            return q_value / visit_count + self.exploration_param * math.sqrt(
                math.log(total_visits) / visit_count
            )
        
        move = max(legal_moves, key=ucb1)
        self.logger.info(f"UCB1 values for moves at state {hex(state)[:6]}: {[ucb1(m) for m in legal_moves]}")
        return move

if __name__ == "__main__":
    random.seed(42)
    b = Board(5, 5)
    b.fill_random()
    b.update()
    b.display()
    
    # Initialize the MCTS with the given board `b` and log output to a file
    mcts = MCTS_CandyCrush(b, exploration_param=1.4, max_depth=5, n_simulation=500, write_log_file=True)
    
    # Run MCTS to find the best move with step-by-step logs
    best_move = mcts.best_move()  # Adjust number of simulations if needed
    print(f"\Possible moves: {b.get_legal_moves()}")
    # Display the mean reward for each possible move
    for move in b.get_legal_moves():
        state = b.state()
        print(f"Mean reward for move {move}: {mcts.Q.get((state, move), 0) / (mcts.N.get((state, move), 1))}")
    print(f"\nFinal best move from MCTS: {best_move}")

