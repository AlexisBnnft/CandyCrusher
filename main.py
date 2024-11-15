import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import sys
import numpy as numpy
import asyncio
import numpy as numpy



import math
import random
import logging
from copy import deepcopy

class MCTS_CandyCrush_Complex:
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
        self.state_action_to_board = {}

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


    async def best_move(self, return_all = False, N_random = 3):
        """Runs simulations to find the best move from the root state."""
        for i in range(self.n_simulation):
            self.logger.info(f"\n--- Simulation {i + 1} ---")
            await self.run_simulation(N_random=N_random)

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
        asyncio.sleep(0.1)

        if not return_all:
            return best_action
        else:
            all_moves_info = []
            for move in legal_moves:
                visits = self.N.get((root_state, move), 0)
                mean_reward = self.Q.get((root_state, move), 0)
                all_moves_info.append((move, visits, mean_reward))
            return best_action, all_moves_info

    async def run_simulation(self, current_board=None, N_random = 10, depth = 0):
        """Simulates a game sequence from the root state."""
        asyncio.sleep(0)
        
        if current_board is None:
            current_board = deepcopy(self.root_board)  # Start with the root board for the first call
        state = current_board.state()
        legal_moves = current_board.get_legal_moves()
        if not legal_moves:
            return 0
        init_move = self.select_move(state, legal_moves)
        if (state, init_move) not in self.state_action_to_board.keys():
            self.logger.info(f"Initializing state-action pair: {(hex(state), init_move)}")
            self.state_action_to_board[(state, init_move)] = []
        if len(self.state_action_to_board[state, init_move]) > N_random:
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
            self.logger.info(f"Updating state-action pair: {(hex(state), init_move)} to add a new state -> {hex(current_board.state())}")
        # If the new state has not been visited before, run a simulation
        if current_board.state() not in self.N_state:
            rollout_board = deepcopy(current_board)
            ## Random rollout
            self.logger.info("State has not been visited before. Running random rollout, but saving it for future.")
            # Expansion
            self.N_state[current_board.state()] = 1
            for iter in range(self.N_rollout):
                legal_moves = current_board.get_legal_moves()
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                Action(rollout_board).swap(*move[0], *move[1])
                rollout_board.update()
                # Increase depth
                depth += 1 
            # Calculate the reward at the end of the simulation
            reward = current_board.score
            self.logger.info(f"End of simulation with depth {depth}. Reward (Score): {reward}")
            self.N[(current_board.state(), init_move)] = self.N.get((current_board.state(), init_move), 0) + 1
            self.Q[(current_board.state(), init_move)] = (self.Q.get((current_board.state(), init_move), 0) * (self.N[(current_board.state(), init_move)] - 1) + reward) / self.N[(current_board.state(), init_move)]

            # Backpropagate reward
            self.N[(state, init_move)] = self.N.get((state, init_move), 0) + 1
            self.Q[(state, init_move)] = (self.Q.get((state, init_move), 0) * (self.N[(state, init_move)] - 1) + reward) / self.N[(state, init_move)]

            self.N_state[state] = self.N_state.get(state, 0) + 1
            self.logger.info(f"We know have in Q and N a number of states and moves: , {len(self.Q)}")
            return reward
        
        # If the new state has been visited before, run a simulation
        else:
            self.logger.info("State has been visited before. Running simulation.")
            #self.logger.info(current_board.display())

            current_score = current_board.score
            reward = await self.run_simulation(current_board=current_board.copy(), depth = depth + 1, N_random = N_random)
            true_reward = reward - current_score
            # Backpropagate reward
            self.N[(current_board.state(), init_move)] = self.N.get((current_board.state(), init_move), 0) + 1
            self.Q[(current_board.state(), init_move)] = (self.Q.get((current_board.state(), init_move), 0) * (self.N[(current_board.state(), init_move)] - 1) + true_reward) / self.N[(current_board.state(), init_move)]
            self.N_state[current_board.state()] = self.N_state.get(current_board.state(), 0) + 1
            
            self.N[(state, init_move)] = self.N.get((state, init_move), 0) + 1
            self.Q[(state, init_move)] = (self.Q.get((state, init_move), 0) * (self.N[(state, init_move)] - 1) + reward) / self.N[(state, init_move)]
            self.N_state[state] = self.N_state.get(state, 0) + 1
            return reward
        
            
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
N_CANDY = 6 # Number of different candies
NUM_DISCO = 7
TYPES = {'normal','raye_hor','raye_ver','sachet','disco','empty'}
TYPE_TO_ID = {'normal':0,'raye_hor':1,'raye_ver':2,'sachet': 3,'disco':4,'empty':5}
ID_TO_TYPE = {0:'normal',1:'raye_hor',2:'raye_ver',3:'sachet',4:'disco',5:'empty'}
TYPE_DISPLAY = {'normal':'N','raye_hor':'H','raye_ver':'V','sachet':'S','disco':'D','empty':'E'}
class Candy:
    def __init__(self, id, type='normal'):
        self.id=id
        if type not in TYPES:
                raise ValueError(f"Type {type} is not a valid candy type.")
        self.type=type

    def __str__(self):
        if self.type == 'empty':
            return ' '
        return str(self.id)
    
    def __repr__(self):
        return f"Candy({self.id})"
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __ne__(self, other):
        return self.id != other.id
    
    def get_type(self):
        return self.type

DIRECTIONS = {
    'UP': 1,
    'DOWN': 2,
    'LEFT': 3,
    'RIGHT': 4
}

class Board:
    def __init__(self, N, M): # N rows, M columns
        """
        Initialize the board with N rows and M columns.
        """
        self.N = N
        self.M = M
        self.board = numpy.array([[Candy(0,'empty') for _ in range(M)] for _ in range(N)])
        self.score = 0
    
    def __eq__(self, value: object) -> bool:
        """
        Check if two boards are equal.
        """
        return numpy.all(self.board == value.board)

    def display(self):
        """
        Display the current state of the board.
        """
        viz_board = numpy.array([[str(self.board[i, j]) for j in range(self.M)] for i in range(self.N)])
        for row in viz_board:
            print('|' + ' '.join(row) + '|')
        print()

    def display_with_type(self):
        """
        Display the current state of the board.
        """
        viz_board = numpy.array([[str(self.board[i, j]) + "_"+ str(TYPE_DISPLAY[self.board[i, j].type]) for j in range(self.M)] for i in range(self.N)])
        for row in viz_board:
            print('|' + ' '.join(row) + '|')
        print()


    def empty(self):
        """
        Empty the board.
        """
        self.board = numpy.array([[Candy(0,'empty') for _ in range(self.M)] for _ in range(self.N)])

    def copy(self):
        """
        Return a copy of the board, piece by piece.
        """
        new_board = Board(self.N, self.M)
        for i in range(self.N):
            for j in range(self.M):
                id, type = self.board[i, j].id, self.board[i, j].type
                new_board.board[i, j] = Candy(id, type)
        new_board.score = self.score
        return new_board

    def is_full(self):
        """
        Check if the board is full (no empty spaces).
        """
        return numpy.all(self.board != Candy(0,'empty'))
    
    def state(self):
        """
        Get the state of the board as a unique integer hash.
        
        Returns:
            int: A unique integer representing the board state.
        """
        # Flatten the board and concatenate each cell's value into a single string
        flattened = ''.join(str(cell.id) + str(TYPE_TO_ID[(cell.type)]) for row in self.board for cell in row)
        return int(flattened)
    def fill_random(self):
        """
        Fill empty spaces in the board with random candies.
        """
        recently_added=[]
        for row in range(self.N):
            for col in range(self.M):
                if self.board[row, col] == Candy(0,'empty'):
                    self.board[row, col] = Candy(numpy.random.randint(1, N_CANDY + 1))
                    recently_added.append((row, col))
        return recently_added

    def add_piece(self, candy : Candy, row, col):
        """
        Add a candy to the board at the specified position.
        """
        if self.board[row, col] == Candy(0,'empty'):
            self.board[row, col] = candy
            return True
        return False
    
    def make_it_fall(self):
        """
        Make candies fall down to fill empty spaces.
        """
        fall=[]
        for col in range(self.M):
            for row in range(self.N-1, -1, -1):
                if self.board[row, col] == Candy(0,'empty'):
                    for r in range(row-1, -1, -1):
                        if self.board[r, col] != Candy(0,'empty'):
                            self.board[row, col] = self.board[r, col]
                            self.board[r, col] = Candy(0,'empty')
                            fall.append((row, col))
                            break
        return fall

    def check_if_merged_all(self,fall=[],from_move=[]):
        """
        Attention: this may change stuff in the board.
        Check if there are any matches on the entire board. The candies in recently_added might change type.
        """
        matches = []
        new_type = []
        already_visited = []
        recently_added=fall+from_move

        ### TO DO
        # Check is type not normal and goes into a line, should pop so not add to new_type
        ###


        # Different cases when swapping two typed candies
        if len(from_move)>0:
            row1, col1 = from_move[0]
            row2, col2 = from_move[1]

            if self.board[row1, col1].type=='disco' and self.board[row2, col2].type=='normal':
                for i in range(self.N):
                    for j in range(self.M):
                        if self.board[i, j].id==self.board[row2, col2].id:
                            matches.append((i, j))
                self.board[row1, col1].id=numpy.random.randint(1, N_CANDY + 1)
                self.board[row1, col1].type='normal'
                self.score+=5000
                matches.append((row1, col1))

                return matches
            
            if self.board[row1, col1].type=='normal' and self.board[row2, col2].type=='disco':
                for i in range(self.N):
                    for j in range(self.M):
                        if self.board[i, j].id==self.board[row1, col1].id:
                            matches.append((i, j))
                self.board[row2, col2].id=numpy.random.randint(1, N_CANDY + 1)
                self.board[row2, col2].type='normal'
                self.score+=5000
                matches.append((row2, col2))

                return matches


            if self.board[row1, col1].type!='normal' and self.board[row2, col2].type!='normal':
                
                if self.board[row1, col1].type=='disco' and self.board[row2, col2].type=='disco':
                    for i in range(self.N):
                        for j in range(self.M):
                            matches.append((i, j))
                    self.board[row1, col1].id=numpy.random.randint(1, N_CANDY + 1)
                    self.board[row1, col1].type='normal'
                    self.board[row2, col2].id=numpy.random.randint(1, N_CANDY + 1)
                    self.board[row2, col2].type='normal'
                    self.score+=10000
                    return matches

                elif self.board[row1, col1].type=='disco' and self.board[row2, col2].type in {'raye_ver','raye_hor','sachet'}:
                    for i in range(self.N):
                        for j in range(self.M):
                            if self.board[i, j].id==self.board[row2, col2].id:
                                self.board[i, j].type=self.board[row2, col2].type
                                matches.append((i, j))
                    self.board[row1, col1].id=numpy.random.randint(1, N_CANDY + 1)
                    self.board[row1, col1].type='normal'
                    self.score+=5000
                    matches.append((row1, col1))
                    return matches
                elif self.board[row1, col1].type in {'raye_ver','raye_hor','sachet'} and self.board[row2, col2].type=='disco':
                    for i in range(self.N):
                        for j in range(self.M):
                            if self.board[i, j].id==self.board[row1, col1].id:
                                self.board[i, j].type=self.board[row1, col1].type
                                matches.append((i, j))
                    self.board[row2, col2].id=numpy.random.randint(1, N_CANDY + 1)
                    self.board[row2, col2].type='normal'
                    self.score+=5000
                    matches.append((row2, col2))
                    return matches
                
                elif self.board[row1, col1].type in {'raye_ver','raye_hor'} and self.board[row2, col2].type in {'raye_ver','raye_hor'}:
                    self.board[row1,col1].type='raye_ver'
                    self.board[row2,col2].type='raye_hor'
                    matches.append((row1, col1))
                    matches.append((row2, col2))

                elif self.board[row1, col1].type in {'raye_ver','raye_hor'} and self.board[row2, col2].type=='sachet':
                    self.board[row1,col1].type='normal'
                    self.board[row2,col2].type='normal'
                    for i in range(max([0,row1-1]),min([self.N,row1+2])):
                        for j in range(self.M):
                            matches.append((i,j))
                    for i in range(self.N):
                        for j in range(max([0,col1-1]),min([self.M,col1+2])):
                            matches.append((i,j))
                    return matches

                elif self.board[row1, col1].type=='sachet' and self.board[row2, col2].type in {'raye_ver','raye_hor'}:
                    self.board[row1,col1].type='normal'
                    self.board[row2,col2].type='normal'
                    for i in range(max([0,row1-1]),min([self.N,row1+2])):
                        for j in range(self.M):
                            matches.append((i,j))
                    for i in range(self.N):
                        for j in range(max([0,col1-1]),min([self.M,col1+2])):
                            matches.append((i,j))
                    return matches

                elif self.board[row1, col1].type=='sachet' and self.board[row2, col2].type=='sachet':
                    self.board[row1,col1].type='normal'
                    self.board[row2,col2].type='normal'
                    # Same as one sachet pop but with rows that are from +2 to -2 and cols that are from +2 to -2
                    for i in range(max([0,row1-2]),min([self.N,row1+3])):
                        for j in range(max([0,col1-2]),min([self.M,col1+3])):
                            matches.append((i,j))
                    return matches


        # For a "normal" swap, we need to check for matches in both directions for each cell

        for row_study in range(self.N):
            for col_study in range(self.M):

                if self.board[row_study, col_study] == Candy(0,'empty'):
                    raise ValueError("Empty cell found in the board.")
                
                if (row_study, col_study) not in already_visited:
                    
                    # Check horizontal
                    match_length = 1
                    while col_study + match_length < self.M and self.board[row_study, col_study] == self.board[row_study, col_study + match_length]:
                        match_length += 1
                        if match_length==3:
                            already_visited.append((row_study, col_study + 1))
                        if match_length>=3:
                            already_visited.append((row_study, col_study + match_length))
                    if match_length >= 3:
                        one_already_transformed=False
                        if match_length==4: # Add raye
                            for i in range(match_length):
                                if (row_study, col_study + i) in recently_added and not one_already_transformed and self.board[row_study, col_study + i].type=='normal':
                                    self.board[row_study, col_study + i].type='raye_ver'
                                    new_type.append((row_study, col_study + i))
                                    one_already_transformed=True
                                else:
                                    matches.append(((row_study, col_study + i)))
                        elif match_length>=5: # Add disco
                            for i in range(match_length):
                                if (row_study, col_study + i) in recently_added and not one_already_transformed and self.board[row_study, col_study + i].type=='normal':
                                    self.board[row_study, col_study + i].type='disco'
                                    self.board[row_study, col_study + i].id=NUM_DISCO
                                    new_type.append((row_study, col_study + i))
                                    one_already_transformed=True
                                else:
                                    matches.append(((row_study, col_study + i)))

                        for i in range(match_length):
                            matches.append(((row_study, col_study + i)))
                            #Check each column for sachet (Lshape or Tshape) up and down
                            sachet_match_lenght=1
                            up=1
                            down=1
                            while row_study + down < self.N and self.board[row_study + down, col_study + i] == self.board[row_study, col_study + i]:
                                down += 1
                                sachet_match_lenght+=1
                            while row_study - up >= 0 and self.board[row_study - up, col_study + i] == self.board[row_study, col_study + i]:
                                up += 1
                                sachet_match_lenght+=1
                            if sachet_match_lenght>=3:
                                for u in range (up):
                                    already_visited.append((row_study - u, col_study + i))
                                    matches.append(((row_study - u, col_study + i)))
                                for d in range (down):
                                    already_visited.append((row_study + d, col_study + i))
                                    matches.append(((row_study + d, col_study + i)))
                                if (row_study, col_study + i) in recently_added:
                                    self.board[row_study, col_study + i].type='sachet'
                                    new_type.append((row_study, col_study + i))

                    
                    # Check vertical
                    match_length2 = 1
                    while row_study + match_length2 < self.N and self.board[row_study, col_study] == self.board[row_study + match_length2, col_study]:
                        match_length2 += 1
                        if match_length2==3:
                            already_visited.append((row_study + 1, col_study))
                        if match_length2>=3:
                            already_visited.append((row_study + match_length2, col_study))
                    if match_length2 >= 3:
                        one_already_transformed=False
                        if match_length2==4:
                            for i in range(match_length2):
                                if (row_study + i, col_study) in recently_added and not one_already_transformed and self.board[row_study + i, col_study].type=='normal':
                                    self.board[row_study + i, col_study].type='raye_hor'
                                    new_type.append((row_study + i, col_study))
                                    one_already_transformed=True
                                else:
                                    matches.append(((row_study + i, col_study)))
                        elif match_length2>=5:
                            for i in range(match_length2):
                                if (row_study + i, col_study) in recently_added and not one_already_transformed and self.board[row_study + i, col_study].type=='normal':
                                    self.board[row_study + i, col_study].type='disco'
                                    self.board[row_study + i, col_study].id=NUM_DISCO
                                    new_type.append((row_study + i, col_study))
                                    one_already_transformed=True
                                else:
                                    matches.append(((row_study + i, col_study)))

                        for i in range(match_length2):
                            matches.append(((row_study + i, col_study)))
                            #Check each row for sachet (Lshape or Tshape) only checking right
                            sachet_match_lenght=1
                            while col_study + sachet_match_lenght < self.M and self.board[row_study + i, col_study + sachet_match_lenght] == self.board[row_study + i, col_study]:
                                sachet_match_lenght += 1
                                if sachet_match_lenght==3:
                                    already_visited.append((row_study + i, col_study + 1))
                                if sachet_match_lenght>=3:
                                    already_visited.append((row_study + i, col_study + sachet_match_lenght))
                            if sachet_match_lenght>=3:
                                if (row_study + i, col_study) in recently_added:
                                    self.board[row_study + i, col_study].type='sachet'
                                    new_type.append((row_study + i, col_study))


        matches = set(matches)

        for i in new_type:
            if i in matches:
                matches.remove(i)

        return matches  # Return unique matches
    
    def is_ok_to_swap(self, row1, col1, row2, col2):
        """
        Check if the move is valid. Doesn't actually swap the candies, or update their type.
        """
        if abs(row1 - row2) > 1 or abs(col1 - col2) > 1:
            return False
        
        # Check for swap of two typed candies
        if self.board[row1, col1].type!='normal' and self.board[row2, col2].type!='normal':
            return True
        
        if self.board[row1, col1].type=='disco' or self.board[row2, col2].type=='disco':
            return True

        action=Action(self)
        action.raw_swap(row1, col1, row2, col2)
        row,col=[(row1, col1), (row2, col2)]
        
        # Check horizontal for each candy
        for (row, col) in [(row1, col1), (row2, col2)]:
            match_length = 1
            i=1
            while row + i < self.N and self.board[row, col] == self.board[row + i, col]:
                match_length += 1
                i+=1
                if match_length >= 3:
                    action.raw_swap(row1, col1, row2, col2)
                    return True
            i=1
            while row-i >= 0 and self.board[row, col] == self.board[row-i, col]:
                match_length += 1
                i+=1
                if match_length >= 3:
                    action.raw_swap(row1, col1, row2, col2)
                    return True

        # Check vertical for each candy
        for (row, col) in [(row1, col1), (row2, col2)]:
            match_length = 1
            i=1
            while col + i < self.M and self.board[row, col] == self.board[row, col + i]:
                match_length += 1
                i+=1
                if match_length >= 3:
                    action.raw_swap(row1, col1, row2, col2)
                    return True
            i=1
            while col-i >= 0 and self.board[row, col] == self.board[row, col-i]:
                match_length += 1
                i+=1
                if match_length >= 3:
                    action.raw_swap(row1, col1, row2, col2)
                    return True
                
        action.raw_swap(row1, col1, row2, col2)
        return False
        
    def remove_piece(self, row, col,force_remove=False):
        """
        Remove piece from the board, and performs action depending on the type of the candy.
        Force remove is to remove a typed piece that is performing an action to avoid infinite recursion.
        """
        if self.board[row, col].type=='empty':
            return 0
        if force_remove:
            self.board[row, col] = Candy(0,'empty')
            return 0
        if self.board[row, col].type=='normal':
            self.board[row, col] = Candy(0,'empty')
            return 40
        elif self.board[row, col].type=='raye_hor':
            self.remove_piece(row, col,force_remove=True)
            count=500
            for i in range(self.M):
                count+=self.remove_piece(row, i)
            return count
        elif self.board[row, col].type=='raye_ver':
            self.remove_piece(row, col,force_remove=True)
            count=500
            for i in range(self.N):
                count+=self.remove_piece(i, col)
            return count
        elif self.board[row, col].type=='sachet':
            ### A modifier pour double pop
            self.remove_piece(row, col,force_remove=True)
            count=1000  
            for i in range(max([0,row-1]),min([self.N,row+2])):
                for j in range(max([0,col-1]),min([self.M,col+2])):
                    count+=self.remove_piece(i, j)
            return count
        elif self.board[row, col].type=='disco':
            # Remove all candies of a random id
            self.remove_piece(row, col,force_remove=True)
            count=5000
            id_rand=numpy.random.randint(1, N_CANDY + 1)
            for i in range(self.N):
                for j in range(self.M):
                    if self.board[i, j].id==id_rand:
                        count+=self.remove_piece(i, j)
            return count


    def remove_matches(self,fall=[],from_move=[]):
        """
        Remove matches from the board.
        """
        matches = self.check_if_merged_all(fall=fall,from_move=from_move)
        if len(matches) == 0:
            return False
        for (row, col) in matches:
            self.score+=self.remove_piece(row, col)
        return len(matches)
    
    def update(self,from_move=[],step_by_step=False,fall=[]):
        """
        Update the board by removing matches and filling empty spaces.
        """
        updated = False
        from_move=from_move
        while self.remove_matches(fall=fall,from_move=from_move)!=False:
            if step_by_step:
                return True
            fall1=self.make_it_fall()
            fall2=self.fill_random()
            fall=fall1+fall2
            from_move=[]
            updated = True
        if len(self.get_legal_moves()) == 0:
            self.empty()
            self.fill_random()
            updated = True
        return updated
    
    def get_legal_moves(self):
        """
        Get all legal moves on the board.
        """
        legal_moves = []
        
        for row in range(self.N-1):
            for col in range(self.M-1):
                if self.is_ok_to_swap(row, col, row + 1, col):
                    legal_moves.append(((row, col), (row + 1, col)))
                if self.is_ok_to_swap(row, col, row, col + 1):
                    legal_moves.append(((row, col), (row, col + 1)))
        return legal_moves
        
    def scoring_function(self, match_length):
        """
        Define scoring based on match length.
        """
        return match_length # Complètemet arbitraire for now

    def display_move(self, move):
        """
        Display the board with the moved candies printed in red.
        """
        RED = '\033[91m'
        END = '\033[0m'
        
        for i in range(self.N):
            row_display = '|'
            for j in range(self.M):
                if (i, j) == move[0] or (i, j) == move[1]:
                    row_display += RED + str(self.board[i, j].id) + END + ' '
                else:
                    row_display += str(self.board[i, j].id) + ' '
            row_display = row_display.strip() + '|'
            print(row_display)
        print()

UnitVectors = {
    'UP': (-1, 0),
    'DOWN': (1, 0),
    'LEFT': (0, -1),
    'RIGHT': (0, 1)
}

class Action:
    def __init__(self, board: Board):
        """
        Initialize the Action class with a board.
        """
        self.board = board

    def raw_swap(self, row1, col1, row2, col2):
        """
        Swap two candies on the board. This does not check for valid moves.
        """
        if abs(row1 - row2) > 1 or abs(col1 - col2) > 1:
            raise ValueError("Can only swap adjacent candies.")
        self.board.board[row1, col1], self.board.board[row2, col2] = self.board.board[row2, col2], self.board.board[row1, col1]
            
    def swap(self, row1, col1, row2, col2, step_by_step=False):
        """
        Swap two candies on the board if the move is valid.
        """
        if self.board.is_ok_to_swap(row1, col1, row2, col2):
            self.raw_swap(row1, col1, row2, col2)
            self.board.update(from_move=[(row1, col1), (row2, col2)],step_by_step=step_by_step)
            return True
        return False
        
   
def read_board_from_file(file_path):
    """
    Read a board from a text file and return the corresponding Board object.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extract board dimensions from the first line
    dimensions_and_score = lines[0].strip().split(' ')
    N = int(dimensions_and_score[0])
    M = int(dimensions_and_score[1])
    score = int(dimensions_and_score[2])
    # Initialize the board
    board = Board(N, M)
    board.score = score

    # Read the board state from the file
    for i in range(1, N + 1):
        line = lines[i].strip().split(' ')
        for j in range(M):
            candy_id, candy_type = line[j].split('_',1)
            board.board[i - 1, j] = Candy(int(candy_id), candy_type)

    return board


### MCTS PARAM FOR VIZ
EXPLORATION_PARAM = 3000
N_ROLLOUT = 3
N_SIMULATION = 1000
N_RANDOM = 1

class Viz:
    def __init__(self, board: Board, action: Action, is_colloc: bool = False, N_RANDOM = N_RANDOM, 
                 EXPLORATION_PARAM = EXPLORATION_PARAM, N_ROLLOUT = N_ROLLOUT, N_SIMULATION = N_SIMULATION):
        """
        Initialize the Viz class with a board and an action.
        """
        self.board = board
        self.action = action
        self.is_colloc = is_colloc
        self.N_RANDOM = N_RANDOM
        self.EXPLORATION_PARAM = EXPLORATION_PARAM
        self.N_ROLLOUT = N_ROLLOUT
        self.N_SIMULATION = N_SIMULATION

    async def Visualize(self):
        """
        Display the current state of the board using asyncio.
        """
        pygame.init()

        screenwidth = 600
        screenheight = 600
        menu_width = 300
        self.screenwidth = screenwidth + menu_width
        self.screenheight = screenheight + 50
        time_delay = 100
        win = pygame.display.set_mode((self.screenwidth, self.screenheight))
        pygame.display.set_caption("Candy Crush (official version)")

        x_cases = numpy.linspace(screenwidth / (2 * self.board.M), screenwidth - screenwidth / (2 * self.board.M), self.board.M)
        y_cases = numpy.linspace(screenheight / (2 * self.board.N), screenheight - screenheight / (2 * self.board.N), self.board.N)

        width = screenwidth / self.board.M / 2
        height = screenheight / self.board.N / 2

        # Load candy images
        candy_images = []
        for i in range(1, 8):
            if self.is_colloc:
                image = pygame.image.load(f'assets/colloc/candy_{i}.png')
            else:
                image = pygame.image.load(f'assets/candy/candy_{i}.png')
            image = pygame.transform.scale(image, (int(width), int(height)))
            candy_images.append(image)

        run = True
        i_clicked = -1 
        j_clicked = -1
        clicked = False
        highlight_move = False
        best_move = None
        mcts = None
        board_copy = self.board.copy()
        all_move = None
        visible_menu = True
        slot_save = None
        self.board.score = 0
        mcts_mode = True

        x_mcts_mode = 0
        y_mcts_mode = screenheight + 30
        slider_simu = Slider(win, x_mcts_mode+180, y_mcts_mode, 180, 10, min=300, max=3000, step=100, initial=self.N_SIMULATION)
        slider_simu_output = TextBox(win, x_mcts_mode+180+5, y_mcts_mode-30, 175, 20, fontSize=12)
        slider_simu_output.disable()
        slider_explo = Slider(win, x_mcts_mode+380, y_mcts_mode, 180, 10, min=500, max=5000, step=500, initial=self.EXPLORATION_PARAM)
        slider_explo_output = TextBox(win, x_mcts_mode+380+5, y_mcts_mode-30, 175, 20, fontSize=12)
        slider_explo_output.disable()

        while run:
            await asyncio.sleep(0)  # Allow other tasks to run
            
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    run = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_q]:
                run = False

            if keys[pygame.K_w]:
                await asyncio.sleep(1)  # Replace pygame.time.delay
                time_delay = 100 if time_delay == 500 else 500

            if keys[pygame.K_m]:
                if mcts_mode:
                    await asyncio.sleep(0.1)  # Replace pygame.time.delay
                    mcts_mode = False
                    self.screenheight = screenheight
                else:
                    await asyncio.sleep(0.1)  # Replace pygame.time.delay
                    mcts_mode = True
                    self.screenheight = screenheight+50
                win = pygame.display.set_mode((self.screenwidth, self.screenheight))

            if mcts_mode:
                if keys[pygame.K_p]:
                    clicked = False
                    mcts = MCTS_CandyCrush_Complex(self.board, exploration_param=self.EXPLORATION_PARAM, 
                                                 N_rollout=self.N_ROLLOUT, n_simulation=self.N_SIMULATION, 
                                                 no_log=True, write_log_file=False)
                    best_move, all_move = await mcts.best_move(return_all=True, N_random=self.N_RANDOM)
                    highlight_move = True
                slider1 = slider_simu.getValue()
                self.N_SIMULATION = int(slider1)
                slider_simu_output.setText('Number of simulations:  '+str(self.N_SIMULATION))
                slider2 = slider_explo.getValue()
                self.EXPLORATION_PARAM = int(slider2)
                slider_explo_output.setText('Exploration parameter:  '+str(self.EXPLORATION_PARAM))

            if keys[pygame.K_c]:
                if slot_save is not None:
                    await self.save_board_to_file(f'copied_board_{slot_save}.txt')
                else:
                    await self.save_board_to_file('copied_board.txt')
            
            if keys[pygame.K_d]:
                await asyncio.sleep(0.1)  # Replace pygame.time.delay
                if slot_save is not None:
                    slot_save += 1
                else:
                    slot_save = 0
                if slot_save > 9:
                    slot_save = 0
            
            if keys[pygame.K_v]:
                if slot_save is None:
                    board = read_board_from_file('copied_board.txt')
                else:
                    board = read_board_from_file(f'copied_board_{slot_save}.txt')
                self.board = board
                self.action = Action(self.board)
                board_copy = self.board.copy()
                clicked = False
                highlight_move = False
                all_move = None

            if keys[pygame.K_u]:
                self.board.update()

            if keys[pygame.K_ESCAPE]:
                await asyncio.sleep(0.2)  # Replace pygame.time.delay
                if self.screenwidth == screenwidth+menu_width:
                    self.screenwidth = screenwidth
                    visible_menu = False
                else:
                    self.screenwidth = screenwidth+menu_width
                    visible_menu = True
                win = pygame.display.set_mode((self.screenwidth, self.screenheight))

            # Handle candy selection and movement
            if clicked:
                for i in range(N_CANDY):
                    if keys[pygame.K_1 + i] or keys[pygame.K_KP1 + i]:
                        self.board.board[i_clicked, j_clicked] = Candy(i + 1, 'normal')
                        break
                if keys[pygame.K_1 + 6]:
                    self.board.board[i_clicked, j_clicked] = Candy(7, 'disco')
                if keys[pygame.K_1 + 7]:
                    self.board.board[i_clicked, j_clicked] = Candy(numpy.random.randint(1, N_CANDY), 'sachet')
                if keys[pygame.K_1 + 8]:
                    self.board.board[i_clicked, j_clicked] = Candy(numpy.random.randint(1, N_CANDY), 'raye_hor')
                if keys[pygame.K_0]:
                    self.board.board[i_clicked, j_clicked] = Candy(numpy.random.randint(1, N_CANDY), 'raye_ver')
                
                all_move = None

            # Handle mouse clicks
            if pygame.mouse.get_pressed()[0]:
                x, y = pygame.mouse.get_pos()
                for i in range(self.board.N):
                    for j in range(self.board.M):
                        if (x_cases[j] - width / 2 < x < x_cases[j] + width / 2 and 
                            y_cases[i] - height / 2 < y < y_cases[i] + height / 2):
                            i_clicked = i
                            j_clicked = j
                            clicked = True
                            break
            
            display_action = False
            if clicked and keys[pygame.K_UP]:
                if i_clicked - 1 >= 0:
                    board_copy = self.board.copy()
                    self.action.swap(i_clicked, j_clicked, i_clicked - 1, j_clicked, step_by_step=True)
                    clicked = False
                    display_action = True
                    highlight_move = False
                    all_move = None

            if clicked and keys[pygame.K_DOWN]:
                if i_clicked + 1 < self.board.N:
                    board_copy = self.board.copy()
                    self.action.swap(i_clicked, j_clicked, i_clicked + 1, j_clicked, step_by_step=True)
                    clicked = False
                    display_action = True
                    highlight_move = False
                    all_move = None

            if clicked and keys[pygame.K_LEFT]:
                if j_clicked - 1 >= 0:
                    board_copy = self.board.copy()
                    self.action.swap(i_clicked, j_clicked, i_clicked, j_clicked - 1, step_by_step=True)
                    clicked = False
                    display_action = True
                    highlight_move = False
                    all_move = None

            if clicked and keys[pygame.K_RIGHT]:
                if j_clicked + 1 < self.board.M:
                    board_copy = self.board.copy()
                    self.action.swap(i_clicked, j_clicked, i_clicked, j_clicked + 1, step_by_step=True)
                    clicked = False
                    display_action = True
                    highlight_move = False
                    all_move = None

            if not display_action:
                self.board_visual(candy_images, win, x_cases, width, y_cases, height, clicked, 
                                i_clicked, j_clicked, highlight_move, best_move, all_move, 
                                visible_menu, time_delay, save_slot=slot_save, mcts_mode=mcts_mode)
                pygame.display.update()

            while display_action:
                self.board_visual(candy_images, win, x_cases, width, y_cases, height, 
                                visible_menu=visible_menu, time_delay=time_delay)
                pygame.display.update()
                await asyncio.sleep(time_delay / 1000)  # Convert to seconds
                fall1 = self.board.make_it_fall()
                fall2 = self.board.fill_random()
                fall = fall1 + fall2
                self.board_visual(candy_images, win, x_cases, width, y_cases, height, 
                                visible_menu=visible_menu, time_delay=time_delay)
                pygame.display.update()
                await asyncio.sleep(time_delay / 1000)  # Convert to seconds
                display_action = self.board.update(fall=fall, step_by_step=True)

            if keys[pygame.K_s]:
                clicked = False
                self.board.empty()
                self.board.fill_random()
                self.board.update()
                self.board.score = 0

            if keys[pygame.K_r]:
                clicked = False
                self.board = board_copy
                self.action = Action(self.board)

            pygame.display.update()
            pygame_widgets.update(events)

        pygame.display.quit()
        pygame.quit()
        sys.exit()

    async def save_board_to_file(self, file_path):
        """
        Save the current board to a text file asynchronously.
        """
        with open(file_path, 'w') as file:
            file.write(f"{self.board.N} {self.board.M} {self.board.score}\n")
            for i in range(self.board.N):
                row = []
                for j in range(self.board.M):
                    candy = self.board.board[i, j]
                    row.append(f"{candy.id}_{candy.type}")
                file.write(' '.join(row) + '\n')
            await asyncio.sleep(0)  # Allow other tasks to run


    def board_visual(self,candy_images,win,x_cases,width,y_cases,height,clicked=False,i_clicked=0,j_clicked=0,highlight_move=False,best_move=None, all_move = None, visible_menu = False, time_delay=None, save_slot = None,mcts_mode = False):

        win.fill((0, 0, 0))
        win.blit(pygame.image.load('assets/background/image.png'), (0, 0))
        if highlight_move:
            i1, j1 = best_move[0]
            i2, j2 = best_move[1]
            pygame.draw.rect(win, (0, 255, 0), (x_cases[j1] - width / 2 - 4, y_cases[i1] - height / 2 - 4, width + 8, height + 8))
            pygame.draw.rect(win, (0, 255, 0), (x_cases[j2] - width / 2 - 4, y_cases[i2] - height / 2 - 4, width + 8, height + 8))

        for i in range(self.board.N):
            for j in range(self.board.M):
                if self.board.board[i, j] != Candy(0,'empty'):
                    candy_index = int(self.board.board[i, j].id) - 1
                    candy_type = self.board.board[i, j].type
                    if clicked and i == i_clicked and j == j_clicked:
                        pygame.draw.rect(win, (255, 255, 255), (x_cases[j] - width / 2 - 2, y_cases[i] - height / 2 - 2, width + 4, height + 4))
                    win.blit(candy_images[candy_index], (x_cases[j] - width / 2, y_cases[i] - height / 2))
                    # Add horizontal stripes for raye_hor,raye_ver and sachet typed candies
                    if candy_type == 'raye_hor':
                        pygame.draw.line(win, (255, 255, 255), (x_cases[j] - width / 2, y_cases[i]), (x_cases[j] + width / 2, y_cases[i]), 2)
                    if candy_type=='raye_ver':
                        pygame.draw.line(win, (255, 255, 255), (x_cases[j], y_cases[i] - height / 2), (x_cases[j], y_cases[i] + height / 2), 2)
                    if candy_type=='sachet':
                        pygame.draw.rect(win, (255, 255, 255), (x_cases[j] - width / 4, y_cases[i] - height / 4, width / 2, height / 2), 2)
                else:
                    pygame.draw.circle(win, (250, 0, 0), (x_cases[j], y_cases[i]), 20)

        if all_move is not None:    
            for move, visits, mean_reward in all_move:
                i1, j1 = move[0]
                i2, j2 = move[1]
                # Write as a text the number of visits and the mean reward
                font = pygame.font.Font(None, 18)
                text1 = font.render(f"N: {visits}", True, (255, 255, 255))
                middle_x = (x_cases[j1] + x_cases[j2]) / 2
                middle_y = (y_cases[i1] + y_cases[i2]) / 2
                win.blit(text1, (middle_x-20, middle_y - 10))
                text2 = font.render(f"Q: {mean_reward/100:.1f}", True, (255, 255, 255))
                win.blit(text2, (middle_x-20, middle_y + 10))
        
        if visible_menu:
            x_menu = self.screenwidth - 300
            y_menu = 20
            font = pygame.font.Font(None, 24)
            menu_text_1 = font.render(f"Speed (change with W)", True, (255, 255, 255))
            win.blit(menu_text_1, (x_menu, y_menu))
            menu_text_2 = font.render(f"{time_delay}", True, (255, 255, 255))
            win.blit(menu_text_2, (x_menu, y_menu+30))
            menu_text_2 = font.render(f"Copy/Save Slot : {save_slot}", True, (255, 255, 255))
            win.blit(menu_text_2, (x_menu, y_menu+60))

            shortcut_text = font.render(f"Shortcuts:", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+90))
            shortcut_text = font.render(f"U: Update the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+120))
            shortcut_text = font.render(f"S: Empty the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+150))
            shortcut_text = font.render(f"R: Return to previous board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+180))
            shortcut_text = font.render(f"C: Copy the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+210))
            shortcut_text = font.render(f"V: Paste the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+240))
            shortcut_text = font.render(f"Click: Click the candy to swap", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+270))
            shortcut_text = font.render(f"Arrows: Move the candy", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+300))
            shortcut_text = font.render(f"1-9: Change the Candy type", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+330))
            shortcut_text = font.render(f"On Browser, the MCTS might be slow", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+500))
            shortcut_text = font.render(f"Time for 300 simulations: 2s", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+530))
            shortcut_text = font.render(f"Time for 3000 simulations: 20s", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+560))


        if mcts_mode:
            events = pygame.event.get()
            pygame_widgets.update(events)
            font = pygame.font.Font(None, 24)
            shortcut_text = font.render(f"P: Run the MCTS", True, (255, 255, 255))
            win.blit(shortcut_text, (20, self.screenheight-40))


        # Display the score

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.board.score}", True, (255, 255, 255))
        win.blit(text, (10, 10))

import platform
async def main():
    # Initialize the game
    b = Board(7, 7)
    a = Action(b)
    b.fill_random()
    b.update()
    b.score = 0
    v = Viz(b, a)
    
    # Start the visualization
    await v.Visualize()

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        # For pygbag/web deployment
        asyncio.run(main())
    else:
        # For local Python environment
        import sys
        if sys.version_info >= (3, 11):
            # Python 3.11+ preferred way
            asyncio.run(main())
        else:
            # Older Python versions
            loop = asyncio.get_event_loop()
            loop.run_until_complete(main())