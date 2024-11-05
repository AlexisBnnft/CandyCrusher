import numpy as np
from candy import *


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
        self.board = np.array([[Candy(0,'empty') for _ in range(M)] for _ in range(N)])
        self.score = 0
    
    def __eq__(self, value: object) -> bool:
        """
        Check if two boards are equal.
        """
        return np.all(self.board == value.board)

    def display(self):
        """
        Display the current state of the board.
        """
        for row in self.board:
            print('|' + ' '.join(row) + '|')
        print()


    def empty(self):
        """
        Empty the board.
        """
        self.board = np.array([[Candy(0,'empty') for _ in range(self.M)] for _ in range(self.N)])

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
        return np.all(self.board != Candy(0,'empty'))
    
    def fill_random(self):
        """
        Fill empty spaces in the board with random candies.
        """
        recently_added=[]
        for row in range(self.N):
            for col in range(self.M):
                if self.board[row, col] == Candy(0,'empty'):
                    self.board[row, col] = Candy(np.random.randint(1, N_CANDY + 1))
                    full_row=False
            if full_row:  # Si la ligne est pleine, normalement les rows en dessous sont pleines aussi
                break
    
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
                            self.board[r, col] = ' '
                            break
        return fall

    def check_if_merged_all(self,fall=[],from_move=[]):
        """
        Attention: this may change stuff in the board.
        Check if there are any matches on the entire board. The candies in recently_added might change type.
        """
        matches = []

        # Check for matches in both directions for each cell
        for row in range(self.N):
            for col in range(self.M):
                if self.board[row, col] != ' ':
                    # Check horizontal
                    match_length = 1
                    while col_study + match_length < self.M and self.board[row_study, col_study] == self.board[row_study, col_study + match_length]:
                        match_length += 1
                        if match_length==3:
                            already_visited.append((row_study, col_study + 1))
                        if match_length>=3:
                            already_visited.append((row_study, col_study + match_length))
                    if match_length >= 3:
                        for i in range(match_length):
                            matches.append(((row, col + i)))
                    
                    # Check vertical
                    match_length = 1
                    while row + match_length < self.N and self.board[row, col] == self.board[row + match_length, col]:
                        match_length += 1
                    if match_length >= 3:
                        for i in range(match_length):
                            matches.append(((row + i, col)))
        return set(matches)  # Return unique matches
        
    def remove_piece(self, row, col):
        """
        Remove piece from the board.
        """
        self.board[row, col] = ' '

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
    
    def state(self):
        """
        Get the state of the board as a unique integer hash.
        
        Returns:
            int: A unique integer representing the board state.
        """
        # Flatten the board and concatenate each cell's value into a single string
        flattened = ''.join(str(cell.id) + str(cell.type) for row in self.board for cell in row)
        return int(flattened)
    
    def update(self):
        """
        Update the board by removing matches and filling empty spaces.
        """
        updated = False
        while self.remove_matches():
            self.fill_random() #Modifié pour fill que les cases vides
            updated = True
        return updated
    
    def from_state(state, rows, cols):
        """
        Convert a unique integer hash back to the board state.
        
        Args:
            state (int): The unique integer representing the board state.
            rows (int): The number of rows in the board.
            cols (int): The number of columns in the board.
        
        Returns:
            list: The board represented as a list of lists.
        """
        state_str = str(state)
        board = []
        for i in range(rows):
            row = [int(state_str[j]) for j in range(i * cols, (i + 1) * cols)]
            board.append(row)
        return board


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
        
    def swap(self, row1, col1, row2, col2, check_only=False):
        """
        Swap two candies on the board if the move is valid.
        """
        if self.board.is_ok_to_swap(row1, col1, row2, col2):
            self.raw_swap(row1, col1, row2, col2)
            self.board.update(from_move=[(row1, col1), (row2, col2)],step_by_step=step_by_step)
            return True
        return False
        
        
