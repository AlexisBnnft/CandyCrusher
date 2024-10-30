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
        self.board = np.array([[' ' for _ in range(M)] for _ in range(N)])

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
        self.board = np.array([[' ' for _ in range(self.M)] for _ in range(self.N)])

    def is_full(self):
        """
        Check if the board is full (no empty spaces).
        """
        return np.all(self.board != ' ')
    
    def no_moves(self):
        """
        Check if there are any possible moves left.
        """
        ## Import actions? double import?
    
    def fill_random(self):
        """
        Fill empty spaces in the board with random candies.
        """
        for row in range(self.N):
            full_row=True
            for col in range(self.M):
                if self.board[row, col] == ' ':
                    self.board[row, col] = Candy(np.random.randint(1, N_CANDY + 1))
                    full_row=False
            if full_row:  # Si la ligne est pleine, normalement les rows en dessous sont pleines aussi
                break
    
    def add_piece(self, candy : Candy, row, col):
        """
        Add a candy to the board at the specified position.
        """
        if self.board[row, col] == ' ':
            self.board[row, col] = candy
            return True
        return False
    
    def make_it_fall(self):
        """
        Make candies fall down to fill empty spaces.
        """
        for col in range(self.M):
            for row in range(self.N-1, -1, -1):
                if self.board[row, col] == ' ':
                    for r in range(row-1, -1, -1):
                        if self.board[r, col] != ' ':
                            self.board[row, col] = self.board[r, col]
                            self.board[r, col] = ' '
                            break

    def check_if_merged_all(self):
        """
        Check if there are any matches on the entire board. This may be computational heavy in larger boards.
        """
        matches = []

        # Check for matches in both directions for each cell
        for row in range(self.N):
            for col in range(self.M):
                if self.board[row, col] != ' ':
                    # Check horizontal
                    match_length = 1
                    while col + match_length < self.M and self.board[row, col] == self.board[row, col + match_length]:
                        match_length += 1
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

    def remove_matches(self):
        """
        Remove matches from the board.
        """
        matches = self.check_if_merged_all()
        if len(matches) == 0:
            return False
        for (row, col) in matches:
            self.remove_piece(row, col)
        self.make_it_fall() # Make it rain baby
        return len(matches)
    
    def update(self):
        """
        Update the board by removing matches and filling empty spaces.
        """
        updated = False
        while self.remove_matches():
            self.fill_random() #Modifié pour fill que les cases vides
            updated = True
        return updated
    
    def get_legal_moves(self):
        """
        Get all legal moves on the board.
        """
        legal_moves = []
        
        for row in range(self.N - 1):
            for col in range(self.M - 1):
                if Action(self).swap(row, col, row + 1, col, check_only=True):
                    legal_moves.append(((row, col), (row, col + 1)))
                if Action(self).swap(row, col, row, col + 1, check_only=True):
                    legal_moves.append(((row, col), (row + 1, col)))
        return legal_moves
        




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

    def raw_swap(self, row1, col1, row2, col2): # j'ai mis row1, col1, row2, col2 mais je pense qu'on peut faire position1, position2 peut être
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
        ok_to_swap = False
        candy1 = self.board.board[row1, col1]
        candy2 = self.board.board[row2, col2]
        self.raw_swap(row1, col1, row2, col2)
        for direction in ['UP', 'LEFT']:  # check ligne verticale et ligne horizontale pour les 2

            align_max1=1
            current_row1, current_col1 = row2+UnitVectors[direction][0], col2+UnitVectors[direction][1]
            while current_row1 >= 0 and current_col1 >= 0 and self.board.board[current_row1, current_col1] == candy1:
                align_max1+=1
                current_row1 += UnitVectors[direction][0]
                current_col1 += UnitVectors[direction][1]
            current_row1, current_col1 = row2-UnitVectors[direction][0], col2-UnitVectors[direction][1]
            while current_row1 < self.board.N and current_col1 < self.board.M and self.board.board[current_row1, current_col1] == candy1:
                align_max1+=1
                current_row1 -= UnitVectors[direction][0]
                current_col1 -= UnitVectors[direction][1]

            if align_max1 >= 3:
                ok_to_swap = True
                break

            align_max2=1
            current_row2, current_col2 = row1+UnitVectors[direction][0], col1+UnitVectors[direction][1]
            while current_row2 >= 0 and current_col2 >= 0 and self.board.board[current_row2, current_col2] == candy2:
                align_max2+=1
                current_row2 += UnitVectors[direction][0]
                current_col2 += UnitVectors[direction][1]
            current_row2, current_col2 = row1-UnitVectors[direction][0], col1-UnitVectors[direction][1]
            while current_row2 < self.board.N and current_col2 < self.board.M and self.board.board[current_row2, current_col2] == candy2:
                align_max2+=1
                current_row2 -= UnitVectors[direction][0]
                current_col2 -= UnitVectors[direction][1]
            
            
            if align_max2 >= 3:
                ok_to_swap = True
                break
            
        if not ok_to_swap:
            self.raw_swap(row1, col1, row2, col2)
            return False
        if check_only:
            self.raw_swap(row1, col1, row2, col2)
        return True
        
        
