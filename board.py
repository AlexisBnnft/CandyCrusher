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

    def is_full(self):
        """
        Check if the board is full (no empty spaces).
        """
        return np.all(self.board != ' ')
    
    def fill_random(self):
        """
        Fill the board with random candies.
        """
        for row in range(self.N):
            for col in range(self.M):
                self.board[row, col] = Candy(np.random.randint(1, N_CANDY + 1))
    
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


