from board import Board
from candy import Candy

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
        
    def swap(self, row1, col1, row2, col2):
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
        return True
        

        