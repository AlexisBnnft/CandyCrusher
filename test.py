import unittest
from board import Board, Action
from candy import Candy, N_CANDY
from copy import deepcopy
import numpy as np


class TestBoard(unittest.TestCase):

    def initialize_normal_board(self):
        # Initialize this board
        # |5 5 3 1 5 5|
        # |3 6 1 3 3 5|
        # |1 1 3 4 1 6|
        # |3 4 3 1 5 5|
        # |3 1 1 2 1 2|
        np.random.seed(0)   
        board = Board(5,6)
        board.board = np.array([
                [Candy(5, "normal"), Candy(5, "normal"), Candy(3, "normal"), Candy(1, "normal"), Candy(5, "normal"), Candy(5, "normal")],
                [Candy(3, "normal"), Candy(6, "normal"), Candy(1, "normal"), Candy(3, "normal"), Candy(3, "normal"), Candy(5, "normal")],
                [Candy(1, "normal"), Candy(1, "normal"), Candy(3, "normal"), Candy(4, "normal"), Candy(1, "normal"), Candy(6, "normal")],
                [Candy(3, "normal"), Candy(4, "normal"), Candy(3, "normal"), Candy(1, "normal"), Candy(5, "normal"), Candy(5, "normal")],
                [Candy(3, "normal"), Candy(1, "normal"), Candy(1, "normal"), Candy(2, "normal"), Candy(1, "normal"), Candy(2, "normal")]
            ])
        return board

    def initialize_normal_board_2nd(self):
        # Initialize this board
        # |3 5 3 6 5 5|
        # |3 6 6 3 6 6|
        # |1 3 3 6 1 6|
        # |3 4 3 1 5 5|
        # |3 1 1 2 1 1|
        np.random.seed(0)   
        board = Board(5,6)
        board.board = np.array([
                [Candy(3, "normal"), Candy(5, "normal"), Candy(3, "normal"), Candy(6, "normal"), Candy(5, "normal"), Candy(5, "normal")],
                [Candy(3, "normal"), Candy(6, "normal"), Candy(6, "normal"), Candy(3, "normal"), Candy(6, "normal"), Candy(6, "normal")],
                [Candy(1, "normal"), Candy(3, "normal"), Candy(3, "normal"), Candy(6, "normal"), Candy(1, "normal"), Candy(6, "normal")],
                [Candy(3, "normal"), Candy(4, "normal"), Candy(3, "normal"), Candy(1, "normal"), Candy(5, "normal"), Candy(5, "normal")],
                [Candy(3, "normal"), Candy(1, "normal"), Candy(1, "normal"), Candy(2, "normal"), Candy(1, "normal"), Candy(1, "normal")]
            ])
        return board

    def test_make_it_fall(self):
        board = self.initialize_normal_board()
        board_copy = deepcopy(board)
        board.make_it_fall()
        assert board == board_copy

        # Test where Candy should fall
        board.board[4,0] = Candy(0, "empty")
        board.board[3,0] = Candy(0, "empty")
        board.board[2,0] = Candy(0, "empty")
        board.make_it_fall()
        # Did the candies fall?
        assert board.board[4,0] != Candy(0, "empty")
        assert board.board[3,0] != Candy(0, "empty")
        # Did the new candies appear?
        board.fill_random()
        assert board.board[0,0] != Candy(0, "empty")
        assert board.board[1,0] != Candy(0, "empty")
        assert board.board[2,0] != Candy(0, "empty")
    
    def test_fill_random(self):
        board = Board(50,50)
        board.fill_random()
        # Check if all the candies appeared
        assert len(np.unique([[board.board[i,j].id] for i in range(board.board.shape[0]) for j in range(board.board.shape[1])])) == N_CANDY

    def test_simple_4_hor_creation_step_by_step(self):
        board = self.initialize_normal_board()
        Action(board).swap(1,2,1,3,step_by_step=True)
        assert board.board[1,2].type == "raye_hor"
        board.make_it_fall()
        board.fill_random()
        board.update()
        assert board.board[3,2].type == "raye_hor"
    
    def test_simple_4_hor_creation(self):
        board = self.initialize_normal_board()
        Action(board).swap(1,2,1,3)
        assert board.board[3,2].type == "raye_hor"
        # Check that every other candy is normal
        for i in range(board.board.shape[0]):
            for j in range(board.board.shape[1]):
                if i == 3 and j == 2:
                    continue
                assert board.board[i,j].type == "normal"

    def test_simple_4_vert_creation_step_by_step(self):
        board = self.initialize_normal_board()
        Action(board).swap(4,3,3,3,step_by_step=True)
        assert board.board[4,3].type == "raye_ver"
        board.make_it_fall()
        board.fill_random()
        board.update()
        assert board.board[4,3].type == "raye_ver"

    def test_simple_4_vert_creation(self):
        board = self.initialize_normal_board()
        Action(board).swap(4,3,3,3)
        assert board.board[4,3].type == "raye_ver"
        for i in range(board.board.shape[0]):
            for j in range(board.board.shape[1]):
                if i == 4 and j == 3:
                    continue
                assert board.board[i,j].type == "normal"
    
    def test_simple_sachet_creation_step_by_step_L_shape(self):
        board = self.initialize_normal_board()
        Action(board).swap(0,2,1,2,step_by_step=True)
        assert board.board[1,2].type == "sachet"
        board.make_it_fall()
        board.fill_random()
        board.update()
        assert board.board[3,2].type == "sachet"

    def test_simple_sachet_creation_L_shape(self):
        board = self.initialize_normal_board()
        Action(board).swap(0,2,1,2)
        assert board.board[3,2].type == "sachet"
        for i in range(board.board.shape[0]):
            for j in range(board.board.shape[1]):
                if i == 3 and j == 2:
                    continue
                assert board.board[i,j].type == "normal"

    def test_simple_sachet_creation_step_by_step_T_shape(self):
        board = self.initialize_normal_board_2nd()
        Action(board).swap(1,4,1,3,step_by_step=True)
        assert board.board[1,3].type == "sachet"
        board.make_it_fall()
        board.fill_random()
        board.update()
        assert board.board[2,3].type == "sachet"

    def test_simple_sachet_creation_T_shape(self):
        board = self.initialize_normal_board_2nd()
        Action(board).swap(1,4,1,3)
        assert board.board[2,3].type == "sachet"
    
    def test_simple_disco_horiz(self):
        board = self.initialize_normal_board_2nd()
        Action(board).swap(3,3,4,3)
        assert board.board[4,3].type == "disco"

        # It should have created another disco
        disco_count = 0
        for i in range(board.board.shape[0]):
            for j in range(board.board.shape[1]):
                if board.board[i,j].type == "disco":
                    disco_count += 1
        assert disco_count == 2

    def test_simple_disco_vert(self):
        board = self.initialize_normal_board_2nd()
        Action(board).swap(2,0,2,1)
        assert board.board[4,0].type == "disco"


    
if __name__ == '__main__':
    unittest.main()