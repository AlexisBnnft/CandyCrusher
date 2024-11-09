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
        viz_board = np.array([[str(self.board[i, j]) for j in range(self.M)] for i in range(self.N)])
        for row in viz_board:
            print('|' + ' '.join(row) + '|')
        print()

    def display_with_type(self):
        """
        Display the current state of the board.
        """
        viz_board = np.array([[str(self.board[i, j]) + "_"+ str(TYPE_DISPLAY[self.board[i, j].type]) for j in range(self.M)] for i in range(self.N)])
        for row in viz_board:
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
                    self.board[row, col] = Candy(np.random.randint(1, N_CANDY + 1))
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
                self.board[row1, col1].id=np.random.randint(1, N_CANDY + 1)
                self.board[row1, col1].type='normal'
                matches.append((row1, col1))
                return matches
            
            if self.board[row1, col1].type=='normal' and self.board[row2, col2].type=='disco':
                for i in range(self.N):
                    for j in range(self.M):
                        if self.board[i, j].id==self.board[row1, col1].id:
                            matches.append((i, j))
                self.board[row2, col2].id=np.random.randint(1, N_CANDY + 1)
                self.board[row2, col2].type='normal'
                matches.append((row2, col2))
                return matches


            if self.board[row1, col1].type!='normal' and self.board[row2, col2].type!='normal':
                
                if self.board[row1, col1].type=='disco' and self.board[row2, col2].type=='disco':
                    for i in range(self.N):
                        for j in range(self.M):
                            matches.append((i, j))
                    self.board[row1, col1].id=np.random.randint(1, N_CANDY + 1)
                    self.board[row1, col1].type='normal'
                    self.board[row2, col2].id=np.random.randint(1, N_CANDY + 1)
                    self.board[row2, col2].type='normal'
                    return matches

                elif self.board[row1, col1].type=='disco' and self.board[row2, col2].type in {'raye_ver','raye_hor','sachet'}:
                    for i in range(self.N):
                        for j in range(self.M):
                            if self.board[i, j].id==self.board[row2, col2].id:
                                self.board[i, j].type=self.board[row2, col2].type
                                matches.append((i, j))
                    self.board[row1, col1].id=np.random.randint(1, N_CANDY + 1)
                    self.board[row1, col1].type='normal'
                    matches.append((row1, col1))
                    return matches
                elif self.board[row1, col1].type in {'raye_ver','raye_hor','sachet'} and self.board[row2, col2].type=='disco':
                    for i in range(self.N):
                        for j in range(self.M):
                            if self.board[i, j].id==self.board[row1, col1].id:
                                self.board[i, j].type=self.board[row1, col1].type
                                matches.append((i, j))
                    self.board[row2, col2].id=np.random.randint(1, N_CANDY + 1)
                    self.board[row2, col2].type='normal'
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
                    for i in range(max([0,row2-1]),min([self.N,row2+2])):
                        for j in range(max([0,col2-1]),min([self.M,col2+2])):
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
                        for j in range(self.M):
                            matches.append((i,j))
                    for i in range(self.N):
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
            return 1
        elif self.board[row, col].type=='raye_hor':
            self.remove_piece(row, col,force_remove=True)
            count=1
            for i in range(self.M):
                count+=self.remove_piece(row, i)
            return count
        elif self.board[row, col].type=='raye_ver':
            self.remove_piece(row, col,force_remove=True)
            count=1
            for i in range(self.N):
                count+=self.remove_piece(i, col)
            return count
        elif self.board[row, col].type=='sachet':
            ### A modifier pour double pop
            self.remove_piece(row, col,force_remove=True)
            count=1
            
            for i in range(max([0,row-1]),min([self.N,row+2])):
                for j in range(max([0,col-1]),min([self.M,col+2])):
                    count+=self.remove_piece(i, j)
            return count
        elif self.board[row, col].type=='disco':
            # Remove all candies of a random id
            self.remove_piece(row, col,force_remove=True)
            count=1
            id_rand=np.random.randint(1, N_CANDY + 1)
            for i in range(self.N):
                for j in range(self.M):
                    if self.board[i, j].id==id_rand:
                        count+=self.remove_piece(i, j)


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
        
        
