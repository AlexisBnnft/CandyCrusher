import asyncio
import pygame
import platform
import numpy



N_CANDY = 6 # Number of different candies
NUM_DISCO = 7
TYPES = {'normal','raye_hor','raye_ver','sachet','disco','empty'}
TYPE_TO_ID = {'normal':0,'raye_hor':1,'raye_ver':2,'sachet': 3,'disco':4,'empty':5}
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

class Viz:
    def __init__(self, board: Board, action: Action, is_colloc: bool = False):
        """Initialize the Viz class with a board and an action."""
        self.board = board
        self.action = action
        self.is_colloc = is_colloc
        self.running = True
        
        # Initialize display variables
        self.screenwidth = 800
        self.screenheight = 800
        self.time_delay = 100
        self.clicked = False
        self.i_clicked = -1
        self.j_clicked = -1
        self.highlight_move = False
        self.best_move = None
        self.all_move = None
        self.visible_menu = False
        self.board_copy = self.board.copy()
        self.candy_images = []
        
        # Initialize pygame window
        self.win = pygame.display.set_mode((self.screenwidth, self.screenheight))

        self.x_cases = numpy.linspace(self.screenwidth / (2 * self.board.M), self.screenwidth - self.screenwidth / (2 * self.board.M), self.board.M)
        self.y_cases = numpy.linspace(self.screenheight / (2 * self.board.N), self.screenheight - self.screenheight / (2 * self.board.N), self.board.N)

        self.width = self.screenwidth / self.board.M / 2
        self.height = self.screenheight / self.board.N / 2

    async def load_images(self):
        """Load candy images asynchronously"""
        try:
            for i in range(1, 8):
                path = f'assets/{"colloc" if self.is_colloc else "candy"}/candy_{i}.png'
                # Check if file exists and is accessible
                if not os.path.exists(path):
                    print(f"Warning: Image file not found: {path}")
                    continue
                
                # Load and scale image
                try:
                    image = pygame.image.load(path)
                    image = pygame.transform.scale(image, 
                                                (int(self.screenwidth / self.board.M / 2),
                                                 int(self.screenheight / self.board.N / 2)))
                    self.candy_images.append(image)
                    print("Loaded image", path)
                except pygame.error as e:
                    print(f"Error loading image {path}: {e}")
                
                # Small delay to prevent blocking
                await asyncio.sleep(0.1)
            
            return True
        except Exception as e:
            print(f"Error in load_images: {e}")
            return False

    async def initialize(self):
        """Initialize pygame and setup display"""
        if self.initialized:
            return True
        
        try:
            pygame.init()
            self.win = pygame.display.set_mode((self.screenwidth, self.screenheight))
            pygame.display.set_caption("Candy Crush (official version)")

            # Calculate grid positions
            self.x_cases = numpy.linspace(self.screenwidth / (2 * self.board.M), 
                                     self.screenwidth - self.screenwidth / (2 * self.board.M), 
                                     self.board.M)
            self.y_cases = numpy.linspace(self.screenheight / (2 * self.board.N), 
                                     self.screenheight - self.screenheight / (2 * self.board.N), 
                                     self.board.N)

            self.width = self.screenwidth / self.board.M / 2
            self.height = self.screenheight / self.board.N / 2

            # Load images
            if not await self.load_images():
                print("Failed to load images")
                return False

            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error in initialize: {e}")
            return False
        
    async def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_q]:
            self.running = False
            return

        await self.handle_keyboard_input(keys)
        await self.handle_mouse_input()

    async def handle_keyboard_input(self, keys):
        """Handle keyboard inputs"""
        if keys[pygame.K_w]:
            await asyncio.sleep(0.05)
            self.time_delay = 100 if self.time_delay == 500 else 500

        if keys[pygame.K_m]:
            self.clicked = False
            mcts = MCTS_CandyCrush_Complex(self.board, exploration_param=5000, 
                                         N_rollout=4, n_simulation=500*5*2, 
                                         no_log=True, write_log_file=False)
            self.best_move, self.all_move = mcts.best_move(return_all=True)
            self.highlight_move = True

        if keys[pygame.K_c]:
            self.save_board_to_file('copied_board.txt')
        
        if keys[pygame.K_v]:
            self.board = read_board_from_file('copied_board.txt')
            self.action = Action(self.board)
            self.board_copy = self.board.copy()
            self.clicked = False
            self.highlight_move = False
            self.all_move = None

        await self.handle_movement_keys(keys)

    async def handle_movement_keys(self, keys):
        """Handle movement-related keyboard inputs"""
        if self.clicked:
            directions = {
                pygame.K_UP: (-1, 0),
                pygame.K_DOWN: (1, 0),
                pygame.K_LEFT: (0, -1),
                pygame.K_RIGHT: (0, 1)
            }
            
            for key, (di, dj) in directions.items():
                if keys[key]:
                    new_i, new_j = self.i_clicked + di, self.j_clicked + dj
                    if 0 <= new_i < self.board.N and 0 <= new_j < self.board.M:
                        self.board_copy = self.board.copy()
                        self.action.swap(self.i_clicked, self.j_clicked, new_i, new_j, step_by_step=True)
                        self.clicked = False
                        await self.update_board()

    async def handle_mouse_input(self):
        """Handle mouse inputs"""
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            for i in range(self.board.N):
                for j in range(self.board.M):
                    if (self.x_cases[j] - self.width / 2 < x < self.x_cases[j] + self.width / 2 and 
                        self.y_cases[i] - self.height / 2 < y < self.y_cases[i] + self.height / 2):
                        self.i_clicked = i
                        self.j_clicked = j
                        self.clicked = True
                        break

    async def update_board(self):
        """Update the board state"""
        while True:
            self.board_visual()
            pygame.display.update()
            await asyncio.sleep(self.time_delay / 1000)
            
            fall1 = self.board.make_it_fall()
            fall2 = self.board.fill_random()
            fall = fall1 + fall2
            
            self.board_visual()
            pygame.display.update()
            await asyncio.sleep(self.time_delay / 1000)
            
            if not self.board.update(fall=fall, step_by_step=True):
                break

    def board_visual(self):
        """Draw the game board"""
        self.win.fill((0, 0, 0))

        if self.highlight_move:
            i1, j1 = self.best_move[0]
            i2, j2 = self.best_move[1]
            pygame.draw.rect(self.win, (0, 255, 0), 
                           (self.x_cases[j1] - self.width / 2 - 4, 
                            self.y_cases[i1] - self.height / 2 - 4, 
                            self.width + 8, self.height + 8))
            pygame.draw.rect(self.win, (0, 255, 0), 
                           (self.x_cases[j2] - self.width / 2 - 4, 
                            self.y_cases[i2] - self.height / 2 - 4, 
                            self.width + 8, self.height + 8))

        # Draw candies
        for i in range(self.board.N):
            for j in range(self.board.M):
                candy = self.board.board[i, j]
                if candy != Candy(0, 'empty'):
                    self.draw_candy(i, j, candy)
                else:
                    pygame.draw.circle(self.win, (250, 0, 0), 
                                    (self.x_cases[j], self.y_cases[i]), 20)

        self.draw_move_info()
        self.draw_menu()
        self.draw_score()

    def draw_candy(self, i, j, candy):
        """Draw individual candy with its effects"""
        print(candy.id)
        print()
        candy_index = int(candy.id) - 1
        if self.clicked and i == self.i_clicked and j == self.j_clicked:
            pygame.draw.rect(self.win, (255, 255, 255), 
                           (self.x_cases[j] - self.width / 2 - 2, 
                            self.y_cases[i] - self.height / 2 - 2, 
                            self.width + 4, self.height + 4))
            
        self.win.blit(self.candy_images[candy_index], 
                     (self.x_cases[j] - self.width / 2, 
                      self.y_cases[i] - self.height / 2))

        # Draw candy effects
        if candy.type == 'raye_hor':
            pygame.draw.line(self.win, (255, 255, 255), 
                           (self.x_cases[j] - self.width / 2, self.y_cases[i]),
                           (self.x_cases[j] + self.width / 2, self.y_cases[i]), 2)
        elif candy.type == 'raye_ver':
            pygame.draw.line(self.win, (255, 255, 255),
                           (self.x_cases[j], self.y_cases[i] - self.height / 2),
                           (self.x_cases[j], self.y_cases[i] + self.height / 2), 2)
        elif candy.type == 'sachet':
            pygame.draw.rect(self.win, (255, 255, 255),
                           (self.x_cases[j] - self.width / 4,
                            self.y_cases[i] - self.height / 4,
                            self.width / 2, self.height / 2), 2)

    def draw_move_info(self):
        """Draw move information"""
        if self.all_move is not None:
            font = pygame.font.Font(None, 18)
            for move, visits, mean_reward in self.all_move:
                i1, j1 = move[0]
                i2, j2 = move[1]
                middle_x = (self.x_cases[j1] + self.x_cases[j2]) / 2
                middle_y = (self.y_cases[i1] + self.y_cases[i2]) / 2
                
                text1 = font.render(f"N: {visits}", True, (255, 255, 255))
                text2 = font.render(f"Q: {mean_reward/100:.1f}", True, (255, 255, 255))
                self.win.blit(text1, (middle_x-20, middle_y - 10))
                self.win.blit(text2, (middle_x-20, middle_y + 10))

    def draw_menu(self):
        """Draw the menu if visible"""
        if self.visible_menu:
            x_menu = 800
            y_menu = 20
            font = pygame.font.Font(None, 24)
            menu_items = [
                f"Speed (change with W)",
                f"{self.time_delay}",
                "",
                "Shortcuts:",
                "M: Run MCTS",
                "U: Update the board",
                "ESC: Show/hide menu",
                "S: Empty the board",
                "R: Return to previous board",
                "C: Copy the board",
                "V: Paste the board",
                "Arrows: Move the candy"
            ]
            
            for i, item in enumerate(menu_items):
                text = font.render(item, True, (255, 255, 255))
                self.win.blit(text, (x_menu, y_menu + i * 30))

    def draw_score(self):
        """Draw the score"""
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.board.score}", True, (255, 255, 255))
        self.win.blit(text, (10, 10))

    async def run(self):
        """Main game loop"""
        if not await self.initialize():
            print("Failed to initialize game")
            return

        try:
            while self.running:
                await self.handle_events()
                self.board_visual()
                pygame.display.flip()
                await asyncio.sleep(0.01)
        except Exception as e:
            print(f"Error in main game loop: {e}")
        finally:
            pygame.quit()

    def save_board_to_file(self, file_path):
        """Save the current board to a text file"""
        with open(file_path, 'w') as file:
            file.write(f"{self.board.N} {self.board.M} {self.board.score}\n")
            for i in range(self.board.N):
                row = [f"{self.board.board[i, j].id}_{self.board.board[i, j].type}"
                      for j in range(self.board.M)]
                file.write(' '.join(row) + '\n')

async def main():
    # Initialize pygame for web
    pygame.init()
    # Main loop
    running = True

    board = Board(8, 8)
    board.fill_random()
    board.update()
    action = Action(board)
    viz = Viz(board, action)
    await viz.load_images()
    while running:
        await viz.handle_events()
        viz.board_visual()
        pygame.display.flip()
        await asyncio.sleep(0.01)
        

if __name__ == '__main__':
    if platform.system() == 'Emscripten':
        # Running in browser via pygbag
        asyncio.run(main())
    else:
        # Running natively
        asyncio.run(main())