import pygame
import numpy as np
import asyncio
from board import Board, Action, read_board_from_file
from candy import Candy, N_CANDY
from mcts_complex import MCTS_CandyCrush_Complex

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
        self.initialized = False
        self.candy_images = []

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
                except pygame.error as e:
                    print(f"Error loading image {path}: {e}")
                
                # Small delay to prevent blocking
                await asyncio.sleep(0)
            
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
            self.x_cases = np.linspace(self.screenwidth / (2 * self.board.M), 
                                     self.screenwidth - self.screenwidth / (2 * self.board.M), 
                                     self.board.M)
            self.y_cases = np.linspace(self.screenheight / (2 * self.board.N), 
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