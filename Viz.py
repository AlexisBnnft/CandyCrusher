
from board import Board, Action,read_board_from_file
from candy import Candy
from candy import N_CANDY
import pygame
import sys
import numpy as np
from mcts import MCTS_CandyCrush
from mcts_complex import MCTS_CandyCrush_Complex
from tqdm import tqdm


class Viz:

    def __init__(self, board: Board, action: Action, is_colloc: bool = False):
        """
        Initialize the Viz class with a board and an action.
        """

        self.board = board
        self.action = action
        self.is_colloc = is_colloc
    
    def Visualize(self):
        """
        Display the current state of the board.
        """

        pygame.init()

        screenwidth = 800
        screenheight = 800
        time_delay = 100
        win = pygame.display.set_mode((screenwidth, screenheight))
        pygame.display.set_caption("Candy Crush (official version)")

        x_cases = np.linspace(screenwidth / (2 * self.board.M), screenwidth - screenwidth / (2 * self.board.M), self.board.M)
        y_cases = np.linspace(screenheight / (2 * self.board.N), screenheight - screenheight / (2 * self.board.N), self.board.N)

        width = screenwidth / self.board.M / 2
        height = screenheight / self.board.N / 2

        # Load candy images
        candy_images = []
        for i in range(1, 8):  # Assuming there are 10 types of candies
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
        visible_menu = False
        while run:
            
            pygame.time.delay(50)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_q]:
                run = False

            if keys[pygame.K_w]:
                time_delay = 100 if time_delay == 500 else 500

            if keys[pygame.K_m]:
                clicked = False
                mcts = MCTS_CandyCrush_Complex(self.board, exploration_param=5000, N_rollout=4, n_simulation=500*5*2, no_log = True, write_log_file = False)
                best_move, all_move = mcts.best_move(return_all=True)
                highlight_move = True
            

            if keys[pygame.K_c]:
                # Copy the given board to a file
                self.save_board_to_file('copied_board.txt')
            
            if keys[pygame.K_v]:
                # Paste the board from the file
                board = read_board_from_file('copied_board.txt')
                self.board = board
                self.action = Action(self.board)
                board_copy = self.board.copy()
                clicked = False
                highlight_move = False
                all_move = None


            if keys[pygame.K_u]:
                self.board.update()

            if keys[pygame.K_ESCAPE]:  # Press Escape to show popup
                if screenwidth == 1000:
                    screenwidth = 800
                else:
                    screenwidth = 1000
                    visible_menu = True
                win = pygame.display.set_mode((screenwidth, screenheight))
        


            if clicked:
                for i in range(N_CANDY):
                    if keys[pygame.K_1 + i] or keys[pygame.K_KP1 + i]:
                        self.board.board[i_clicked, j_clicked] = Candy(i + 1, 'normal')
                        break
                if keys[pygame.K_1 + 6]:
                    self.board.board[i_clicked, j_clicked] = Candy(7, 'disco')
                if keys[pygame.K_1 + 7]:
                    self.board.board[i_clicked, j_clicked] = Candy(np.random.randint(1, N_CANDY), 'sachet')
                if keys[pygame.K_1 + 8]:
                    self.board.board[i_clicked, j_clicked] = Candy(np.random.randint(1, N_CANDY), 'raye_hor')
                if keys[pygame.K_0]:
                    self.board.board[i_clicked, j_clicked] = Candy(np.random.randint(1, N_CANDY), 'raye_ver')
                
                all_move = None

            
            # Get where the mouse clicked

            if pygame.mouse.get_pressed()[0]:
                x, y = pygame.mouse.get_pos()
                for i in range(self.board.N):
                    for j in range(self.board.M):
                        if x_cases[j] - width / 2 < x < x_cases[j] + width / 2 and y_cases[i] - height / 2 < y < y_cases[i] + height / 2:
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

            if display_action==False:
                self.board_visual(candy_images,win,x_cases,width,y_cases,height,clicked,i_clicked,j_clicked,highlight_move,best_move,all_move, visible_menu,time_delay)
                pygame.display.update()


            while display_action:
                self.board_visual(candy_images,win,x_cases,width,y_cases,height)
                pygame.display.update()
                pygame.time.delay(time_delay)
                fall1=self.board.make_it_fall()
                fall2=self.board.fill_random()
                fall=fall1+fall2
                self.board_visual(candy_images,win,x_cases,width,y_cases,height)
                pygame.display.update()
                pygame.time.delay(time_delay)
                display_action=self.board.update(fall=fall,step_by_step=True)

            

            
            if keys[pygame.K_s]:
                clicked = False
                self.board.empty()
                self.board.fill_random()
                self.board.update()
                self.board.score = 0 # Et non mon grand !

            if keys[pygame.K_r]:
                clicked = False
                self.board = board_copy
                self.action = Action(self.board)


            pygame.display.update()

            

        pygame.display.quit()
        pygame.quit()
        sys.exit()


    def board_visual(self,candy_images,win,x_cases,width,y_cases,height,clicked=False,i_clicked=0,j_clicked=0,highlight_move=False,best_move=None, all_move = None, visible_menu = False, time_delay=None):

        win.fill((0, 0, 0))

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
            x_menu = 800
            y_menu = 20
            font = pygame.font.Font(None, 24)
            menu_text_1 = font.render(f"Speed (change with W)", True, (255, 255, 255))
            win.blit(menu_text_1, (x_menu, y_menu))
            menu_text_2 = font.render(f"{time_delay}", True, (255, 255, 255))
            win.blit(menu_text_2, (x_menu, y_menu+30))

            shortcut_text = font.render(f"Shortcuts:", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+90))
            shortcut_text = font.render(f"M: Run MCTS", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+120))
            shortcut_text = font.render(f"U: Update the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+150))
            shortcut_text = font.render(f"ESC: Show/hide menu", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+180))
            shortcut_text = font.render(f"S: Empty the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+210))
            shortcut_text = font.render(f"R: Reset the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+240))
            shortcut_text = font.render(f"C: Copy the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+270))
            shortcut_text = font.render(f"V: Paste the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+270))
            shortcut_text = font.render(f"Arrows: Move the candy", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+300))


        # Display the score

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {self.board.score}", True, (255, 255, 255))
        win.blit(text, (10, 10))




    def save_board_to_file(self, file_path):
        """
        Save the current board to a text file.
        """
        with open(file_path, 'w') as file:
            # Write the board dimensions
            file.write(f"{self.board.N} {self.board.M}\n")
            
            # Write the board state
            for i in range(self.board.N):
                row = []
                for j in range(self.board.M):
                    candy = self.board.board[i, j]
                    row.append(f"{candy.id}_{candy.type}")
                file.write(' '.join(row) + '\n')
