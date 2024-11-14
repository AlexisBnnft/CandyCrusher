
from board import Board, Action,read_board_from_file
from candy import Candy
from candy import N_CANDY
import pygame
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import sys
import numpy as np
from mcts import MCTS_CandyCrush
from mcts_complex import MCTS_CandyCrush_Complex
from tqdm import tqdm

### MCTS PARAM FOR VIZ
EXPLORATION_PARAM = 5000
N_ROLLOUT = 3
N_SIMULATION = 3000
N_RANDOM = 1


class Viz:

    def __init__(self, board: Board, action: Action, is_colloc: bool = False, N_RANDOM = N_RANDOM, EXPLORATION_PARAM = EXPLORATION_PARAM, N_ROLLOUT = N_ROLLOUT, N_SIMULATION = N_SIMULATION):
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


    
    def Visualize(self):
        """
        Display the current state of the board.
        """

        pygame.init()

        screenwidth = 600
        screenheight = 600
        menu_width = 200
        self.screenwidth = screenwidth
        self.screenheight = screenheight
        time_delay = 100
        win = pygame.display.set_mode((self.screenwidth, self.screenheight))
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
        slot_save = None
        self.board.score = 0
        mcts_mode = False


        x_mcts_mode = 0
        y_mcts_mode = screenheight + 30
        slider_simu = Slider(win, x_mcts_mode+180, y_mcts_mode, 180, 10, min=500, max=10000, step=500, initial=self.N_SIMULATION)
        slider_simu_output = TextBox(win, x_mcts_mode+180+5, y_mcts_mode-30, 175, 20, fontSize=12)
        slider_simu_output.disable()
        slider_explo = Slider(win, x_mcts_mode+380, y_mcts_mode, 180, 10, min=500, max=10000, step=500, initial=self.EXPLORATION_PARAM)
        slider_explo_output = TextBox(win, x_mcts_mode+380+5, y_mcts_mode-30, 175, 20, fontSize=12)
        slider_explo_output.disable()

        while run:
            
            pygame.time.delay(50)

            events = pygame.event.get()

            for event in events:
                if event.type == pygame.QUIT:
                    run = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_q]:
                run = False

            if keys[pygame.K_w]:
                pygame.time.delay(50)
                time_delay = 100 if time_delay == 500 else 500

            if keys[pygame.K_m]:
                if mcts_mode:
                    mcts_mode = False
                    self.screenheight=screenheight
                else:
                    mcts_mode = True
                    self.screenheight=screenheight+50
                win = pygame.display.set_mode((self.screenwidth, self.screenheight))

            if mcts_mode:
                if keys[pygame.K_p]:
                    clicked = False
                    mcts = MCTS_CandyCrush_Complex(self.board, exploration_param=self.EXPLORATION_PARAM, N_rollout=self.N_ROLLOUT, n_simulation=self.N_SIMULATION, no_log = True, write_log_file = False)
                    best_move, all_move = mcts.best_move(return_all=True, N_random = self.N_RANDOM)
                    highlight_move = True
                slider1=slider_simu.getValue()
                self.N_SIMULATION = int(slider1)
                slider_simu_output.setText('Number of simulations:  '+str(self.N_SIMULATION))
                slider2=slider_explo.getValue()
                self.EXPLORATION_PARAM = int(slider2)
                slider_explo_output.setText('Exploration parameter:  '+str(self.EXPLORATION_PARAM))
            

            if keys[pygame.K_c]:
                if slot_save is not None:
                    self.save_board_to_file(f'copied_board_{slot_save}.txt')
                else:
                    self.save_board_to_file('copied_board.txt')
            
            if keys[pygame.K_d]:
                pygame.time.delay(100)
                if slot_save is not None:
                    slot_save += 1
                else:
                    slot_save = 0
                if slot_save > 9:
                    slot_save = 0
            
            if keys[pygame.K_v]:
                # Paste the board from the file
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

            if keys[pygame.K_ESCAPE]:  # Press Escape to show popup
                pygame.time.delay(50)
                if self.screenwidth == screenwidth+menu_width:
                    self.screenwidth = screenwidth
                    visible_menu = False
                else:
                    self.screenwidth = screenwidth+menu_width
                    visible_menu = True
                win = pygame.display.set_mode((self.screenwidth, self.screenheight))
        


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
                self.board_visual(candy_images,win,x_cases,width,y_cases,height,clicked,i_clicked,j_clicked,highlight_move,best_move,all_move, visible_menu,time_delay, save_slot=slot_save,mcts_mode=mcts_mode)
                pygame.display.update()


            while display_action:
                self.board_visual(candy_images,win,x_cases,width,y_cases,height,visible_menu=visible_menu,time_delay=time_delay)
                pygame.display.update()
                pygame.time.delay(time_delay)
                fall1=self.board.make_it_fall()
                fall2=self.board.fill_random()
                fall=fall1+fall2
                self.board_visual(candy_images,win,x_cases,width,y_cases,height,visible_menu=visible_menu,time_delay=time_delay)
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
            pygame_widgets.update(events)

            

        pygame.display.quit()
        pygame.quit()
        sys.exit()


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
            x_menu = self.screenwidth - 190
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
            shortcut_text = font.render(f"M: Enter MCTS Mode", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+120))
            shortcut_text = font.render(f"U: Update the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+150))
            shortcut_text = font.render(f"ESC: Show/hide menu", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+180))
            shortcut_text = font.render(f"S: Empty the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+210))
            shortcut_text = font.render(f"R: Return to previous board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+240))
            shortcut_text = font.render(f"C: Copy the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+270))
            shortcut_text = font.render(f"V: Paste the board", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+300))
            shortcut_text = font.render(f"Arrows: Move the candy", True, (255, 255, 255))
            win.blit(shortcut_text, (x_menu, y_menu+330))


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




    def save_board_to_file(self, file_path):
        """
        Save the current board to a text file.
        """
        with open(file_path, 'w') as file:
            # Write the board dimensions
            file.write(f"{self.board.N} {self.board.M} {self.board.score}\n")
            
            # Write the board state
            for i in range(self.board.N):
                row = []
                for j in range(self.board.M):
                    candy = self.board.board[i, j]
                    row.append(f"{candy.id}_{candy.type}")
                file.write(' '.join(row) + '\n')
