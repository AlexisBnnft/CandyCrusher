
from board import Board, Action
from candy import Candy
import pygame
import sys
import numpy as np

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

        board_copy = self.board.copy()

        while run:
            
            pygame.time.delay(50)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            keys = pygame.key.get_pressed()

            if keys[pygame.K_q]:
                run = False
            
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
            if clicked and keys[pygame.K_DOWN]:
                if i_clicked + 1 < self.board.N:
                    board_copy = self.board.copy()
                    self.action.swap(i_clicked, j_clicked, i_clicked + 1, j_clicked, step_by_step=True)
                    clicked = False
                    display_action = True
            if clicked and keys[pygame.K_LEFT]:
                if j_clicked - 1 >= 0:
                    board_copy = self.board.copy()
                    self.action.swap(i_clicked, j_clicked, i_clicked, j_clicked - 1, step_by_step=True)
                    clicked = False
                    display_action = True
            if clicked and keys[pygame.K_RIGHT]:
                if j_clicked + 1 < self.board.M:
                    board_copy = self.board.copy()
                    self.action.swap(i_clicked, j_clicked, i_clicked, j_clicked + 1, step_by_step=True)
                    clicked = False
                    display_action = True

            if display_action==False:
                self.board_visual(candy_images,win,x_cases,width,y_cases,height,clicked,i_clicked,j_clicked)
                pygame.display.update()


            while display_action:
                self.board_visual(candy_images,win,x_cases,width,y_cases,height)
                pygame.display.update()
                pygame.time.delay(1000)
                fall1=self.board.make_it_fall()
                fall2=self.board.fill_random()
                fall=fall1+fall2
                self.board_visual(candy_images,win,x_cases,width,y_cases,height)
                pygame.display.update()
                pygame.time.delay(1000)
                display_action=self.board.update(fall=fall,step_by_step=True)

            

            
            if keys[pygame.K_s]:
                clicked = False
                self.board.empty()
                self.board.fill_random()
                self.board.update()

            if keys[pygame.K_r]:
                clicked = False
                self.board = board_copy
                self.action = Action(self.board)

            if keys[pygame.K_c]:
                self.save_board_to_file('study_board.txt')


            pygame.display.update()

            

        pygame.display.quit()
        pygame.quit()
        sys.exit()


    def board_visual(self,candy_images,win,x_cases,width,y_cases,height,clicked=False,i_clicked=0,j_clicked=0):

        win.fill((0, 0, 0))

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