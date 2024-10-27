
from board import Board
from candy import Candy
from action import Action
import pygame
import sys
import numpy as np

class Viz:

    def __init__(self, board: Board, action: Action):
        """
        Initialize the Viz class with a board and an action.
        """

        self.board = board
        self.action = action
    
    def Visualize(self):
        """
        Display the current state of the board.
        """

        pygame.init()

        screenwidth=500
        screenheight=500

        win=pygame.display.set_mode((screenwidth,screenheight))
        pygame.display.set_caption("Candy Crush (official version)")


        x_cases = np.linspace(screenwidth / (2 * self.board.M), screenwidth - screenwidth / (2 * self.board.M), self.board.M)
        y_cases = np.linspace(screenheight / (2 * self.board.N), screenheight - screenheight / (2 * self.board.N), self.board.N)

        width=screenwidth/self.board.M/2
        height=screenheight/self.board.N/2

        colors = [(153,0,76), (0, 0,153), (0,102,51), (255,128,0), (255,102,102), (229,204,255), (0, 0, 128), (128, 128, 128), (128, 0, 0), (128, 128, 0)]

        run=True
        i_clicked = -1 
        j_clicked = -1
        clicked=False

        while run:
            
            pygame.time.delay(50)

            for event in pygame.event.get():
                if event.type==pygame.QUIT:
                    run=False

            keys=pygame.key.get_pressed()

            if keys[pygame.K_q]:
                run=False
            
            # Get where the mouse clicked

            if pygame.mouse.get_pressed()[0]:
                x, y = pygame.mouse.get_pos()
                for i in range(self.board.N):
                    for j in range(self.board.M):
                        if x_cases[j]-width/2 < x < x_cases[j]+width/2 and y_cases[i]-height/2 < y < y_cases[i]+height/2:
                            i_clicked = i
                            j_clicked = j
                            clicked=True
                            break
            
            if clicked and keys[pygame.K_UP]:
                if i_clicked-1 >= 0:
                    self.action.swap(i_clicked, j_clicked, i_clicked-1, j_clicked)
                    clicked=False
            if clicked and keys[pygame.K_DOWN]:
                if i_clicked+1 < self.board.N:
                    self.action.swap(i_clicked, j_clicked, i_clicked+1, j_clicked)
                    clicked=False
            if clicked and keys[pygame.K_LEFT]:
                if j_clicked-1 >= 0:
                    self.action.swap(i_clicked, j_clicked, i_clicked, j_clicked-1)
                    clicked=False
            if clicked and keys[pygame.K_RIGHT]:
                if j_clicked+1 < self.board.M:
                    self.action.swap(i_clicked, j_clicked, i_clicked, j_clicked+1)
                    clicked=False
            
            if keys[pygame.K_s]:
                clicked=False
                self.board.empty()
                self.board.fill_random()
                self.board.update()

            win.fill((0,0,0))

            for i in range(self.board.N):
                for j in range(self.board.M):
                    if self.board.board[i,j] != ' ':
                        if clicked and i == i_clicked and j == j_clicked:
                            pygame.draw.rect(win, (255, 255, 255), (x_cases[j]-width/2-2, y_cases[i]-height/2-2, width+4, height+4))
                            pygame.draw.rect(win, colors[int(self.board.board[i,j])-1], (x_cases[j]-width/2, y_cases[i]-height/2, width, height))
                        else:
                            pygame.draw.rect(win, colors[int(self.board.board[i,j])-1], (x_cases[j]-width/2, y_cases[i]-height/2, width, height))
                    else:
                        pygame.draw.circle(win, (250,0,0), (x_cases[j], y_cases[i], 20))
                        

            pygame.display.update()

            # 1 second pause

            pygame.time.delay(500)

            # Update board and display

            self.board.update()

            win.fill((0,0,0))

            for i in range(self.board.N):
                for j in range(self.board.M):
                    if self.board.board[i,j] != ' ':
                        if clicked and i == i_clicked and j == j_clicked:
                            pygame.draw.rect(win, (255, 255, 255), (x_cases[j]-width/2-2, y_cases[i]-height/2-2, width+4, height+4))
                            pygame.draw.rect(win, colors[int(self.board.board[i,j])-1], (x_cases[j]-width/2, y_cases[i]-height/2, width, height))
                        else:
                            pygame.draw.rect(win, colors[int(self.board.board[i,j])-1], (x_cases[j]-width/2, y_cases[i]-height/2, width, height))
                    else:
                        pygame.draw.circle(win, (250,0,0), (x_cases[j], y_cases[i], 20))
                        

            pygame.display.update()

        
        pygame.display.quit()
        pygame.quit()
        sys.exit()

        