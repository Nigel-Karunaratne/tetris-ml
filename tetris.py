import pygame
import numpy as np
import random

pygame.init()
font = pygame.font.SysFont("monospace", 18)

GRID_X = 10
GRID_Y = 20
SCREEN_W = 300
SCREEN_H = 700
PIECE_SIZE = 30

SHAPE_S = [
[' ',' ',' ',' '],
[' ',' ','0','0'],
[' ','0','0',' '],
[' ',' ',' ',' '],
]

SHAPE_Z = [
[' ',' ',' ',' '],
['1','1',' ',' '],
[' ','1','1',' '],
[' ',' ',' ',' '],
]

SHAPE_L = [
[' ','2',' ',' '],
[' ','2',' ',' '],
[' ','2','2',' '],
[' ',' ',' ',' '],
]

SHAPE_J = [
[' ',' ','3',' '],
[' ',' ','3',' '],
[' ','3','3',' '],
[' ',' ',' ',' '],
]

SHAPE_T = [
[' ',' ',' ',' '],
['4','4','4',' '],
[' ','4',' ',' '],
[' ',' ',' ',' '],
]

SHAPE_O = [
[' ',' ',' ',' '],
[' ','5','5',' '],
[' ','5','5',' '],
[' ',' ',' ',' '],
]

SHAPE_I = [
[' ','6',' ',' '],
[' ','6',' ',' '],
[' ','6',' ',' '],
[' ','6',' ',' '],
]

shapes = [SHAPE_S, SHAPE_Z, SHAPE_L, SHAPE_J, SHAPE_T, SHAPE_O, SHAPE_I]
shapes_colors = [(255,0,0),(0,255,0),(0,0,255),(255,165,0),(255,0,255),(255,255,0),(0,255,255)]

class TetrisGame:
    def __init__(self, gameSpeed = 60, humanGame=False):
        self.gameSpeed = gameSpeed
        self.humanGame=humanGame

        self.py_screen = pygame.display.set_mode((SCREEN_W,SCREEN_H))
        pygame.display.set_caption("simple paithon tetris")
        self.py_clock = pygame.time.Clock()
        self.reset_game()
        return
    
    def spawn_new_piece_from_bag(self): #also checks for gameover
        idx = random.choice(self.bag)
        self.bag.remove(idx)
        if len(self.bag) <= 0:
            self.bag = list(range(0,7))
        self.currentPiece = shapes[idx]
        self.currentPiecePos = [0, GRID_X // 2]
        self.currentPieceRotation = 0
        if self.check_collision(0,0) != 0:
            self.gameOver = True
        return
    
    def place_current_piece(self):
        for y,row in enumerate(self.currentPiece):
            for x,cell in enumerate(row):
                if cell != ' ':
                    self.board[self.currentPiecePos[0] + y][self.currentPiecePos[1] + x] = self.currentPiece[y][x]
        self.clear_lines()
        self.spawn_new_piece_from_bag()
        self.natural_fall_reset()
        return
    
    def move_left(self):
        if self.check_collision(-1,0) == 0:
            self.currentPiecePos[1] -= 1
        return
    def move_right(self):
        if self.check_collision(1,0) == 0:
            self.currentPiecePos[1] += 1
        return
    def move_down(self):
        if self.check_collision(0,1) == 0:
            self.currentPiecePos[0] += 1
        else:
            self.place_current_piece()
        return
    
    def rotate(self):
        _oldCurrentPiece = self.currentPiece.copy()
        self.currentPiece = [list(row) for row in zip(*self.currentPiece[::-1])] #TODO - review this!
        colResult = self.check_collision(0,0)
        if colResult != 0:
            self.currentPiece = _oldCurrentPiece
        return
        if colResult == 1: #on right, see if move over by 1 fixes things
            self.currentPiecePos[1] -= 1
        elif colResult == -1:
            self.currentPiecePos[1] += 1
        return
    
    def natural_fall_increment(self):
        self.naturalFallTimer -= self.level
        if self.naturalFallTimer <= 0:
            self.natural_fall_reset()
            self.move_down()
        return
    def natural_fall_reset(self):
        self.naturalFallTimer = self.naturalFallReset
        return
    
    def clear_lines(self):
        _newBoard = []
        _linesCleared = 0
        for row in self.board:
            if not (' ' in row):
                _linesCleared += 1
            else:
                _newBoard.append(row)
        for _ in range(_linesCleared):
            _newBoard.insert(0, [' '] * GRID_X)
        self.board = _newBoard
        self.clearedLines += _linesCleared
        self.level = (self.clearedLines // 5) + 1
        return

    # 0 for no collision, -1 for left, 1 for right, 2 for above/below
    def check_collision(self, dx, dy):
        for y,row in enumerate(self.currentPiece):
            for x,cell in enumerate(row):
                if cell != ' ':
                    nx = self.currentPiecePos[1] + x + dx
                    ny = self.currentPiecePos[0] + y + dy
                    if nx >= GRID_X:
                        return 1
                    if nx < 0:
                        return -1
                    if ny >= GRID_Y or (ny >= 0 and self.board[ny][nx] != ' '):
                        return 2
        return 0
    
    def draw(self, font):
        self.py_screen.fill((0,0,0))
        for y in range(GRID_Y):
            for x in range(GRID_X):
                pygame.draw.rect(self.py_screen, 'white', (x * PIECE_SIZE, y * PIECE_SIZE, PIECE_SIZE, PIECE_SIZE), 1)

                if self.board[y][x] != ' ':
                    pygame.draw.rect(self.py_screen, shapes_colors[int(self.board[y][x])], pygame.Rect(x * PIECE_SIZE, y * PIECE_SIZE, PIECE_SIZE, PIECE_SIZE))
        for y, row in enumerate(self.currentPiece):
            for x, block in enumerate(row):
                if block != ' ':
                    pygame.draw.rect(self.py_screen, (200,200,200), pygame.Rect((x + self.currentPiecePos[1]) * PIECE_SIZE, (y + self.currentPiecePos[0]) * PIECE_SIZE, PIECE_SIZE, PIECE_SIZE))
        
        levelLabel = font.render("Level: " + str(self.level),1, (255,255,255))
        scoreLabel = font.render("Lines Cleared: " + str(self.clearedLines),1, (255,255,255))
        self.py_screen.blit(levelLabel, (20,620))
        self.py_screen.blit(scoreLabel, (20,650))

        if tetris.gameOver:
            gameOverLabel = font.render("GAME OVER", 1, (255,255,255))
            self.py_screen.blit(gameOverLabel, (20, 670))
        return
    

    def reset_game(self):
        self.level : int = 1
        self.clearedLines : int = 0
        self.gameOver = False
        # self.board = np.full((GRID_X, GRID_Y), ' ', dtype='<U1') #<U1 = unicode str, len=1
        self.board = [[' '] * GRID_X for _ in range(GRID_Y)]
        self.naturalFallReset = 60
        self.naturalFallTimer = self.naturalFallReset
        self.bag = list(range(0,7))
        self.currentPiece = None
        self.currentPiecePos = None    #NOTE: (y,x)
        self.frame_iteration = 0
        self.spawn_new_piece_from_bag()
        return

    # Returns reward, isGameOver, state
    def play_step(self, action) -> tuple[int, bool, int]:
        self.frame_iteration += 1
        reward = 0
        moved_down = False
        oldClearedLines = self.clearedLines

        # TODO - process action

        if not self.humanGame:
            pass
        else:
            pressed = pygame.key.get_pressed()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.gameOver = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.move_left()
                    elif event.key == pygame.K_RIGHT:
                        self.move_right()
                    elif event.key == pygame.K_z:
                        self.rotate()
                if pressed[pygame.K_DOWN]:
                        moved_down = True
                        self.move_down()

        self.natural_fall_increment() if not moved_down else self.natural_fall_reset()
        
        reward = (self.clearedLines - oldClearedLines) * 5
        if self.gameOver:
            reward = -10
        
        self.draw(font)
        pygame.display.flip()
        self.py_clock.tick(tetris.gameSpeed)  # Speed of the game
        
        return reward, self.gameOver, self.get_state()
    
    def get_state(self):
        # flat_board = [(0 if x==' ' else 1) for row in self.board for x in row]
        
        flat_board = np.array(self.board).flatten()
        flat_board = np.where(flat_board == ' ', 0, 1)
        flat_currentPiece = np.array(self.currentPiece).flatten()
        flat_currentPiece = np.where(flat_currentPiece == ' ', 0 ,1)
        flat_currentPiecePos = np.array(self.currentPiecePos)
        
        return np.concatenate([flat_board, flat_currentPiece, flat_currentPiecePos])
    
    def get_state_size(self):
        return 218 #grid size + current piece shape + lines cleared
    def get_action_size(self):
        return 4 #left, right, rotate, down

if __name__ == "__main__":
    tetris = TetrisGame()

    while not tetris.gameOver:
        tetris.play_step('none')
    # OUT OF WHILE LOOP

    # TODO - remove this! only for testing/standalone
    while True:
        tetris.py_screen.fill((0,0,0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        tetris.draw(font)
        pygame.display.flip()
    pygame.quit()
