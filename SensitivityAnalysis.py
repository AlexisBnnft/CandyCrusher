from master import *
import tqdm as tqdm
import numpy as np
import pandas as pd

## Test mcts fixed depth stratgey for different exploration parameters, for 10 different boards, over 7 moves each time.
## Store in a dataframe, then to a csv file.

# Define the parameters
strategy = 'mcts_fixed_depth'
n_moves = 7
n_boards = 10
exploration_params = [1000, 2000, 3000, 4000, 5000]

df=pd.DataFrame(columns=exploration_params, index=range(n_boards))

for i in tqdm.tqdm(range(n_boards)):
    board = Board(7, 7)
    board.fill_random()
    board.update()
    board.score = 0
    for exploration_param in exploration_params:
        bcopy = board.copy()
        master = Master(strategy, bcopy, n_moves)
        master.params['exploration_param'] = exploration_param
        score = master.run_simulation()
        df.at[i, exploration_param] = score
    
df.to_csv('explo_param_analysis.csv')
