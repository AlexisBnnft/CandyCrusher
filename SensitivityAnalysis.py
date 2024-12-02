from master import *
import tqdm as tqdm
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

## Test mcts fixed depth stratgey for different exploration parameters, for 10 different boards, over 7 moves each time.
## Store in a dataframe, then to a csv file.

# Define the parameters
strategy = "mcts_fixed_depth"
n_moves = 10
n_boards = 100
n_random_params = [1, 2, 3, 5]

df = pd.DataFrame(columns=n_random_params, index=range(n_boards))
df.to_csv("explo_param_analysis_100.csv")


def simulate_board(i):
    board = Board(7, 7)
    board.fill_random()
    board.update()
    board.score = 0
    board_results = {}
    for n_random in n_random_params:
        bcopy = board.copy()
        master = Master(strategy, bcopy, n_moves)
        master.params["N_random"] = n_random
        score = master.run_simulation()
        board_results[n_random] = score
    return i, board_results


def main():
    max_workers = 60  # Set the maximum number of workers to avoid the handle limit
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(simulate_board, i) for i in range(n_boards)]
        for future in tqdm.tqdm(futures):
            i, board_results = future.result()
            for exploration_param, score in board_results.items():
                df.at[i, exploration_param] = score
                df.to_csv("explo_param_analysis_100.csv")


if __name__ == "__main__":
    main()
