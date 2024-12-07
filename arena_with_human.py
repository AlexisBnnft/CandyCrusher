import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from board import Board, state_to_board, Action
from master import Master
import time as time_module

N = 1000
strategies = ["random", 'greedy', "offline", "mcts_base", "mcts_fixed_depth", "combined"]

# Prepare DataFrames for storing results and times
results = pd.DataFrame(columns=strategies, index=range(N))
time_df = pd.DataFrame(columns=strategies, index=range(N))

df = pd.read_csv('human.csv')

# This function will be executed in parallel for each board index.
def evaluate_board(i, state):
    if pd.isna(state):
        return i, None  # Signal that this index has no valid board

    board = state_to_board(state, 7, 7)
    board.score = 0
    game_results = {}
    for s in strategies:
        b_test = board.copy()
        m = Master(s, b_test)
        start_time = time_module.time()
        score = m.run_simulation()
        end_time = time_module.time()
        game_results[s] = (score, end_time - start_time)
    return i, game_results

def main():
    # Extract states from the CSV
    states = df.board_init.values[:N]

    # Use a ProcessPoolExecutor to evaluate each board in parallel
    # Adjust workers as needed (e.g., workers=4 or None to use all cores)
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_board, i, state) for i, state in enumerate(states)]
        
        # Use tqdm to track progress
        for f in tqdm(futures, total=len(futures)):
            i, game_results = f.result()
            # If game_results is None, that means state was invalid, just skip
            if game_results is None:
                continue
            # Populate DataFrames with results
            for s, (score, elapsed) in game_results.items():
                results.loc[i, s] = score
                time_df.loc[i, s] = elapsed

    # Save results to CSV
    results.to_csv("results_arena_with_human.csv", index=False)
    time_df.to_csv("time_arena_with_human.csv", index=False)

if __name__ == "__main__":
    main()