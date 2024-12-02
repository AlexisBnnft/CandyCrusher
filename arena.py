import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from board import Board
from master import Master
import time as time_module

N = 1000
strategies = ["random", 'greedy', "offline", "mcts_base", "mcts_fixed_depth", "combined"]
results = pd.DataFrame(columns=strategies, index=range(N))
time_df = pd.DataFrame(columns=strategies, index=range(N))


def simulate_game(i):
    b = Board(7, 7)
    b.fill_random()
    b.update()
    b.score = 0
    game_results = {}
    time_results = {}
    for s in strategies:
        b_test = b.copy()
        m = Master(s, b_test)
        start_time = time_module.time()
        score = m.run_simulation()
        end_time = time_module.time()
        game_results[s] = score
        time_results[s] = end_time - start_time
    return i, game_results, time_results


def main():
    with ProcessPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(simulate_game, i) for i in range(N)]
        for future in tqdm(futures):
            i, game_results, time_results = future.result()
            for s, score in game_results.items():
                results.loc[i, s] = score
            for s, duration in time_results.items():
                time_df.loc[i, s] = duration

    results.to_csv("results_arena.csv", index=False)
    time_df.to_csv("time_arena.csv", index=False)


if __name__ == "__main__":
    main()