import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from board import Board
from master import Master

N = 200
strategies = ["random", "mcts_base", "mcts_fixed_depth", "offline", "combined"]
results = pd.DataFrame(columns=strategies, index=range(N))


def simulate_game(i):
    b = Board(7, 7)
    b.fill_random()
    b.update()
    b.score = 0
    game_results = {}
    for s in strategies:
        b_test = b.copy()
        m = Master(s, b_test)
        score = m.run_simulation()
        game_results[s] = score
    return i, game_results


def main():
    with ProcessPoolExecutor(max_workers=60) as executor:
        futures = [executor.submit(simulate_game, i) for i in range(N)]
        for future in tqdm(futures):
            i, game_results = future.result()
            for s, score in game_results.items():
                results.loc[i, s] = score

    results.to_csv("results_arena.csv", index=False)


if __name__ == "__main__":
    main()
