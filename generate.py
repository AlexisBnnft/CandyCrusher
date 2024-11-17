import argparse
from board import *
import tqdm as tq
from mcts_complex import MCTS_CandyCrush_Complex


def main():

    parser = argparse.ArgumentParser(description="Candy Crush Game")
    parser.add_argument(
        "-n", "--number", type=int, default=10, help="Number of game to generate"
    )
    parser.add_argument(
        "-expl",
        "--exploration_param",
        type=int,
        default=500,
        help="Exploration parameter for MCTS",
    )
    parser.add_argument(
        "-nrol", "--n_rollout", type=int, default=2, help="Number of rollouts for MCTS"
    )
    parser.add_argument(
        "-nsim",
        "--n_simulation",
        type=int,
        default=2000,
        help="Number of simulations for MCTS",
    )
    parser.add_argument(
        "-nrand",
        "--n_random",
        type=int,
        default=2,
        help="Number of random moves for MCTS",
    )
    args = parser.parse_args()

    n_game = args.number
    explor = args.exploration_param
    n_roll = args.n_rollout
    n_sim = args.n_simulation
    n_rand = args.n_random

    for i in tq.tqdm(range(n_game)):
        b = Board(7, 7)
        a = Action(b)
        b.fill_random()
        b.update()
        b.score = 0

        mcts = MCTS_CandyCrush_Complex(
            b,
            exploration_param=explor,
            N_rollout=n_roll,
            n_simulation=n_sim,
            no_log=True,
            write_log_file=False,
        )
        best_move, all_move = mcts.best_move(return_all=True, N_random=n_rand)

        # Save the board state as well as all_moves list
        with open(f"generated/board_n_{i}.txt", "w") as f:

            # Write the parameters

            f.write(f"{explor}\n")
            f.write(f"{n_roll}\n")
            f.write(f"{n_sim}\n")
            f.write(f"{n_rand}\n")

            f.write(f"{b.state()}\n")
            for move in all_move:
                f.write(f"{move}\n")


if __name__ == "__main__":
    main()
