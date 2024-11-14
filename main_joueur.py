import argparse
from Viz import *
from board import *

def main():
    parser = argparse.ArgumentParser(description='Candy Crush Game')
    parser.add_argument('-m', '--mode', type=str, default='normal', help='Mode of the game (e.g., fun, normal)')
    parser.add_argument('-expl', '--exploration_param', type=int, default=EXPLORATION_PARAM, help='Exploration parameter for MCTS')
    parser.add_argument('-nrol', '--n_rollout', type=int, default=N_ROLLOUT, help='Number of rollouts for MCTS')
    parser.add_argument('-nsim', '--n_simulation', type=int, default=N_SIMULATION, help='Number of simulations for MCTS')
    parser.add_argument('-nrand', '--n_random', type=int, default=N_RANDOM, help='Number of random moves for MCTS')
    args = parser.parse_args()


    # Update the constants in Viz.py
    Viz.EXPLORATION_PARAM = args.exploration_param
    Viz.N_ROLLOUT = args.n_rollout
    Viz.N_SIMULATION = args.n_simulation
    Viz.N_RANDOM = args.n_random

    b = Board(7, 7)
    a = Action(b)
    b.fill_random()
    b.update()
    if args.mode == 'fun':
        v = Viz(b, a, True)
    else:
        v = Viz(b, a)
    v.EXPLORATION_PARAM = args.exploration_param
    v.N_ROLLOUT = args.n_rollout
    v.N_SIMULATION = args.n_simulation
    v.N_RANDOM = args.n_random

    v.Visualize()

if __name__ == "__main__":
    main()