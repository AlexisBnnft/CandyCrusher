import argparse
from Viz import *
from board import *

def main():
    parser = argparse.ArgumentParser(description='Candy Crush Game')
    parser.add_argument('--mode', type=str, default='normal', help='Mode of the game (e.g., fun, normal)')
    args = parser.parse_args()

    b = Board(6, 7)
    b.update()
    a = Action(b)
    b.fill_random()

    if args.mode == 'fun':
        v = Viz(b, a, True)
    else:
        v = Viz(b, a)

    v.Visualize()

if __name__ == "__main__":
    main()