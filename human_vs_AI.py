
from board import *
from candy import *
from analysis import *
import matplotlib.pyplot as plt
from viz import Viz
import pandas as pd



N_b = 200
try:
    human_df = pd.read_csv("human.csv")
    human_df.drop("Unnamed: 0", axis=1, inplace=True)
    print("Resuming previous session")
except:
    print("Initializing new session")
    human_df = pd.DataFrame(columns=["score", 'board_init'], index = range(N_b))
for i in range(10):
    first_nan_index = human_df['score'].isnull().idxmax()
    b = Board(7,7)
    b.fill_random()
    b.update()
    b.score = 0
    v = Viz(b, Action(b), stop_at_10=True)
    human_df.iloc[first_nan_index + i]['board_init'] = b.state()
    try:
        v.Visualize()
    except:
        human_df.iloc[first_nan_index + i]['score'] = b.score

    human_df.to_csv("human.csv")