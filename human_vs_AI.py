
from board import *
from candy import *
from analysis import *
import matplotlib.pyplot as plt
from viz import Viz
import pandas as pd



N_b = 200
human_df = pd.DataFrame(columns=["score", 'board_init'], index = range(N_b))
for i in range(10):
    b = Board(7,7)
    b.fill_random()
    b.update()
    b.score = 0
    v = Viz(b, Action(b), stop_at_10=True)
    human_df.iloc[i]['board_init'] = b.state()
    try:
        v.Visualize()
    except:
        human_df.iloc[i]['score'] = b.score

    human_df.to_csv("human1.csv")