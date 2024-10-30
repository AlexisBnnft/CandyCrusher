from Viz import *
from board import *

b = Board(6,7)
b.update()
a=Action(b)
b.fill_random()
v = Viz(b,a)
v.Visualize()