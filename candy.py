from numpy import *

N_CANDY = 6 # Number of different candies

class Candy:
    def __init__(self, id_candy):
        self.id_candy = id_candy

    def __str__(self):
        return str(self.id_candy)
    
    def __repr__(self):
        return f"Candy({self.id_candy})"
    
    def __eq__(self, other):
        return self.id_candy == other.id_candy
    
    def __ne__(self, other):
        return self.id_candy != other.id_candy
    
