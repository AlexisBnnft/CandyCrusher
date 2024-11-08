from numpy import *

N_CANDY = 6 # Number of different candies
NUM_DISCO = 7
TYPES = {'normal','raye_hor','raye_ver','sachet','disco','empty'}
TYPE_TO_ID = {'normal':0,'raye_hor':1,'raye_ver':2,'sachet': 3,'disco':4,'empty':5}
TYPE_DISPLAY = {'normal':'N','raye_hor':'H','raye_ver':'V','sachet':'S','disco':'D','empty':'E'}
class Candy:
    def __init__(self, id, type='normal'):
        self.id=id
        if type not in TYPES:
                raise ValueError(f"Type {type} is not a valid candy type.")
        self.type=type

    def __str__(self):
        if self.type == 'empty':
            return ' '
        return str(self.id)
    
    def __repr__(self):
        return f"Candy({self.id})"
    
    def __eq__(self, other):
        return self.id == other.id
    
    def __ne__(self, other):
        return self.id != other.id
    
    def get_type(self):
        return self.type