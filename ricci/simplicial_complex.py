import itertools
from collections import defaultdict

class SimplicialComplex:
    def __init__(self):
        self.simplices = defaultdict(set) 
        self.weights = {} 

    def add_simplex(self, simplex):
        simplex = tuple(sorted(simplex))
        dim = len(simplex) - 1
        self.simplices[dim].add(simplex)
        self.weights[simplex] = 1.0
        for k in range(dim):
            for face in itertools.combinations(simplex, k + 1):
                face = tuple(sorted(face))
                if face not in self.simplices[k]:
                    self.simplices[k].add(face)
                    self.weights[face] = 1.0
