import random
class RandomInertia:
    def __init__(self):
        pass
    def CalculateW(self, w0, iterations, max_iter):
        """Return a weight value in the range [0.5,1)"""
        
        return 0.5 + random.random()/2.0