import numpy as np
class Bounds:
    def __init__(self, lower, upper, enforce="clip"):
        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.enforce = enforce.lower()
    def Upper(self):
        return self.upper
    def Lower(self):
        return self.lower
    def Limits(self, pos):
        npart, ndim = pos.shape
        for i in range(npart):
            if (self.enforce == "resample"):
                for j in range(ndim):
                    if (pos[i,j] <= self.lower[j]) or (pos[i,j] >= self.upper[j]):
                        pos[i,j] = self.lower[j] + (self.upper[j]-self.lower[j])*np.random.random()
            else:
                for j in range(ndim):
                    if (pos[i,j] <= self.lower[j]):
                        pos[i,j] = self.lower[j]
                    if (pos[i,j] >= self.upper[j]):
                        pos[i,j] = self.upper[j]
            pos[i] = self.Validate(pos[i])
        return pos
    def Validate(self, pos):
        return pos




































