import numpy as np
from math import floor

class SphereInitializer:
    def __init__(self, npart=10, ndim=3, bounds=None):
        self.npart = npart
        self.ndim = ndim
        self.bounds = bounds

    def InitializeSwarm(self):
        self.swarm = np.zeros((self.npart, self.ndim))
        
        if (self.bounds == None):
            #  No bounds given, just use [0,1)
            lo = np.zeros(self.ndim)
            hi = np.ones(self.ndim)
        else:
            #  Bounds given, use them
            lo = self.bounds.Lower()
            hi = self.bounds.Upper()

        radius = 0.5
        for i in range(self.npart):
            p = np.random.normal(size=self.ndim)
            self.swarm[i] = radius + radius* p / np.sqrt(np.dot(p,p))
        self.swarm = np.abs(hi-lo)*self.swarm + lo

        if (self.bounds != None):
            self.swarm = self.bounds.Limits(self.swarm)

        return self.swarm