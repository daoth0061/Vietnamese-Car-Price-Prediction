import numpy as np
class RandomInitializer:
    def __init__(self, npart=10, ndim=3, bounds=None):
        self.npart = npart
        self.ndim = ndim
        self.bounds = bounds
    def InitializeSwarm(self):
        if (self.bounds == None):
            self.swarm = np.random.random((self.npart, self.ndim))
        else:
            self.swarm = np.zeros((self.npart, self.ndim))
            lo = self.bounds.Lower()
            hi = self.bounds.Upper()
            for i in range(self.npart):
                for j in range(self.ndim):
                    self.swarm[i,j] = lo[j] + (hi[j]-lo[j])*np.random.random()
            self.swarm = self.bounds.Limits(self.swarm) 
            '''
            Seems meaningless, 
            but the call to Limits ensures that Validate will be called on the newly initialized swarm.
            '''
        return self.swarm