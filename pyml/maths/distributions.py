import numpy as np

class uniform:
    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
    
    def pdf(self, query):
        if self.lower <= query <= self.upper:
            return 1 / (self.upper - self.lower)
        else:
            return 0

class exponential:
    def __init__(self, rate):
        self.rate = rate
    
    def pdf(self, query):
        if query < 0:
            return 0
        else:
            return self.rate * np.exp(-self.rate * query)