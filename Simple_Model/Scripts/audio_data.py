from numpy import ndarray

class AudioData:
    def __init__(self, samples: ndarray, sample_rate: float):
        self.samples = samples
        self.sample_rate = sample_rate


