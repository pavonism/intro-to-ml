from numpy import ndarray


class AudioData:
    def __init__(self, samples: ndarray, 
                 sample_rate: float, 
                 train_validaton_test : str, 
                 speaker_id : str, 
                 class_id : int):
        self.samples = samples
        self.sample_rate = sample_rate
        self.train_validaton_test = train_validaton_test
        self.speaker_id = speaker_id
        self.class_id = class_id
