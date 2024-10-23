from numpy import ndarray


class AudioData:
    def __init__(self, samples: ndarray, 
                 sample_rate: float, 
                 tag : str, 
                 speaker_id : str, 
                 repetition_no : str):
        self.samples = samples
        self.sample_rate = sample_rate
        self.tag = tag
        self.speaker_id = speaker_id
        self.repetition_no = repetition_no
