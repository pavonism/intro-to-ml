import random
from collections.abc import Callable

class Compose:
    def __init__(self, transformations : list[tuple[Callable, float]]):
        self.transformations = transformations
        random.seed(400)

    def augment(self, audio, minOneAugmentation = None):
        _audio = audio
        infoString = ""

        for i, (t, p) in enumerate(self.transformations):
            if p > random.uniform(0, 1):
                _audio = t(_audio)
                infoString = infoString + str(i)

        if  infoString == "" and minOneAugmentation == True:
            k = random.randint(0, len(self.transformations) - 1)
            t, p = self.transformations[k]
            _audio = t(_audio)
            infoString = infoString + str(k)
            
        if infoString == '':
            infoString = None
        return _audio, infoString

    
    def __call__(self, audio, minOneAugmentation = None):
        return self.augment(audio)
