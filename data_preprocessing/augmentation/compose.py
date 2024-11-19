import random
from collections.abc import Callable

class Compose:
    def __init__(self, transformations : list[tuple[Callable, float]]):
        self.transformations = transformations
        random.seed(400)

    def augment(self, audio):
        _audio = audio
        augmentationsPerformed = 0
        infoString = ""

        for i, (t, p) in enumerate(self.transformations):
            if p > random.uniform(0, 1):
                _audio = t(_audio)
                augmentationsPerformed += 1
                infoString = infoString + str(i)
        if infoString == '':
            infoString = None
        return _audio, infoString

    def augmentMaxNo(self, audio, maxAugmentations):
        _audio = audio
        augmentationsPerformed = 0
        infoString = ""

        while augmentationsPerformed <= maxAugmentations:
            for i, (t, p) in enumerate(self.transformations):
                if p > random.uniform(0, 1) and augmentationsPerformed <= maxAugmentations:
                    _audio = t(_audio)
                    augmentationsPerformed += 1
                    infoString = infoString + str(i)
        return _audio, infoString
    
    def __call__(self, audio, maxAugmentations = None):
        print(f"maxAugmentations: {maxAugmentations}")
        if maxAugmentations is None:
            return self.augment(audio)
        else:
            assert isinstance(maxAugmentations, int)
            if maxAugmentations == 0:
                return audio
            else:
                return self.augmentMaxNo(audio, maxAugmentations)
