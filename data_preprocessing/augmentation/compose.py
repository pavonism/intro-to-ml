import random
from collections.abc import Callable
import numpy as np
import copy

class Compose:
    def __init__(self, transformations : list[tuple[Callable, float]]):
        self.transformations = transformations
        random.seed(400)

    def select_random_transformation(self):
        """
        Gets random transformation function based on probabilities in transformations list.

        Return
        ------
        transformation function, transformation index (for infoString)
        """
        probability_list = [p for (t, p) in self.transformations]
        probability_list = probability_list/np.sum(probability_list)
        random_index = np.random.choice(np.arange(0, len(probability_list)), p=probability_list)
        return self.transformations[random_index][0], random_index

    def augment(self, audio, minOneAugmentation = None):
        _audio = copy.deepcopy(audio)
        infoString = ""

        for i, (t, p) in enumerate(self.transformations):
            if p > random.uniform(0, 1):
                _audio = t(_audio)
                infoString = infoString + str(i)

        if  infoString == "" and minOneAugmentation == True:
            t, k = self.select_random_transformation()
            _audio = t(_audio)
            infoString = infoString + str(k)
            
        if infoString == '':
            infoString = None
        return _audio, infoString

    
    def __call__(self, audio, minOneAugmentation = None):
        return self.augment(audio)
