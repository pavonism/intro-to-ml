from typing import List, Tuple
from model_training.architectures.simple_convolution_network import (
    SimpleConvolutionArchitecture,
)
from model_training.cnn_classifier import CNNClassifier


class SpectrogramClassifier:
    def __init__(
        self,
        class_labels: List[str],
        model_path: str,
    ):
        self.class_labels = class_labels
        self.model = CNNClassifier(
            path=model_path,
            architecture=SimpleConvolutionArchitecture(),
        )

    def predict(self, path: str) -> Tuple[str, List[float]]:
        """
        Predicts the class of a spectrogram image.

        Parameters:
        path (str): Path to the spectrogram image.

        Returns:
        Tuple[str, List[float]]: Predicted class and list of
            probabilities for each class.
        """
        predictions = self.model.predict_image(path)
        cls = predictions.argmax()
        return self.class_labels[cls], predictions
