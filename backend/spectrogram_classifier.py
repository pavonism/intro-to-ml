import os
from model_training.architectures.simple_convolution_network import (
    SimpleConvolutionArchitecture,
)
from model_training.cnn_classifier import CNNClassifier


class SpectrogramClassifier:
    def __init__(self):
        self.class_labels = os.listdir("data/tsrc_spectrograms_cleaned/train")
        self.model = CNNClassifier(
            path="models/robustness_3_single_cnn_tsrc_spectrograms_cleaned_augmented_sharpened/",
            architecture=SimpleConvolutionArchitecture(),
        )

    def predict(self, path: str) -> str:
        predictions = self.model.predict_image(path)

        for i, pred in enumerate(predictions):
            print(f"{self.class_labels[i]}: {pred:.4f}")

        cls = predictions.argmax()
        return self.class_labels[cls]
