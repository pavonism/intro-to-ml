import glob
import os
from typing import Dict, Literal, Optional
import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import copy
import librosa
import soundfile as sf
import numpy as np

from data_preprocessing.spectrogram_sharpener import sharpen_spectrogram
from model_training.architectures import Architecture
from model_training.audio_dataset import AudioDataset
from model_training.trainer import Trainer

from backend.audio_to_spectrogram_converter import AudioToSpectrogramConverter
from backend.spectrograms.utils import AudioLoader
from data_preprocessing.augmentation import audio as audio_augmentations


class CNNClassifier:
    def __init__(
        self,
        path: str,
        architecture: Architecture,
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.__path = path
        self.__device = device
        self.__model = architecture.get_model().to(self.__device)
        self.__transform = architecture.get_transform()
        self.__image_train_path = ""
        self.__image_val_path = ""

        os.makedirs(self.__path, exist_ok=True)
        model_path = f"{self.__path}/model.pth"
        if os.path.exists(model_path) == True:
            self.__model.load_state_dict(torch.load(model_path, weights_only=True))
            self.__model.eval()

    def fit(
        self,
        image_train_path: str,
        image_val_path: str,
        batch_size: int = 32,
        n_epochs: int = 5,
        optimizer: Literal["Adam", "SGD"] = "SGD",
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        measure_layer_robusness_step: Optional[int] = None,
    ):
        if self.__image_train_path != image_train_path:
            self.__image_train_path = image_train_path
            train_set = AudioDataset(
                image_train_path,
                transform=self.__transform,
                device=self.__device,
            )
            self.__train_loader = DataLoader(
                train_set,
                num_workers=8,
                batch_size=batch_size,
                shuffle=True,
                persistent_workers=True,
            )

        if self.__image_val_path != image_val_path:
            self.__image_val_path = image_val_path
            validation_set = AudioDataset(
                image_val_path,
                transform=self.__transform,
                device=self.__device,
            )
            self.__validation_loader = DataLoader(
                validation_set,
                num_workers=8,
                batch_size=batch_size,
                shuffle=True,
                persistent_workers=True,
            )

        print(
            "Running training on GPU..."
            if "cuda" in self.__device
            else "Running training on CPU..."
        )

        trainer = Trainer(
            model=self.__model,
            path=self.__path,
            train_loader=self.__train_loader,
            validation_loader=self.__validation_loader,
            device=self.__device,
            optimizer=optimizer,
            learning_rate=learning_rate,
            momentum=momentum,
            measure_layer_robusness_step=measure_layer_robusness_step,
        )

        validation_loss = trainer.run(n_epochs)

        return validation_loss, trainer.layer_robustness_results

    def predict(
        self, test_image_path: str, test_audio_path: str = None
    ) -> Dict[str, int]:
        file_predictions = {}

        for file in tqdm(glob.glob(f"{test_image_path}/**/**.png", recursive=True)):
            predictions = self.__predict_image(file)

            if test_audio_path is not None:
                _audio_file = (
                    AudioLoader.split_path(file, -1).removesuffix(".png") + ".wav"
                )
                classid = AudioLoader().split_path(file, -2)
                audio_path = f"{test_audio_path}/{classid}/{_audio_file}"
                audio = AudioLoader()._AudioLoader__load_audio(audio_path, "", "", "")
                audio_augmented = audio_augmentations.getFirstSyllable2(
                    audio, before=40, ratio=0.6
                )
                sf.write(
                    "training_file_augmented.wav",
                    audio_augmented.samples,
                    audio_augmented.sample_rate,
                )
                AudioToSpectrogramConverter(False).convert_file(
                    "training_file_augmented.wav", "training_file_augmented.png"
                )
                sharpen_spectrogram(Image.open("training_file_augmented.png")).save(
                    "training_file_augmented.png"
                )
                predictions_augmented = self.__predict_image(
                    "training_file_augmented.png"
                )
                file_predictions[file] = np.mean(
                    [predictions, predictions_augmented], axis=0
                ).argmax()
            else:
                file_predictions[file] = predictions.argmax()

        return file_predictions

    def __predict_image(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.__transform(image).unsqueeze(0)

        self.__model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.to(self.__device)
            outputs = self.__model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        return probabilities.cpu().numpy().flatten()
