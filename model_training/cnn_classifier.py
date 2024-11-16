import glob
import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from PIL import Image

from model_training.architectures import Architecture
from model_training.audio_dataset import AudioDataset
from model_training.train_loop import Loop


class CNNClassifier:
    def __init__(
        self,
        path: str,
        architecture: Architecture,
    ) -> None:
        self.__path = path
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model = architecture.get_model().to(self.__device)
        self.__transform = architecture.get_transform()

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
    ):
        validation_set = AudioDataset(image_val_path, transform=self.__transform)
        self.__validation_loader = DataLoader(
            validation_set,
            batch_size=batch_size,
            shuffle=True,
        )

        train_set = AudioDataset(image_train_path, transform=self.__transform)
        self.__train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
        )

        print(
            "Running training on GPU..."
            if torch.cuda.is_available()
            else "Running training on CPU..."
        )

        Loop(
            model=self.__model,
            path=self.__path,
            train_loader=self.__train_loader,
            validation_loader=self.__validation_loader,
            device=self.__device,
            num_epochs=n_epochs,
        )

    def predict(self, test_image_path: str) -> Dict[str, int]:
        file_predictions = {}

        for file in glob.glob(f"{test_image_path}/**/**.png", recursive=True):
            predictions = self.__predict_image(file)
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
