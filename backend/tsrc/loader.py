import glob
from typing import List
import librosa

from backend.audio_data import AudioData
from backend.tsrc.data import Environments


class TSRCLoader:
    """
    TSRCLoader is responsible for loading "tensorflow-speech-recognition-challenge" dataset from a specified path.
    Attributes:
        path (str): The base path where audio files are stored.
    Methods:
        get_audio() -> List[AudioData]:
            Retrieves audio data based on the specified filters. If no filters are provided, it retrieves all available audio data.
        get_all_noisy_audio() -> List[AudioData]:
            Retrieves all audio data that is classified as noisy.
    """

    def __init__(self, path: str) -> None:
        self.__path = path
        pass

    def get_audio(
        self,
        environments: List[str] = [],
    ) -> List[AudioData]:
        """
        Retrieves audio data for the specified environments.
        Args:
            environments (List[str], optional): A list of environment names to retrieve audio data for.
                                                If not provided, retrieves audio data for all environments.
        Returns:
            List[AudioData]: A list of AudioData objects for the specified environments.
        """

        environments = environments if environments else Environments.get_all()

        return [
            self.__load_audio(file)
            for environment in environments
            for file in self.__get_files(environment)
        ]

    def get_all_noisy_audio(self) -> List[AudioData]:
        """
        Retrieve all audio data from noisy environments.
        This method fetches and returns audio data that has been recorded in
        environments classified as noisy.
        Returns:
            List[AudioData]: A list of audio data objects from noisy environments.
        """

        return self.get_audio(
            environments=Environments.get_all_noisy(),
        )

    def __get_files(self, environment: str):
        found_files = glob.glob(
            f"{self.__path}/**/{environment}/**.wav", recursive=True
        )

        return found_files if found_files else []

    def __load_audio(self, file_path: str) -> AudioData:
        y, sr = librosa.load(file_path)
        return AudioData(y, sr)
