import glob
from typing import List
import librosa

from backend.audio_data import AudioData
from backend.daps.data import Devices, Environments, Scripts, Speakers


class DAPSLoader:
    """
    A class to load and manage audio data from the DAPS dataset.
    Attributes:
        path (str): The path to the dataset directory.
    Methods:
    -------
    get_audio(speakers: List[str] = [], scripts: List[str] = [], devices: List[str] = [], environments: List[str] = []) -> List[AudioData]
        Retrieves audio data based on specified filters or all available data if no filters are provided.
    get_all_noisy_audio() -> List[AudioData]
        Retrieves all audio data recorded in noisy environments.
    """

    def __init__(self, path: str) -> None:
        self.__path = path
        pass

    def get_audio(
        self,
        speakers: List[str] = [],
        scripts: List[str] = [],
        devices: List[str] = [],
        environments: List[str] = [],
    ) -> List[AudioData]:
        """
        Retrieves audio data based on specified filters or all available data if no filters are provided.

        Parameters:
            speakers (List[str]): List of speaker identifiers to filter the audio data.
            scripts (List[str]): List of script identifiers to filter the audio data.
            devices (List[str]): List of device identifiers to filter the audio data.
            environments (List[str]): List of environment identifiers to filter the audio data.

        Returns:
            List[AudioData]: A list of AudioData objects that match the specified filters.
        """

        speakers = speakers if speakers else Speakers.get_all()
        scripts = scripts if scripts else Scripts.get_all()
        devices = devices if devices else Devices.get_all()
        environments = environments if environments else Environments.get_all()

        return [
            self.__load_audio(speaker, script, device, environment)
            for speaker in speakers
            for script in scripts
            for device in devices
            for environment in environments
            if self.__get_file(speaker, script, device, environment) is not None
        ]

    def get_all_noisy_audio(self) -> List[AudioData]:
        """
        Retrieves all audio data recorded in noisy environments.

        Returns:
            List[AudioData]: A list of AudioData objects recorded in noisy environments.
        """

        return self.get_audio(
            speakers=Speakers.get_all(),
            scripts=Scripts.get_all(),
            devices=Devices.get_all(),
            environments=Environments.get_all_noisy(),
        )

    def __get_file(self, speaker: str, script: str, device: str, environment: str):
        searched_file = f"{speaker}_{script}_{device}_{environment}.wav"
        found_files = glob.glob(f"{self.__path}/**/{searched_file}", recursive=True)

        return found_files[0] if found_files else None

    def __load_audio(
        self, speaker: str, script: str, device: str, environment: str
    ) -> AudioData:
        file_path = self.__get_file(speaker, script, device, environment)
        y, sr = librosa.load(file_path)
        return AudioData(y, sr)
