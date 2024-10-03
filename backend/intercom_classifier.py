import os
import tempfile
from backend.audio_cleaner import AudioCleaner
from backend.audio_to_spectrogram_converter import AudioToSpectrogramConverter
from backend.spectrogram_classifier import SpectrogramClassifier
from backend.spectrogram_cleaner import SpectrogramCleaner


class IntercomClassifier:
    def __init__(
        self,
        audio_cleaner: AudioCleaner = AudioCleaner(),
        audio_to_spectrogram_converter: AudioToSpectrogramConverter = AudioToSpectrogramConverter(),
        spectrogram_cleaner: SpectrogramCleaner = SpectrogramCleaner(),
        spectrogram_classifier: SpectrogramClassifier = SpectrogramClassifier(),
    ):
        self.audio_cleaner = audio_cleaner
        self.audio_to_spectrogram_converter = audio_to_spectrogram_converter
        self.spectrogram_cleaner = spectrogram_cleaner
        self.spectrogram_classifier = spectrogram_classifier

    def predict_audio(self, path: str) -> bool:
        self.audio_cleaner.clean(path)
        spectrogram_file = self.get_temp_spectrogram_path()
        print(path, spectrogram_file)
        self.audio_to_spectrogram_converter.convert_file(path, spectrogram_file)
        self.spectrogram_cleaner.clean(spectrogram_file)
        result = self.spectrogram_classifier.predict(spectrogram_file)
        # self.remove_temp_spectrogram(spectrogram_file)

        return result

    def get_temp_spectrogram_path(self) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file_path = temp_file.name
            return temp_file_path

    def remove_temp_spectrogram(self, path: str):
        os.remove(path)
