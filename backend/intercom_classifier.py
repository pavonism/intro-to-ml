import os
import tempfile
from backend.audio_cleaner import AudioCleaner
from backend.audio_to_spectrogram_converter import AudioToSpectrogramConverter
from backend.spectrogram_classifier import SpectrogramClassifier
from backend.spectrogram_sharpener import SpectrogramSharpener


class WordAudioClassifier:
    def __init__(
        self,
        class_labels: list[str],
        model_path: str,
        verbose=False,
        audio_cleaner: AudioCleaner = AudioCleaner(),
        audio_to_spectrogram_converter: AudioToSpectrogramConverter = AudioToSpectrogramConverter(),
        spectrogram_cleaner: SpectrogramSharpener = SpectrogramSharpener(),
    ):
        self._verbose = verbose
        self.audio_cleaner = audio_cleaner
        self.audio_to_spectrogram_converter = audio_to_spectrogram_converter
        self.spectrogram_cleaner = spectrogram_cleaner
        self.spectrogram_classifier = SpectrogramClassifier(
            class_labels=class_labels,
            model_path=model_path,
        )

    def predict_word(self, path: str) -> str:
        cleaned_file = self.get_temp_cleaned_audio_path()
        self.audio_cleaner.clean_audio_file(path, cleaned_file)
        spec_file = self.get_temp_spectrogram_path()

        if self._verbose:
            print(cleaned_file, spec_file)

        self.audio_to_spectrogram_converter.convert_file(cleaned_file, spec_file)
        self.spectrogram_cleaner.sharpen_spectrogram_file(spec_file, spec_file)
        result, probas = self.spectrogram_classifier.predict(spec_file)
        # self.remove_temp_spectrogram(spectrogram_file)

        if self._verbose:
            for i, pred in enumerate(probas):
                print(f"{self.class_labels[i]}: {pred:.4f}")

        return result

    def get_temp_cleaned_audio_path(self) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file_path = temp_file.name
            return temp_file_path

    def get_temp_spectrogram_path(self) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            temp_file_path = temp_file.name
            return temp_file_path

    def remove_temp_spectrogram(self, path: str):
        os.remove(path)
