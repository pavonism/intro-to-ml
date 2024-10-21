import io
import PIL
import PIL.Image
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image

from backend.audio_data import AudioData


class AudioToSpectrogramConverter:
    def __init__(self):
        matplotlib.use("Agg")

    def convert_file(self, audio_path: str, image_path: str):
        y, sr = librosa.load(audio_path)

        spec = np.abs(librosa.stft(y, hop_length=512))
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        librosa.display.specshow(spec, sr=sr)

        # TODO: Convert this to mel spectrogram (??)
        # mel_spect = librosa.feature.melspectrogram(
        #     y=y,
        #     sr=sr,
        #     n_fft=2048,
        #     hop_length=1024,
        # )

        # mel_spect = librosa.power_to_db(spec, ref=np.max)
        # librosa.display.specshow(mel_spect)

        plt.axis("off")
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0)

        pass

    def convert(self, audio: AudioData, dpi: int = 100) -> Image:
        spec = np.abs(librosa.stft(audio.samples, hop_length=512))
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        librosa.display.specshow(spec, sr=audio.sample_rate)

        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png", dpi=dpi)
        buf.seek(0)
        return PIL.Image.open(buf)
    
    def convert_to_bytes(self, audio: AudioData, dpi: int = 100) -> Image:
        spec = np.abs(librosa.stft(audio.samples, hop_length=512))
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        librosa.display.specshow(spec, sr=audio.sample_rate)

        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches="tight", pad_inches=0, format="png", dpi=dpi)
        buf.seek(0)
        return buf
