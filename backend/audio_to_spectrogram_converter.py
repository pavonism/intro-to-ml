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
    def __init__(self, use_agg: bool = True):
        if use_agg:
            matplotlib.use("Agg")

    def convert_file(
        self,
        audio_path: str,
        image_path: str,
        size_in_pixels: tuple = (300, 400),
    ):
        DPI = 100
        width_in_inches = (size_in_pixels[0] + 30) / DPI
        height_in_inches = (size_in_pixels[1] + 30) / DPI

        y, sr = librosa.load(audio_path)

        spec = np.abs(librosa.stft(y, hop_length=512))
        spec = librosa.amplitude_to_db(spec, ref=np.max)

        plt.figure(figsize=(width_in_inches, height_in_inches), dpi=DPI)
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
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
        plt.tight_layout()
        plt.savefig(
            image_path,
            bbox_inches="tight",
            pad_inches=0,
            format="png",
            dpi=DPI,
        )
        plt.close()

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
