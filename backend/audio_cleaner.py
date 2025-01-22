import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np


class AudioCleaner:
    def _denoise_audio(self, y, sr):
        # Reducing noise
        y_denoised = nr.reduce_noise(y=y, sr=sr)

        # Normalizing the audio
        if np.max(np.abs(y_denoised)) != 0:
            y_denoised = y_denoised / np.max(np.abs(y_denoised))

        return y_denoised

    def _resample_audio(self, y, sr, target_sr):
        if sr != target_sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        return y, sr

    def _trim_silence(self, y):
        y, _ = librosa.effects.trim(y)
        return y

    def _clean_audio(self, y, sr):
        y = self._denoise_audio(y, sr)
        y = self._trim_silence(y)
        y, sr = self._resample_audio(y, sr, 16000)

        return y, sr

    def clean_audio_file(self, input_path: str, output_path: str):
        y, sr = librosa.load(input_path, sr=None)
        y, sr = self._clean_audio(y, sr)

        sf.write(output_path, y, sr)
