import glob
import os
from pathlib import Path
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm


def denoise_audio(y, sr):
    # Reducing noise
    y_denoised = nr.reduce_noise(y=y, sr=sr)

    # Normalizing the audio
    if np.max(np.abs(y_denoised)) != 0:
        y_denoised = y_denoised / np.max(np.abs(y_denoised))

    return y_denoised


def resample_audio(y, sr, target_sr):
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr


def trim_silence(y):
    y, _ = librosa.effects.trim(y)
    return y


def clean_audio(y, sr):
    y = denoise_audio(y, sr)
    y = trim_silence(y)
    y, sr = resample_audio(y, sr, 16000)

    return y, sr


def clean_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, sr = clean_audio(y, sr)

    return y, sr


def clean_audio_folder(input_path: str, output_path: str):
    for dataset in ["train", "test", "validation"]:
        print(f"Cleaning audio files for {dataset} data...")
        for label in ["0", "1"]:
            output_directory = os.path.join(output_path, dataset, label)
            os.makedirs(output_directory, exist_ok=True)

            for file in tqdm(
                glob.glob(
                    f"{input_path}/{dataset}/{label}/**.wav",
                    recursive=True,
                )
            ):
                y, sr = clean_audio_file(file)

                file_path = Path(file)
                output_file_path = os.path.join(
                    output_directory,
                    file_path.name,
                )

                sf.write(output_file_path, y, sr)
