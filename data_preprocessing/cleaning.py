import glob
import os
from pathlib import Path
import noisereduce as nr
import librosa
import soundfile as sf
import numpy as np
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor


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


def clean_audio_folder(input_path: str, output_path: str, n_jobs: int = 4):
    for dataset in ["train", "test", "validation"]:
        print(f"Cleaning audio files for {dataset} data...")

        def process_word(word_index: int, word: str):
            output_directory = os.path.join(output_path, dataset, word)
            os.makedirs(output_directory, exist_ok=True)

            for file in tqdm(
                glob.glob(
                    f"{input_path}/{dataset}/{word}/**.wav",
                    recursive=True,
                ),
                position=word_index,
                desc=f"Cleaning {word}",
            ):
                y, sr = clean_audio_file(file)

                file_path = Path(file)
                output_file_path = os.path.join(
                    output_directory,
                    file_path.name,
                )

                sf.write(output_file_path, y, sr)

        tasks = [
            (i, word)
            for i, word in enumerate(os.listdir(os.path.join(input_path, dataset)))
        ]

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            list(
                executor.map(lambda args: process_word(*args), tasks),
            )
