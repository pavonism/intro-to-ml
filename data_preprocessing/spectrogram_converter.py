import glob
import os

from tqdm import tqdm
from pathlib import Path

from backend.audio_to_spectrogram_converter import AudioToSpectrogramConverter


def convert_audio_files_to_images(input_path: str, output_path: str):
    converter = AudioToSpectrogramConverter()

    for subset in ["train", "test", "validation"]:
        for word in os.listdir(f"{input_path}/{subset}"):
            if word == "_background_noise_":
                continue

            word_directory = f"{output_path}/{subset}/{word}"
            os.makedirs(word_directory, exist_ok=True)

            for file in tqdm(
                glob.glob(
                    f"{input_path}/{subset}/{word}/**.wav",
                    recursive=True,
                ),
                desc=f"Dataset: {subset}. Converting {word} files to images",
            ):
                output_file = f"{word_directory}/{Path(file).stem}.png"

                if os.path.exists(output_file):
                    continue

                converter.convert_file(file, output_file)
