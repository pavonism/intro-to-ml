from concurrent.futures import ThreadPoolExecutor
import glob
import os
from pathlib import Path
from tqdm.notebook import tqdm

from backend.spectrogram_sharpener import SpectrogramSharpener


def sharpen_spectrograms(input_path: str, output_path: str, n_jobs: int = 4):
    sharpener = SpectrogramSharpener()

    for subset in ["train", "test", "validation"]:
        words = os.listdir(f"{input_path}/{subset}")
        words = [word for word in words if word != "_background_noise_"]

        def sharpen_word(word_index: int, word: str):
            word_directory = f"{output_path}/{subset}/{word}"
            os.makedirs(word_directory, exist_ok=True)

            for file in tqdm(
                glob.glob(
                    f"{input_path}/{subset}/{word}/*.png",
                    recursive=True,
                ),
                desc=f"Dataset: {subset}. Converting {word} files to images",
                position=word_index,
            ):
                output_file = f"{word_directory}/{Path(file).stem}.png"

                if os.path.exists(output_file):
                    continue

                sharpener.sharpen_spectrogram_file(file, output_file)

        arguments = [(index, word) for index, word in enumerate(words)]

        with ThreadPoolExecutor(n_jobs) as executor:
            list(executor.map(lambda args: sharpen_word(*args), arguments))
