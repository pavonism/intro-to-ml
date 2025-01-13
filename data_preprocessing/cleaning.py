import glob
import os
from pathlib import Path
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor

from backend.audio_cleaner import AudioCleaner


def clean_audio_folder(input_path: str, output_path: str, n_jobs: int = 4):
    audio_cleaner = AudioCleaner()

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
                file_path = Path(file)
                output_file_path = os.path.join(
                    output_directory,
                    file_path.name,
                )

                audio_cleaner.clean_audio_file(file, output_file_path)

        tasks = [
            (i, word)
            for i, word in enumerate(os.listdir(os.path.join(input_path, dataset)))
        ]

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            list(
                executor.map(lambda args: process_word(*args), tasks),
            )
