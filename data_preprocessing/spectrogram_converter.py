import glob
import os

from tqdm import tqdm
from pathlib import Path

from backend.audio_to_spectrogram_converter import AudioToSpectrogramConverter


def convert_audio_files_to_images(input_path: str, output_path: str):
    converter = AudioToSpectrogramConverter()

    for dataset in ["train", "test", "validation"]:
        print(f"Converting audio files to images for {dataset} data...")
        os.makedirs(f"{output_path}/{dataset}", exist_ok=True)
        for file in tqdm(glob.glob(f"{input_path}/{dataset}/**.wav", recursive=True)):
            output_file = f"{output_path}/{dataset}/{Path(file).stem}.png"

            if not os.path.exists(output_file):
                converter.convert_file(file, output_file, (330, 430))
