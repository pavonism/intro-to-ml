import glob
import os

import librosa
from tqdm import tqdm
import soundfile as sf


def split_audio_files(input_path: str, output_path: str, duration: int = 3):
    for dataset in ["train", "test", "validation"]:
        print(f"Splitting audio files for {dataset} data...")
        os.makedirs(f"{output_path}/{dataset}", exist_ok=True)

        for file in tqdm(glob.glob(f"{input_path}/{dataset}/**.wav", recursive=True)):
            y, sr = librosa.load(file, sr=None)
            duration_in_samples = duration * sr
            for i in range(0, len(y), duration_in_samples):
                y_split = y[i : i + duration_in_samples]
                output_file_path = os.path.join(
                    output_path,
                    dataset,
                    f"{os.path.basename(file).split('.')[0]}_{i // duration_in_samples}.wav",
                )
                sf.write(output_file_path, y_split, sr)
