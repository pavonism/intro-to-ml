import glob
import os
from pathlib import Path
import shutil
import random

from tqdm import tqdm


def get_speaker_id(path: str):
    return Path(path).stem.split("_")[0]


def get_speaker_to_files(input_path: str):
    speaker_to_files = {}

    for file in glob.glob(f"{input_path}/**/**.wav", recursive=True):
        speaker = get_speaker_id(file)
        speaker_to_files.setdefault(speaker, []).append(file)

    return speaker_to_files


def train_test_split(input_path: str, output_path: str):
    os.makedirs(output_path, exist_ok=True)

    for word in tqdm(os.listdir(input_path)):
        if word == "_background_noise_":
            continue

        word_path = os.path.join(input_path, word)
        speaker_to_files = get_speaker_to_files(word_path)
        speakers = list(speaker_to_files.keys())

        random.shuffle(speakers)
        train_split = int(0.7 * len(speakers))
        test_split = int(0.2 * len(speakers)) + train_split

        train_speakers = speakers[:train_split]
        test_speakers = speakers[train_split:test_split]
        validation_speakers = speakers[test_split:]

        copy_files(
            output_path,
            word,
            speaker_to_files,
            train_speakers,
            "train",
        )
        copy_files(
            output_path,
            word,
            speaker_to_files,
            test_speakers,
            "test",
        )
        copy_files(
            output_path,
            word,
            speaker_to_files,
            validation_speakers,
            "validation",
        )


def copy_files(
    output_path: str,
    word: str,
    speaker_to_files: dict[str, list[str]],
    speakers: list[str],
    subset: str,
):
    directory = os.path.join(output_path, subset, word)
    os.makedirs(directory, exist_ok=True)

    for speaker in speakers:
        for file in speaker_to_files[speaker]:
            output_file = os.path.join(
                directory,
                os.path.basename(file),
            )

            shutil.copy(file, output_file)
