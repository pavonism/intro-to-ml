import glob
import os
from pathlib import Path
import shutil
import random

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def get_speaker_id(path: str):
    return Path(path).stem.split("_")[0]


def get_speaker_to_files(input_path: str):
    speaker_to_files = {}

    for file in glob.glob(f"{input_path}/**/**.wav", recursive=True):
        speaker = get_speaker_id(file)
        speaker_to_files.setdefault(speaker, []).append(file)

    return speaker_to_files


def gather_test_speakers(input_path: str):
    all_test_speakers = set()

    for word in os.listdir(input_path):
        if word == "_background_noise_":
            continue

        word_path = os.path.join(input_path, word)
        speaker_to_files = get_speaker_to_files(word_path)
        speakers = list(speaker_to_files.keys())

        already_tested_speakers = all_test_speakers.intersection(speakers)
        speakers_without_tested = list(set(speakers) - already_tested_speakers)

        test_split_ratio = (0.2 * len(speakers) - len(already_tested_speakers)) / len(
            speakers_without_tested
        )

        random.shuffle(speakers_without_tested)
        train_split = int(0.7 * len(speakers_without_tested))
        test_split = int(test_split_ratio * len(speakers_without_tested)) + train_split

        test_speakers = speakers_without_tested[train_split:test_split] + list(
            already_tested_speakers
        )

        all_test_speakers.update(test_speakers)

    return all_test_speakers


def train_test_split(input_path: str, output_path: str, n_jobs: int = 8):
    os.makedirs(output_path, exist_ok=True)

    all_test_speakers = gather_test_speakers(input_path)

    def process_word(word):
        if word == "_background_noise_":
            return

        word_path = os.path.join(input_path, word)
        speaker_to_files = get_speaker_to_files(word_path)

        speakers = list(speaker_to_files.keys())
        test_speakers = all_test_speakers.intersection(speakers)
        speakers_without_tested = list(set(speakers) - set(test_speakers))

        random.shuffle(speakers_without_tested)
        train_split = int(0.9 * len(speakers_without_tested))

        train_speakers = speakers_without_tested[:train_split]
        validation_speakers = speakers_without_tested[train_split:]

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

    words = [word for word in os.listdir(input_path) if word != "_background_noise_"]

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        list(tqdm(executor.map(process_word, words), total=len(words)))


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
