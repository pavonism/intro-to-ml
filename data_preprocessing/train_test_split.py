import glob
import os
from pathlib import Path
import shutil
from tqdm import tqdm
import random
from typing import Dict, List, Tuple


def divide_into_classes(
    path: str,
    positive_class_count: int = 50,
    min_available_files_for_positive_class: int = 20,
) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    files = glob.glob(f"{path}/**/*_nohash_0.wav", recursive=True)
    speaker_id_to_files: Dict[str, List[str]] = {}

    for file in files:
        speaker_id = file.split("\\")[-1].split("_")[-3]
        if speaker_id not in speaker_id_to_files:
            speaker_id_to_files[speaker_id] = []
        speaker_id_to_files[speaker_id].append(file)

    speakers_with_at_least_min_available_files = [
        speaker_id
        for speaker_id, files in speaker_id_to_files.items()
        if len(files) >= min_available_files_for_positive_class
    ]

    positive_class = random.sample(
        speakers_with_at_least_min_available_files,
        positive_class_count,
    )

    negative_class = [
        speaker_id
        for speaker_id in speaker_id_to_files
        if speaker_id not in positive_class
    ]

    return positive_class, negative_class, speaker_id_to_files


def train_test_split_files(
    input_path: str,
    train_path: str,
    test_path: str,
    validation_path: str,
    testing_split: float = 0.1,
    validation_split: float = 0.1,
):
    for path in [train_path, test_path, validation_path]:
        for label in ["0", "1"]:
            os.makedirs(os.path.join(path, label), exist_ok=True)

    positive_class, negative_class, speaker_to_files = divide_into_classes(input_path)

    positive_class_test_files = split_positive_class(
        testing_split,
        positive_class,
        speaker_to_files,
    )

    negative_class_test_files = split_negative_class(
        testing_split,
        negative_class,
        speaker_to_files,
    )

    positive_class_validation_files = split_positive_class(
        validation_split,
        positive_class,
        speaker_to_files,
    )

    negative_class_validation_files = split_negative_class(
        validation_split,
        negative_class,
        speaker_to_files,
    )

    positive_class_train_files = [
        file for speaker in positive_class for file in speaker_to_files[speaker]
    ]

    negative_class_train_files = [
        file for speaker in negative_class for file in speaker_to_files[speaker]
    ]

    for file in tqdm(glob.glob(f"{input_path}/**/**_nohash_0.wav", recursive=True)):
        word = Path(file).parts[-2]
        file_name = os.path.basename(file)

        if file in positive_class_test_files:
            shutil.copy(file, os.path.join(test_path, "1", f"{word}_{file_name}"))
        elif file in negative_class_test_files:
            shutil.copy(file, os.path.join(test_path, "0", f"{word}_{file_name}"))
        elif file in positive_class_validation_files:
            shutil.copy(file, os.path.join(validation_path, "1", f"{word}_{file_name}"))
        elif file in negative_class_validation_files:
            shutil.copy(file, os.path.join(validation_path, "0", f"{word}_{file_name}"))
        elif file in positive_class_train_files:
            shutil.copy(file, os.path.join(train_path, "1", f"{word}_{file_name}"))
        elif file in negative_class_train_files:
            shutil.copy(file, os.path.join(train_path, "0", f"{word}_{file_name}"))


def split_positive_class(
    split_factor: float,
    positive_class_speakers: List[str],
    speaker_to_files: Dict[str, List[str]],
):
    split = random.sample(
        positive_class_speakers,
        int(len(positive_class_speakers) * split_factor // 2),
    )

    dataset_split_class_files = []

    for speaker_id in split:
        random_sample = random.sample(speaker_to_files[speaker_id], 3)
        for file in random_sample:
            dataset_split_class_files.append(file)
            speaker_to_files[speaker_id].remove(file)

    return dataset_split_class_files


def split_negative_class(
    split_factor: float,
    negative_class_speakers: List[str],
    speaker_to_files: Dict[str, List[str]],
):
    available_speakers = [
        speaker_id
        for speaker_id in negative_class_speakers
        if len(speaker_to_files[speaker_id]) >= 3
    ]

    split = random.sample(
        available_speakers,
        int(len(negative_class_speakers) * split_factor // 2),
    )

    dataset_split_class_files = []

    for speaker_id in split:
        random_sample = random.sample(speaker_to_files[speaker_id], 3)
        for file in random_sample:
            dataset_split_class_files.append(file)
            speaker_to_files[speaker_id].remove(file)

    return dataset_split_class_files
