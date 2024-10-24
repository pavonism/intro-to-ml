import glob
import os
from pathlib import Path
import shutil

from tqdm import tqdm


def train_test_split_files(
    input_path: str,
    train_path: str,
    test_path: str,
    validation_path: str,
):
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)

    with open(f"{input_path}/testing_list.txt", "r") as f:
        testing_files = set(line.strip() for line in f)

    with open(f"{input_path}/validation_list.txt", "r") as f:
        validation_files = set(line.strip() for line in f)

    for file in tqdm(glob.glob(f"{input_path}/**/**_nohash_0.wav", recursive=True)):
        word = Path(file).parts[-2]
        file_name = os.path.basename(file)
        filename_with_word = f"{word}/{os.path.basename(file)}"

        if filename_with_word in testing_files:
            shutil.copy(file, os.path.join(test_path, f"{word}_{file_name}"))
        elif filename_with_word in validation_files:
            shutil.copy(file, os.path.join(validation_path, f"{word}_{file_name}"))
        else:
            shutil.copy(file, os.path.join(train_path, f"{word}_{file_name}"))
