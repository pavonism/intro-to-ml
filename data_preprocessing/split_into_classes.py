import os
import random


def get_all_labels(path: str):
    return [file.split("_")[0] for file in os.listdir(path)]


def split_into_classes(input_path: str, output_path: str):
    labels = get_all_labels(f"{input_path}/train")

    positive_class = random.sample(labels, len(labels) // 2)
    negative_class = [label for label in labels if label not in positive_class]

    for clas in labels:
        if os.path.exists(path + f"/{clas}") == False:
            os.makedirs(path + f"/{clas}")
        for file in os.listdir(path):
            if file.split("_")[0] == clas:
                os.rename(path + f"/{file}", path + f"/{clas}/{file}")
