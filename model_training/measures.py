import glob
import os
from typing import Dict, List

from matplotlib import pyplot as plt
import numpy as np

from model_training.cnn_classifier import CNNClassifier


def get_true_and_predicted_labels(model: CNNClassifier, path: str, path_audio : str = None) -> Dict[str, int]:
    """
    When path_audio is specified, model makes two predictions:
    1. On normal file
    2. On file processed with function getFirstSyllable2
    """
    true_labels = get_true_labels(path)
    predictions = model.predict(path, path_audio)

    y_true = []
    y_pred = []

    for file in true_labels:
        y_true.append(true_labels[file])
        y_pred.append(predictions[file])

    return y_true, y_pred


def get_true_labels(path: str) -> Dict[str, int]:
    file_dicionary = {}

    for index, word in enumerate(os.listdir(path)):
        for file in glob.glob(f"{path}\\{word}\\*.png"):
            file_dicionary[file] = index

    return file_dicionary


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / len(y_true)


def micro_f1_score(y_true: List[int], y_pred: List[int]) -> float:
    confusion = confusion_matrix(y_true, y_pred)

    tp = np.sum(np.diag(confusion))
    fp = np.sum(confusion, axis=0) - np.diag(confusion)
    fn = np.sum(confusion, axis=1) - np.diag(confusion)

    total_tp = tp
    total_fp = np.sum(fp)
    total_fn = np.sum(fn)

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def macro_f1_score(y_true: List[int], y_pred: List[int]) -> float:
    confusion = confusion_matrix(y_true, y_pred)

    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)

    return np.mean(2 * (precision * recall) / (precision + recall))


def weighted_f1_score(y_true: List[int], y_pred: List[int]) -> float:
    confusion = confusion_matrix(y_true, y_pred)

    tp = np.diag(confusion)
    fp = np.sum(confusion, axis=0) - tp
    fn = np.sum(confusion, axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    precision = np.nan_to_num(precision)
    recall = np.nan_to_num(recall)

    weights = np.sum(confusion, axis=1) / np.sum(confusion)

    return np.sum(weights * 2 * (precision * recall) / (precision + recall))


def f1_score(y_true: List[int], y_pred: List[int], average: str = "macro") -> float:
    match average:
        case "micro":
            return micro_f1_score(y_true, y_pred)
        case "macro":
            return macro_f1_score(y_true, y_pred)
        case "weighted":
            return weighted_f1_score(y_true, y_pred)
        case _:
            raise ValueError("Invalid average type")


def confusion_matrix(y_true: List[int], y_pred: List[int]):
    n_classes = len(set(y_true))
    confusion = np.zeros((n_classes, n_classes))

    for true, pred in zip(y_true, y_pred):
        confusion[true, pred] += 1

    return confusion


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(True)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


def summary(y_true: List[int], y_pred: List[int], classes: List[str]):
    print(f"Accuracy: {accuracy(y_true, y_pred)}")
    print(f"Micro F1 Score: {micro_f1_score(y_true, y_pred)}")
    print(f"Macro F1 Score: {macro_f1_score(y_true, y_pred)}")
    print(f"Weighted F1 Score: {weighted_f1_score(y_true, y_pred)}")
    plot_confusion_matrix(confusion_matrix(y_true, y_pred), classes)
