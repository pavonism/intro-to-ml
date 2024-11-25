import os
import optuna
from model_training.architectures import Architecture
from model_training.cnn_classifier import CNNClassifier


def optimize_hyperparameters(
    architecture: Architecture,
    dataset_path: str,
    folder_path: str,
):
    os.makedirs(folder_path, exist_ok=True)

    def objective_function(trial: optuna.Trial):
        cnn_classifier = CNNClassifier(
            f"{folder_path}/{trial.number}",
            architecture,
        )

        learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        momentum = trial.suggest_float("momentum", 0.5, 0.99)
        batch_size = 96

        validation_loss = cnn_classifier.fit(
            image_train_path=f"{dataset_path}/train",
            image_val_path=f"{dataset_path}/validation",
            n_epochs=5,
            optimizer="SGD",
            learning_rate=learning_rate,
            momentum=momentum,
            batch_size=batch_size,
        )

        return validation_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective_function, n_trials=50)

    print("Best hyperparameters:", study.best_trial.params)
    print("Best validation loss:", study.best_trial.value)
