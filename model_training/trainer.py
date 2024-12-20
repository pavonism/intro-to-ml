from typing import Literal, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.functional import F
import time
from tqdm import tqdm
import copy


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        path: str,
        train_loader: torch.utils.data.DataLoader,
        validation_loader: torch.utils.data.DataLoader,
        device: torch.device,
        learning_rate: float,
        momentum: float,
        optimizer: Literal["Adam", "SGD"] = "SGD",
        measure_layer_robusness_step: Optional[int] = None,
    ):
        self._model = model
        self._path = path
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._device = device
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._optimizer = optimizer

        self._measure_layer_robustness_step = measure_layer_robusness_step
        self._total_steps = 0
        self._total_epochs = 0
        self.layer_robustness_results = []

    def _save_results(self, train_loss: float, val_loss: float):
        with open(f"{self._path}/train_losses.csv", "a") as f:
            f.write("%s\n" % train_loss)

        with open(f"{self._path}/val_losses.csv", "a") as f:
            f.write("%s\n" % val_loss)

    def run(
        self,
        num_epochs: int,
    ):
        criterion = nn.CrossEntropyLoss()
        optimizer = (
            optim.Adam(
                self._model.parameters(),
                lr=self._learning_rate,
            )
            if self._optimizer == "Adam"
            else optim.SGD(
                self._model.parameters(),
                lr=self._learning_rate,
                momentum=self._momentum,
            )
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=10,
            threshold=1e-4,
            min_lr=1e-6,
            verbose=True,
        )

        train_losses, val_losses = [], []
        val_loss_rising = 0
        self._initial_state = copy.deepcopy(self._model.state_dict())

        if self._measure_layer_robustness_step:
            torch.save(
                self._model.state_dict(), f"{self._path}/model_{self._total_steps}.pth"
            )

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = self._train(criterion, optimizer)
            val_loss = self._validate(criterion)

            if epoch > 0:
                val_loss_rising = (
                    0 if val_loss < val_losses[-1] else val_loss_rising + 1
                )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)
            self._total_epochs += 1

            print(f"Epoch {epoch} Done after {time.time() - start_time} seconds")
            self._save_results(train_loss, val_loss)

            if val_loss_rising > 5:
                print(
                    "Validation score increased five times in a row. Early stopping..."
                )
                break

        print("Finished Training")
        torch.save(
            self._model.state_dict(), f"{self._path}/model_{self._total_steps}.pth"
        )

        return val_losses[-1]

    def _train(
        self,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        self._model.train()
        running_loss = 0.0
        for images, labels in tqdm(self._train_loader, desc="Training"):
            images, labels = (
                images.to(self._device, non_blocking=True),
                labels.to(self._device, non_blocking=True),
            )

            optimizer.zero_grad()
            outputs = self._model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            self._total_steps += 1

            if (
                self._measure_layer_robustness_step
                and self._total_steps % self._measure_layer_robustness_step == 0
            ):
                print("Measuring layer robustness...")
                self._measure_layer_robustness()

        train_loss = running_loss / len(self._train_loader.dataset)
        return train_loss

    def _validate(self, criterion):
        self._model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(self._validation_loader, desc="Validation"):
                images, labels = (
                    images.to(self._device, non_blocking=True),
                    labels.to(self._device, non_blocking=True),
                )

                outputs = self._model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(self._validation_loader.dataset)
        return val_loss

    def _evaluate(self) -> float:
        self._model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(self._validation_loader, desc="Evaluating"):
                images, labels = images.to(self._device), labels.to(self._device)
                outputs = F.softmax(self._model(images), dim=1)
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def _measure_layer_robustness(
        self,
    ):
        eval_accuracy = self._evaluate()

        real_state = copy.deepcopy(self._model.state_dict())
        for layer in real_state.keys():
            if "weight" not in layer:
                continue

            current_state = copy.deepcopy(real_state)
            current_state[layer] = self._initial_state[layer].clone()
            self._model.load_state_dict(current_state)
            init_acc = self._evaluate()

            mean, std = self.get_mean_std_for_layer(real_state[layer])
            current_state[layer] = torch.normal(
                mean,
                std,
                size=self._initial_state[layer].size(),
                device=self._device,
            )
            self._model.load_state_dict(current_state)
            rnd_acc = self._evaluate()

            result_dict = dict(
                step=self._total_steps,
                epoch=self._total_epochs,
                layer=layer,
                base_accuracy=eval_accuracy,
                init_accuracy=init_acc,
                random_accuracy=rnd_acc,
            )
            self.layer_robustness_results.append(result_dict)

        self._model.load_state_dict(real_state)

        print("Saving weights...")
        torch.save(
            self._model.state_dict(), f"{self._path}/model_{self._total_steps}.pth"
        )

        self._model.train()

    def get_mean_std_for_layer(self, weights: torch.Tensor):
        weights = weights.cpu().detach().numpy().flatten()
        bins = int(np.sqrt(len(weights)))

        mean = np.mean(weights)
        std = np.std(weights)

        bin_counts, hist_edges = np.histogram(weights, bins=bins)
        outlayer_bin_counts = [
            bin_counts[i]
            for i in range(bins)
            if mean - 2 * std > hist_edges[i] or hist_edges[i] > mean + 2 * std
        ]

        if len(outlayer_bin_counts) == 0:
            return mean, std

        mean_outlayer_bin_count = np.mean(outlayer_bin_counts)
        bin_counts = bin_counts - mean_outlayer_bin_count
        bin_counts[bin_counts < 0] = 0

        filtered_mean = (
            sum([hist_edges[i] * bin_counts[i] for i in range(bins)]) / bin_counts.sum()
        )

        filtered_std = np.sqrt(
            np.sum(
                [
                    bin_counts[i] * (hist_edges[i] - filtered_mean) ** 2
                    for i in range(bins)
                ]
            )
            / bin_counts.sum()
        )

        return filtered_mean, filtered_std
