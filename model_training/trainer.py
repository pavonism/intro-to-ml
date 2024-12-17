from typing import Literal
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
    ):
        self._model = model
        self._path = path
        self._train_loader = train_loader
        self._validation_loader = validation_loader
        self._device = device
        self._learning_rate = learning_rate
        self._momentum = momentum
        self._optimizer = optimizer

    def _save_results(self, train_loss: float, val_loss: float):
        with open(f"{self._path}/train_losses.csv", "a") as f:
            f.write("%s\n" % train_loss)

        with open(f"{self._path}/val_losses.csv", "a") as f:
            f.write("%s\n" % val_loss)

    def run(
        self,
        num_epochs: int,
        epoch_checkpoints: bool = False,
        measure_layer_robusness: bool = False,
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
        initial_state = copy.deepcopy(self._model.state_dict())
        layer_robustness_results = []

        if epoch_checkpoints:
            torch.save(self._model.state_dict(), f"{self._path}/model_0.pth")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss = self._train(criterion, optimizer)
            val_loss = self._validate(criterion)

            if epoch > 0:
                val_loss_rising = (
                    0 if val_loss < val_losses[-1] else val_loss_rising + 1
                )

            if epoch_checkpoints:
                torch.save(
                    self._model.state_dict(), f"{self._path}/model_{epoch + 1}.pth"
                )

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if measure_layer_robusness:
                results = self._measure_layer_robustness(initial_state, epoch)
                layer_robustness_results.extend(results)

            print(f"Epoch {epoch} Done after {time.time() - start_time} seconds")
            self._save_results(train_loss, val_loss)

            if val_loss_rising > 5:
                print(
                    "Validation score increased five times in a row. Early stopping..."
                )
                break

        print("Finished Training")
        torch.save(self._model.state_dict(), f"{self._path}/model.pth")

        return val_losses[-1], layer_robustness_results

    def _train(self, criterion, optimizer):
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
                outputs = F.softmax(self._model(images))
                predicted = torch.argmax(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total

    def _measure_layer_robustness(
        self,
        initial_state: dict,
        epoch: int,
    ):
        eval_accuracy = self._evaluate()

        results = []
        real_state = copy.deepcopy(self._model.state_dict())
        for layer in real_state.keys():
            current_state = copy.deepcopy(real_state)
            current_state[layer] = initial_state[layer].clone()
            self._model.load_state_dict(current_state)
            init_acc = self._evaluate()

            current_state[layer] = torch.randn_like(initial_state[layer])
            self._model.load_state_dict(current_state)
            rnd_acc = self._evaluate()

            result_dict = dict(
                epoch=epoch,
                layer=layer,
                base_accuracy=eval_accuracy,
                init_accuracy=init_acc,
                random_accuracy=rnd_acc,
            )
            results.append(result_dict)

        self._model.load_state_dict(real_state)
        return results
