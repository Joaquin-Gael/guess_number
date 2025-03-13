from typing import Callable, Any
import torch as th
import torchvision as thv
#import numpy as np
import polars as pl
from pathlib import Path
from tqdm import tqdm
from timeit import default_timer as timer

def print_timer(start: float, end: float, device='cpu'):
    total_time = (end - start)
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def accuracy_fn(y_true: th.Tensor, y_pred: th.Tensor) -> float:
    correct = th.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def clean_data(data_list: list[Any]) -> list[float]:
    data = [float(x) if x is not None else 0.0 for x in data_list]
    return data

DATA_PATH = Path(__file__).parent.parent.joinpath('data')
MODELS_PATH = Path(__file__).parent.parent.joinpath('models')

transform = thv.transforms.Compose([
    thv.transforms.RandomRotation(degrees=15),        # Rotación aleatoria de ±15 grados
    thv.transforms.RandomHorizontalFlip(p=0.5),      # Voltear horizontalmente con probabilidad 0.5
    thv.transforms.RandomResizedCrop(size=(28, 28)), # Recorte redimensionado a 28x28
    thv.transforms.ToTensor(),                       # Convertir a tensor
    thv.transforms.Normalize((0.5,), (0.5,))
])

train_data = thv.datasets.MNIST(
    DATA_PATH,
    train=True,
    download=True,
    transform=transform
)

test_data = thv.datasets.MNIST(
    DATA_PATH,
    train=False,
    download=True,
    transform=transform
)

train_data_loader = th.utils.data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_data_loader = th.utils.data.DataLoader(
    test_data,
    batch_size=32,
    shuffle=True
)

class GuessNumberModelI(th.nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int, lr=0.001):
        super().__init__()
        self.connv_I = th.nn.Sequential(
            th.nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            th.nn.GELU(),
            th.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            th.nn.GELU(),
            th.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.connv_II = th.nn.Sequential(
            th.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            th.nn.GELU(),
            th.nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            th.nn.GELU(),
            th.nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        self.guess = th.nn.Sequential(
            th.nn.Flatten(),
            th.nn.Linear(
                in_features=hidden_units * 7 * 7,
                out_features=output_shape
            ),
            th.nn.Softmax(dim=1)
        )

        self.loss_fn = th.nn.CrossEntropyLoss()

        self.optimizer = th.optim.Adam(
            params=self.parameters(),
            lr=lr,
            betas=(0.5, 0.999)
        )

        self.stats = pl.DataFrame()


    def forward(self, inputs):
        x = self.connv_I(inputs)
        x = self.connv_II(x)
        x = self.guess(x)

        return x

    def train_step(self,
                    data_loader: th.utils.data.DataLoader,
                    accuracy_fn: Callable[[th.Tensor, th.Tensor], float]):
        train_loss = 0
        train_acc = 0
        train_loss_values = []
        train_acc_values = []

        for batch, (X, y) in enumerate(data_loader):
            y_pred: th.Tensor = self(X)

            loss = self.loss_fn(y_pred, y)

            train_loss_values.append(loss)

            train_loss += loss

            train_acc += accuracy_fn(
                y,
                y_pred.argmax(dim=1)
            )

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

        train_loss /= len(data_loader)
        train_acc /= len(data_loader)

        train_acc_values.append(train_acc)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")

        return train_loss, train_acc, train_loss_values, train_acc_values

    def test_step(self,
                  data_loader: th.utils.data.DataLoader,
                  accuracy_fn: Callable[[th.Tensor, th.Tensor], float]):
        test_loss, test_acc = 0, 0
        self.eval()
        with th.inference_mode():
            for X, y in data_loader:
            # 1. Forward pass
                test_pred = self(X)

            # 2. Calculate loss and accuracy
                test_loss += self.loss_fn(test_pred, y)
                test_acc += accuracy_fn(y,
                                        test_pred.argmax(dim=1) # Go from logits -> pred labels
                                        )

            # Adjust metrics and print out
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")

        return test_loss, test_acc

    def train_model(self,
                    train_dataloader: th.utils.data.DataLoader,
                    test_dataloader: th.utils.data.DataLoader,
                    accuracy_fn: Callable[[th.Tensor, th.Tensor], float],
                    epochs=6):
        train_time_start_model = timer()

        train_loss_list = []
        train_acc_list = []
        train_loss_values_list = []
        train_acc_values_list = []

        test_loss_list = []
        test_acc_list = []

        for epoch in tqdm(range(epochs)):
            print(f"Epoch: {epoch}\n---------")

            train_loss, train_acc, train_loss_values, train_acc_values = self.train_step(
                data_loader=train_dataloader,
                accuracy_fn=accuracy_fn
            )

            train_loss_list.append(train_loss.detach().numpy())
            train_acc_list.append(train_acc)
            train_loss_values_list.append(train_acc_values_list)
            train_acc_values_list.append(train_acc_values)


            test_loss, test_acc = self.test_step(
                data_loader=test_dataloader,
                accuracy_fn=accuracy_fn
            )

            test_loss_list.append(test_loss.detach().numpy())
            test_acc_list.append(test_acc)

        train_time_end_model = timer()

        total_train_time_model = print_timer(
            start=train_time_start_model,
            end=train_time_end_model
        )

        assert len(train_loss_list) == len(train_acc_list) == len(test_loss_list), "Las longitudes no coinciden"

        avg_acc = sum(clean_data(test_acc_list)) / len(test_acc_list)

        self.stats = self.stats.with_columns([
            pl.Series('train_loss_list', clean_data(train_loss_list)),
            pl.Series('train_acc_list', clean_data(train_acc_list)),
            pl.Series('train_loss_values_list', train_loss_values_list),
            pl.Series('train_acc_values_list', train_acc_values_list),
            pl.Series('test_loss_list', clean_data(test_loss_list)),
            pl.Series('test_acc_list', clean_data(test_acc_list)),
            pl.Series('avg_acc', [avg_acc]),
            pl.Series('epochs', range(1, epochs+1))
        ])

CLASSES_MAP = train_data.class_to_idx
CLASSES_LIST = train_data.classes
MODEL_NAME = '01_pytorch_guess_numbers_model_I.pth'
MODEL_SAVE = MODELS_PATH / MODEL_NAME

if __name__ == '__main__':
    print(f'Images Shape: {train_data.data[0].shape}\n_________')
    print(f'Image Shape DataLoader: {next(iter(train_data_loader))[0].shape}')
    model = GuessNumberModelI(1, 10, len(train_data.classes), lr=0.5)
    model.train_model(
        train_data_loader,
        test_data_loader,
        accuracy_fn,
        epochs=40
    )

    print(model.stats)

    acc_final = model.stats.get_column('avg_acc')[0]

    if acc_final > 50:
        MODELS_PATH.mkdir(exist_ok=True)
        th.save(
            model,
            MODEL_SAVE
        )
        print(f'Model was saved in the path: {MODEL_SAVE}')
    else:
        print(f'The model have a acc pretty low: {acc_final}%')