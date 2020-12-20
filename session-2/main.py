
from dataset import MyDataset
from model import MyModel
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def train_single_epoch(...):
    pass


def eval_single_epoch(...):
    pass


def train_model(config):
    
    my_dataset = MyDataset(...)
    my_model = MyModel(...).to(device)
    for epoch in range(config["epochs"]):
        train_single_epoch(...)
        eval_single_epoch(...)


if __name__ == "__main__":

    config = {
        "hyperparam_1": 1,
        "hyperparam_2": 2,
    }
    train_model(config)
