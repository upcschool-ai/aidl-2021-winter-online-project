
from dataset import MyDataset
from model import MyModel
import torch
from ray import tune
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def accuracy(labels, outputs):
    preds = outputs.argmax(-1)
    acc = (preds == labels.view_as(preds)).float().detach().numpy().mean()
    return acc
    

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

    ray.init(configure_logging=False)
    analysis = tune.run(
        train_model,
        metric="val_loss",
        mode="min",
        num_samples=5,
        config={
            "hyperparam_1": tune.uniform(1, 10),
            "hyperparam_2": tune.grid_search(["relu", "tanh"]),
        })

    print("Best hyperparameters found were: ", analysis.best_config)

