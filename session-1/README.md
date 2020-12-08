# Session 1
Minimal PyTorch project with a custom dataset, a custom model and a train function. The task is to map random normal noise to 0.
## Installation
### With Conda
Create a conda environment by running
```
conda create --name aidl-session1 python=3.8
```
Then, activate the environment
```
conda activate aidl-session1
```
and install the dependencies
```
pip install -r requirements.txt
```
## Running the project
### With VSCode
If you are using VSCode, install the Python extension. Then, this run configuration will work for you:
```
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Session 1",
            "type": "python",
            "request": "launch",
            "program": "session-1/main.py",
            "console": "integratedTerminal",
            "args": ["--n_samples", "100000", "--n_features", "10", "--n_hidden", "20", "--n_outputs", "5", "--epochs", "10", "--batch_size", "100", "--lr", "0.01"]
        }
    ]
}
```
You should place it in `.vscode/launch.json`. For more information about debugging configurations, check https://code.visualstudio.com/docs/python/debugging