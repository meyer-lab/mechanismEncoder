# PyTorch Non-Mechanistic Autoencoder

Simple non-mechanistic autoencoders in PyTorch

## Structure

```
|-- basic_autoencoder
|   |-- data                    <- input data
|   |   `-- amici_trials        <- simulated mechanism data, sorted by number of hidden variables
|   |   `-- me_loader.py        <- DataLoader class for simulated mechanism data
|   |-- encoder                 <- PyTorch autoencoders
|   |   `-- pytorch_encoder     <- simple linear autoencoder
|   `-- lightning_logs          <- logs produced via pytorch-lightning training
|
|-- pytorch_encoder.yaml        <- conda environment file
|-- README.md                   <- README for these basic auto-encoders     
`-- run_encoder.py              <- example encoder usage script
```

## Example Usage

*run_encoder.py* demonstrates an example usage of this PyTorch Autoencoder.
This script accepts a training and testing dataset as command-line arguments,
and can be ran (after installing and activating the provided .yaml environment)
as follows:

```
python run_encoder.py -train [path to training .csv] -test [path to testing .csv]
```
