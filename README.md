# Federated Privacy Attack

This repository includes 8 privacy attack algorithms employed in federated learning. We follow the official implementations
released by the authors of the original paper, and remove redundant parts to make them as simple as possible.

Most algorithms have ability to reconstruct high-quality dummy images through the gradients with respect to a single
data or multiple averaged data. However, for a more complicated environment in federated learning that *averaged
gradients of batch data are locally computed and updated for multiple times*, these algorithms do not work well.

Among them, robbing the fed presents the best attack performance, however, it requires to insert an Imprint module
(containing two fully connected layers) before the learning model. This would significantly degrade the training
performance of federated learning which is unrealistic in the real world applications. Except that, since the magnitude
of model parameters of the second fully connected layer in Imprint module is *too large*, utilizing training model like
ConvNet may cause severe gradient explosion phenomenon to break down the whole process (result in nan stuff).

## Quick Start

### Gradient Attack
Our implementation reproduces and simulates the original environment of privacy attack upon computed gradients. It 
should be noticed that, this setting differs from an actual federated learning scenarios.

All the running shells are located in ./experiments/grad folder, and you can easily run for example DLG experiments by the following
command:
```sh
cd ./experiment/grad
sh dlg.sh
```
Note that, you can change *BASE_DATADIR in .sh file* to your own downloaded data directory.

### federated Attack
Produce privacy attacks in more realistic federated environment. The corresponding shells are located in
./experiments/fed folder, and you can easily run for example DLG method in federated learning by the following
command:
```sh
cd ./experiment/fed
sh dlg.sh
```

## Model
1) Training and evaluation modes are sensitive to the attack performance when dropout layers, batch normalization 
layers and so on are included in the constructed model

2) Dropout layer is sensitive to the outcome of generated dummy images, *remove it or set the dropout rate to zero*
would get much better and more valid results

3) For running attack method of *ig_multi* or *ig_weight*, the model is required to be wrapped by MetaModel, all the modules should be
built 'sequentially'. We reimplement the original source code to avoid *memory leakage* problem.

4) Utilizing pretrained models does affect the attack performance for GGL and feature inference attack in CPA.


## Data Processing
Scaling: for a typical 8-class image ranges from 0 to 255, the scaling process utilizes the following math equation:
```math
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
```

The mean and standard deviation are sensitive to the quality of generated fake images of GAN. We select 0.5 for both of them.

## Deep Leakage Gradient

## Improved Leakage Gradient

## Robbing the Fed

One problem: the initialized model parameters of the second linear layer in ImprintBlock are too large, which could cause severe gradient explosion in client local training (model should be updated for multiple times on each client). However, if using other initialization methods or scaling to small values may deteriorate the quality of inverted images
