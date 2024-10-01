# Privacy Attack in Federated Learning is Not Easy

This repository includes more than 8 privacy attack algorithms employed in federated learning. We follow the official implementations
released by the authors of the original paper, and remove redundant parts to make them as simple as possible.

Most algorithms have ability to reconstruct high-quality dummy images through the gradients with respect to a single
data or multiple averaged data. However, for a more complicated environment in federated learning that *averaged
gradients of batch data are locally computed and updated for multiple times*, these algorithms do not work well.

Among them, robbing the fed presents the best attack performance, however, it requires to insert an Imprint module
(containing two fully connected layers) before the learning model. This would significantly degrade the training
performance of federated learning which is unrealistic in the real world applications. Except that, since the magnitude
of model parameters of the second fully connected layer in Imprint module is *too large*, utilizing training model like
ConvNet may cause severe gradient explosion phenomenon to break down the whole process (result in nan stuff).

## Reference
@misc{zhu2024privacyattackfederatedlearning,
      title={Privacy Attack in Federated Learning is Not Easy: An Experimental Study}, 
      author={Hangyu Zhu and Liyuan Huang and Zhenping Xie},
      year={2024},
      eprint={2409.19301},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2409.19301}, 
}

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

### Server Side Attack
Produce privacy attacks in more realistic federated environment on server side. The corresponding shells are located in
./experiments/fed folder, and you can easily run for example DLG method in federated learning by the following
command:
```sh
cd ./experiment/fed
sh dlg.sh
```

#### Label Restoration
It is known that the server is not accessible to local client labels (training sequence). Thus, several work
have investigated label restoration methods from gradients： 1） iDLG (only valid for single label restoration); 2)
[Towards General Deep Leakage in Federated Learning](https://arxiv.org/pdf/2110.09074) (approximate label counts from
averaged gradients with respect to a batch of data); 3) [Deep Leakage in Federated Learning](https://openreview.net/pdf?id=e7A0B99zJf) 
(based on the previous approach).

### Client Side Attack
Produce DMGAN attack method in federated environment on client side. Just run
```sh
cd ./experiment/fed
sh dmgan.sh
```

## Model
1) Training and evaluation modes are sensitive to the attack performance when dropout layers, batch normalization 
layers and so on are included in the constructed model

2) Dropout layer is sensitive to the outcome of generated dummy images, *remove it or set the dropout rate to zero*
would get much better and more valid results

3) For running attack methods *ig_multi* and *ig_weight*, the model is required to be wrapped by MetaModel and all the model modules should be
built 'sequentially'. We reimplement the original source code to avoid *memory leakage* problem.

4) Utilizing pretrained models does affect the attack performance for GGL and feature inversion attack in CPA.

## Data Processing
Scaling: for a typical 8-class image ranges from 0 to 255, the scaling process is $x_{\text{scaled}}=x/255$, where $x$
is the scaled image and $x$ is the original image.

Normalize: data normalization is sensitive to the quality of dummy images in GRNN. The process of normalization is
shown below:
```math
x_{\text{norm}}=\frac{x_{\text{scaled}}-x_{\mu}}{x_{\text{std}}}
```
where $x_{\mu}$ and $x_{\text{std}}$ are mean and standard deviation of the scaled data $x_{\text{scaled}}$, respectively.
According to our experimental results, setting 0.5 for both $x_{\mu}$ and $x_{\text{std}}$ may enhance the quality
of generated GAN images.

However, for CPA-FI attack using pretrained VGG16 model, adopting the officially released (PyTorch) mean and
standard deviation of ImageNet would reconstruct images with higher quality. This implicitly indicate the fact
that the recovered images are highly dependent on the pretrained model parameters which is actually a restriction
for some specific privacy attack methods.

## Deep Models Under the GAN (DMGAN)
The original paper can be found [here](https://dl.acm.org/doi/10.1145/3133956.3134012).
Possibly the earliest privacy attack method for federated learning (only valid on MNIST dataset). Just run the following command:
```sh
cd ./experiment/fed
sh dmgan.sh
```

## Deep Leakage Gradient (DLG)
The original paper can be found [here](https://proceedings.neurips.cc/paper/2019/file/60a6c4002cc7b29142def8871531281a-Paper.pdf).
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh dlg.sh
```

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh dlg.sh
```

## Improved Deep Leakage Gradient (iDLG)
The original paper can be found [here](https://arxiv.org/pdf/2001.02610.pdf).
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh idlg.sh
```

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh idlg.sh
```

## Inverting Gradients
The original paper can be found [here](https://proceedings.neurips.cc/paper/2020/file/c4ede56bbd98819ae6112b20ac6bf145-Paper.pdf).
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh ig.sh
```

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh ig.sh
```

## Robbing the Fed
The original paper can be found [here](https://arxiv.org/pdf/2110.13057).
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh rtf.sh
```

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh rtf.sh
```

## Generative Gradient Leakage (GGL)
The original paper can be found [here](https://arxiv.org/pdf/2203.15696).
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh ggl.sh
```

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh ggl.sh
```

## Generative Regression Neural Network (GRNN)
The original paper can be found [here](https://dl.acm.org/doi/abs/10.1145/3510032).
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh grnn.sh
```

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh grnn.sh
```

## Cocktail Party Attack (CPA)
The original paper can be found [here](https://proceedings.mlr.press/v202/kariyappa23a/kariyappa23a.pdf).
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh cpa.sh
```

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh cpa.sh
```

## Data Leakage in Federated Averaging (DLF)
The original paper can be found [here](https://openreview.net/pdf?id=e7A0B99zJf), which critically
tackle the data restoration for multiple local updates problems.
Just run the following command for gradient attack:
```sh
cd ./experiment/grad
sh dlf.sh
```
Note that, the official implementation adopts the real labels for image restoration. And you can just set 
restore_label = False in file fleak/dlf_attack.py for better quality of reconstructed images

And run the following command for federated attack:
```sh
cd ./experiment/fed
sh dlf.sh
```
Note that, adopting too large learning rate (e.g. 0.1) would cause unexpected running bugs for label restoration !
