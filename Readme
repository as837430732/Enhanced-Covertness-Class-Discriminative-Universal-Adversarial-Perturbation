# Enhanced Covertness Class Discriminative Universal Adversarial Perturbations

## Experimental environment:

```shell
Ubuntu=18.04.1 LTS
GeForce GTX 1080ti
```

## Requirements:

```shell
python=3.6
torch=1.5.1
```

## Prepare: 

#### CIFAR-10 and CIFAR-100
 1. Download `cifar-10-python.tar.gz` and `cifar-100-python.tar.gz` in [here](http://www.cs.toronto.edu/~kriz/cifar.html)
 2. Unzip above files to this folder `./data`
#### ImageNet
 1. Follow the common setup to make ImageNet compatible with pytorch as described in [here](https://github.com/pytorch/examples/tree/master/imagenet)
 2. Set the path to the pytorch ImageNet dataset folder in the config file
#### GTSRB
 1. Download `GTSRB-Training_fixed.zip, GTSRB_Final_Test_GT.zip, GTSRB_Final_Test_Images.zip` from [here](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html).
 2. Extract the files into the following Folder structure:
```
GTSRB
    ∟- Training
    ∟- Final_Test
    ∟- GT-final_test.csv
```
 3. Set the path to the GTSRB folder in the config file
 4. Run `python ./dataset_utils/gtsrb_preparation.py`. This should generate a "Testing" folder in your GTSRB folder. The dataset is now ready to be used.

## Running Instructions:

 1. [Optional] Train the target model

    please refer to this file `./experiment/train_model.sh`

    you can also use our pre-trained model in `./models`

 2. Generate perturbations

    non-targeted attack: please refer to this file `./experiment/non-targeted_attack.sh`

    one-targeted attack: please refer to this file `./experiment/one-targeted_attack.sh`

    multi-targeted attack: please refer to this file `./experiment/multi-targeted_attack.sh`

#### Acknowledgment
Initial code is based on  [Benz, Philipp et al., 2020](https://github.com/phibenz/double-targeted-uap.pytorch)