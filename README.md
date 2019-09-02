# Revisting Cycle-GAN for semi-supervised segmentation
This repo contains the official Pytorch implementation of the paper: [Revisiting CycleGAN for semi-supervised segmentation](https://arxiv.org/abs/1908.11569)

## Contents
1. [Summary of the Model](#1-summary-of-the-model)
2. [Setup instructions and dependancies](#2-setup-instructions-and-dependancies)
3. [Repository Overview](#3-repository-overview)
4. [Running the model](#4-running-the-model)
5. [Some results of the paper](#5-some-results-of-the-paper)
6. [Contact](#6-contact)
7. [License](#7-license)

## 1. Summary of the Model
The following shows the training procedure for our proposed model

<img src='https://github.com/arnab39/Semi-supervised-cycleGAN/blob/master/examples/model_image.png'>

We propose a training procedure for semi-supervised segmentation using the principles of image-to-image translation using GANs. The proposed procedure has been evaluated on three segmentation datasets, namely **VOC**, **Cityscapes**, **ACDC**. We are easily able to achieve *2-4% improvement in the mean IoU for all of our semisupervised model* as compared to the supervised model on the same amount of data. For further information regarding the model, training procedure details you may refer to the paper for further details.

## 2. Setup instructions and dependancies
The code has been written in *Python 3.6* and *Pytorch v1.0* with *Torchvision v0.3*. You can install all the dependancies for the model by running the following command in a virtual environment
```
pip install -r requirements.txt
```
For training/testing the model, you must first download all the 3 datasets. You can download the processed version of the dataset [here](https://drive.google.com/drive/folders/1v_ahAjNGPHjw9G15dlxjImtolne6HZ0B?usp=sharing). Also for storing the results of the validation/testing datasets, checkpoints and tensorboard logs, the directory structure must in the following way:

    .
    ├── arch
    │   ├── ...
    ├── data                     # Follow the way the dataset has been placed here         
    │   ├── ACDC                 # Here the ACDC dataset must be placed
    │   └── Cityscape            # Here the Cityscapes dataset must be placed
    │   └── VOC2012              # Here the VOC train/val dataset must be placed
    │   └── VOC2012test          # Here the VOC test dataset must be placed
    ├── data_utils
    │   ├── ...
    ├── checkpoints              # Create this directory to store checkpoints   
    ├── examples                 
    ├── main.py                  
    ├── model.py
    ├── README.md
    ├── testing.py
    ├── utils.py
    ├── validation.py
    ├── results                  # Create this directory for storing the results
    │   ├── supervised           # Directory for storing supervised results  
    │   └── unsupervised         # Directory for storing semisupervised results
    └── tensorboard_results      # Create this directory to store tensorboard log curves


## 3. Repository Overview
The following are the information regarding the various important files in the directory and their function:

- `arch` : The directory stores the architectures for the generators and discriminators used in our model
- `data_utils` : The dataloaders and also helper functions for data processing
- `main.py` : Stores the various hyperparameter information and default settings
- `model.py` : Stores the training procedure for both supervised and semisupervised model, and also checkpointing and logging details
- `utils.py` : Stores the helper functions required for training

## 4. Running the model
You configure the various defaults that are being specified in the `main.py` file. And also modify the supervision percentage on the dataset by modifying the dataloader calling function in the `model.py` file.

For training/validation/testing the our proposed semisupervised model:

```
python main.py --model 'semisupervised_cycleGAN' --dataset 'voc2012' --gpu_ids '0' --training True    
```

Similar commands for the *validation* and *testing* can be put up by replacing `--training` with `--validation` and `--testing` respectively.

## 5. Some results of the paper
Some of the results produced by our semisupervised model are as follows. *For more such results, consider seeing the main paper and also its supplementary section*

<img src='https://github.com/arnab39/Semi-supervised-cycleGAN/blob/master/examples/result_square.png'>

*The generated labels are the labels obtained from semisupervised model while the generated image are the images obtained by passing labels from label to image network*

## 6. Contact
If you have found our research work helpful, please consider citing the original paper.

If you have any doubt regarding the codebase, you can open up an issue or mail at sanu.arnab@gmail.com / aagarwal@ma.iitr.ac.in

## 7. License
This repository is licensed under MIT license
