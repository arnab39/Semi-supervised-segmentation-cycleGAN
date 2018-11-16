# Semi-supervised-cycleGAN

## Requirements
- The code has been written in Python (3.5.2) and PyTorch (0.4.1)

## How to run
* To download datasets (eg. horse2zebra)
```
$ sh ./download_dataset.sh horse2zebra
```
* To run training
```
$ python main.py --training True
```
* To run testing
```
$ python main.py --testing True
```

## We need to edit the data processing and loss pass to implement the proposed model.
