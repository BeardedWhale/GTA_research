

This code provides code for experiments on GTA Crime dataset research. It supports following models:


| Base model   | Pretreined on dataset |
|--------------|-----------------------|
|   resnet-18  |        Kinetics       |
|              |   Kinetics + UCF101   |
|   resnet-34  |        Kinetics       |
|              |        Kinetics       |
|   resnet-50  |        Kinetics       |
|              |        Kinetics       |
|  resnext-50  |   Kinetics + HMDB51   |
|              |        Kinetics       |
|  resnext-101 |   Kinetics + HMDB51   |
|              |        Kinetics       |
|   ir-CSN-34  |         $\sim$        |
|   ir-CSN-50  |         $\sim$        |
|   ip-CSN-34  |         $\sim$        |
|   ip-CSN-50  |         $\sim$        |
| Densenet-121 |        Kinetics       |
| Densenet-169 |        Kinetics       |
| Densenet-201 |        Kinetics       |
| Densenet-121 |        Kinetics       |


pretrained models could be found [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing). 



# How to run:
1. Check conf.py file which contains configure options for training/testing. 
2. Check main.py. Choose what experiments setting to run. Contains several experiments: just classification, classification with quadruplet loss, classification with merged classes.
 For only training run train(config) in main.py
 
3. run main.py with suitable options

# Dataset:
[dataset file](dataset/gta_dataset.py) Has functions to process datasets. You can convert dataset to jpgs using `dataset_jpg` function. 
You can generate dataset definition with kfolds for training and testing using `dataset_to_json_kfolds` function.

GTA_crime class defines torch dataset class interface for gta crime dataset. 

# Models:

All models are defined in [model.py](model.py) file. It has main function `generate_model` which generates model with given configurtion.
It also adds finetune block to it or can extend to using Embedding model for Quadruplet loss. 

# Logging
All logging is done by [logger.py](utils/logger.py) module. It automatically logs:

- Accuracy
- mAP
- per class accuracy
- Top 2, top 3 accuracy
- loss
- confusion matrix


# Quadruplet loss
![C labeled formula](http://latex.codecogs.com/svg.latex?\sum_{i}^{N}\left[\left\|f\left(x_{i}^{a}\right)-f\left(x_{i}^{p}\right)\right\|_{2}^{2}-\left\|f\left(x_{i}^{a}\right)-f\left(x_{i}^{n}\right)\right\|_{2}^{2}+\alpha\right])

Quadruplet loss is implemented in [quadruplet_loss.py](utils/quadruplet_loss.py) file. It is implemented with the use of batch hard triplet mining. 
It is implemeted as a sum of triplet loss between classes and between scenes.


### Class configuration:
You can configure any class combination from given default dataset classes:

```yaml
class_map:
  Fire:
    - Arson
    - Explosion
  Property_damage:
    - Robbery
    - Vandalism
  Disarmed_attack:
    - Assault
    - Fight
  Arrest:
    - Arrest
  Shooting:
    - Shooting
```