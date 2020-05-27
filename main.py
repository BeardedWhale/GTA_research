from comet_ml import Experiment
from torch import optim

from conf import parse_opts
from logger import Logger
from model import generate_model
from torchsummary import summary
import torch
from mean import get_mean, get_std, online_mean_and_sd
import os
import json
from dataset.gta_dataset import GTA_crime
from spatial_transforms import Normalize, MultiScaleRandomCrop, Compose, RandomHorizontalFlip, ToTensor, \
    MultiScaleCornerCrop
from target_transforms import ClassLabel
from temporal_transforms import TemporalRandomCrop
import numpy as np
from train import epoch_step
import torch.nn as nn
from utils import STEP, CLASS_MAP

if __name__ == '__main__':
    config = parse_opts()
    experiment = Experiment(api_key="Cbyqfs9Z8auN5ivKsbv2Z6Ogi",
                            project_name="test", workspace="beardedwhale")
    model, params = generate_model(config)
    # tens = torch.tensor(np.random.random((3, 32, 112, 112)))
    # out = model(tens)
    summary(model, input_size=(3, 32, 112, 112))
    dataset_path = '/Users/elizavetabatanina/Projects/gta/GTA_dataset'
    jpg_path = '/Users/elizavetabatanina/Projects/GTA_JPG_DATASET_new'

    config.scales = [config.initial_scale]
    for i in range(1, config.n_scales):
        config.scales.append(config.scales[-1] * config.scale_step)
    config.arch = '{}-{}'.format(config.base_model, config.model_depth)

    config.mean = get_mean(config.norm_value, dataset=config.mean_dataset)
    config.std = get_std(config.norm_value, 'gta')
    print(config)
    with open(os.path.join(config.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(config), opt_file)


    if config.no_mean_norm and not config.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not config.std_norm:
        norm_method = Normalize(config.mean, [1, 1, 1])
    else:
        norm_method = Normalize(config.mean, config.std)

    if not config.no_train:
        assert config.train_crop in ['random', 'corner', 'center']
        if config.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(config.scales, config.sample_size)
        elif config.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(config.scales, config.sample_size)
        elif config.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                config.scales, config.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(config.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(config.sample_duration)
        target_transform = ClassLabel()
        training_data = GTA_crime(
            dataset_path,
            jpg_path,
            'train',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform, sample_duration=config.sample_duration)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_threads,
            pin_memory=True)

        train_logger = Logger(experiment, STEP.TRAIN,  n_classes=config.n_finetune_classes, topk=[1, 2, 3], class_map=CLASS_MAP)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(lr=0.001, params=model.parameters())
        for epoch in range(config.n_epochs):
            epoch_step(epoch, conf=config, criterion=criterion, loaders={STEP.TRAIN: train_loader}, model=model,
                       loggers={STEP.TRAIN: train_logger}, optimizer=optimizer)
    # print(f'CUDA: {config.cuda_id}')
