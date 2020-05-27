from typing import Dict

from comet_ml import Experiment
from torch import optim
from loguru import logger as log
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

from process.steps import epoch_step
import torch.nn as nn
from utils import STEP, CLASS_MAP


def train(config):
    experiment = Experiment(api_key="Cbyqfs9Z8auN5ivKsbv2Z6Ogi",
                            project_name="test", workspace="beardedwhale")
    model, params = generate_model(config)
    summary(model, input_size=(3, config.sample_duration, config.sample_size, config.sample_size))
    dataset_path = config.dataset_path
    jpg_path = config.jpg_dataset_path

    config.scales = [config.initial_scale]
    for i in range(1, config.n_scales):
        config.scales.append(config.scales[-1] * config.scale_step)
    config.arch = '{}-{}'.format(config.base_model, config.model_depth)

    config.mean = get_mean(config.norm_value, dataset=config.mean_dataset)
    config.std = get_std(config.norm_value, 'gta')
    with open(os.path.join(config.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(config), opt_file)

    if config.no_mean_norm and not config.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not config.std_norm:
        norm_method = Normalize(config.mean, [1, 1, 1])
    else:
        norm_method = Normalize(config.mean, config.std)

    loaders: Dict[STEP, torch.utils.data.DataLoader] = {}
    loggers: Dict[STEP, Logger] = {}
    steps: [STEP] = []
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
        log.info(f'Loaded training data: {len(training_data)} samples')
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_threads,
            pin_memory=True)

        train_logger = Logger(experiment, STEP.TRAIN, n_classes=config.n_finetune_classes, topk=[1, 2, 3],
                              class_map=CLASS_MAP)
        loaders[STEP.TRAIN] = train_loader
        loggers[STEP.TRAIN] = train_logger
        steps.append(STEP.TRAIN)
    if not config.no_val:
        val_data = GTA_crime(
            dataset_path,
            jpg_path,
            'test',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform, sample_duration=config.sample_duration)
        log.info(f'Loaded validation data: {len(val_data)} samples')
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_threads,
            pin_memory=True)
        val_logger = Logger(experiment, STEP.VAL, n_classes=config.n_finetune_classes, topk=[1, 2, 3],
                            class_map=CLASS_MAP)
        loaders[STEP.TRAIN] = val_loader
        loggers[STEP.TRAIN] = val_logger
        steps.append(STEP.VAL)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lr=config.learning_rate, params=model.parameters())
    for epoch in range(config.n_epochs):
        epoch_step(epoch, conf=config, criterion=criterion, loaders=loaders, model=model,
                   loggers=loggers, optimizer=optimizer)

        # if STEP.TRAIN in loggers  and STEP.VAL in loggers:
        #     train_logger = loggers[STEP.TRAIN]
        #     val_logger = loggers[STEP.VAL]
        #     train_logger.accuracy_meter.accuracy