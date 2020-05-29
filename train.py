from comet_ml import Experiment
from torch import optim
from loguru import logger as log
from model import generate_model
from torchsummary import summary
from utils.mean import get_mean, get_std
import json
from dataset.gta_dataset import GTA_crime
from utils.spatial_transforms import Normalize, MultiScaleRandomCrop, Compose, RandomHorizontalFlip, ToTensor, \
    MultiScaleCornerCrop
from utils.target_transforms import ClassLabel
from utils.temporal_transforms import TemporalRandomCrop

import torch.nn as nn
from utils.utils import DEFAULT_CLASS_MAP

from typing import Dict

from torch.autograd import Variable

from utils.logger import Logger
from utils.quadruplet_loss import batch_hard_quadruplet_loss
from utils.utils import STEP
import os
import sys
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


def train(config):
    print(config.n_classes)
    if config.use_quadruplet:
        assert config.use_embeddings, "Cannot use quadruplet loss without Embedding model"


    experiment = Experiment(api_key="Cbyqfs9Z8auN5ivKsbv2Z6Ogi",
                            project_name="gta-crime-classification", workspace="beardedwhale")

    params = {'ft_index': config.ft_begin_index,
              'model': config.base_model,
              'model_type': config.model_type,
              'model_depth': config.model_depth,
              'finetuning_block': config.finetune_block}
    experiment.log_parameters(params)
    experiment.log_parameters(vars(config))
    experiment.add_tag(config.model_type)
    if config.use_quadruplet:
        experiment.add_tag('quadruplet_loss')

    model, params = generate_model(config)
    summary(model, input_size=(3, config.sample_duration, config.sample_size, config.sample_size))
    dataset_path = config.dataset_path
    jpg_path = config.jpg_dataset_path

    config.scales = [config.initial_scale]
    for _ in range(1, config.n_scales):
        config.scales.append(config.scales[-1] * config.scale_step)
    config.arch = '{}-{}'.format(config.base_model, config.model_depth)

    config.mean = get_mean(config.norm_value, dataset=config.mean_dataset)
    config.std = get_std(config.norm_value, 'gta')  # TODO handle gta?
    with open(os.path.join(config.result_path, 'config.json'), 'w') as opt_file:
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
    metrics = []
    if config.use_quadruplet:
        metrics.append('QUADRUPLET_LOSS')
        metrics.append('CLASSIFICATION_LOSS')
        
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
        assert os.path.exists(dataset_path)
        training_data = GTA_crime(
            dataset_path,
            jpg_path,
            'train',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=config.sample_duration,
            dataset_conf_path=config.dataset_conf_path)

        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_threads,
            pin_memory=True)

        train_logger = Logger(experiment, STEP.TRAIN, n_classes=config.n_finetune_classes, topk=[1, 2, 3],
                              class_map=list(training_data.class_map.keys()), metrics=metrics)

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
            target_transform=target_transform,
            sample_duration=config.sample_duration,
            dataset_conf_path=config.dataset_conf_path)
        log.info(f'Loaded validation data: {len(val_data)} samples')
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.n_threads,
            pin_memory=True)
        val_logger = Logger(experiment, STEP.VAL, n_classes=config.n_finetune_classes, topk=[1, 2, 3],
                            class_map=list(val_data.class_map.keys()), metrics=metrics)
        loaders[STEP.VAL] = val_loader
        loggers[STEP.VAL] = val_logger
        steps.append(STEP.VAL)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(lr=config.learning_rate, params=model.parameters())
    for epoch in range(config.n_epochs):
        epoch_step(epoch, conf=config, criterion=criterion, loaders=loaders, model=model,
                   loggers=loggers, optimizer=optimizer)


def epoch_step(epoch, loaders: Dict[STEP, DataLoader], model: torch.nn.Module,
               loggers: Dict[STEP, Logger], criterion, optimizer,
               conf):
    n_iter = sum([len(loader) for loader in loaders.values()])
    print('N iter: ', n_iter)
    steps = list(loaders.keys())
    assert list(loggers.keys()) == list(loaders.keys()), "expected same steps for loaders and loggers"
    with tqdm(total=n_iter, file=sys.stdout) as t:
        t.set_description(f'EPOCH: {epoch + 1}/{conf.n_epochs}')
        for step in steps:
            loader = loaders[step]
            logger = loggers[step]
            if step == STEP.TRAIN:
                model.train()
            else:
                model.eval()
            for i, (inputs, targets, scene_targets) in enumerate(loader):
                additional_metrics = {}
                if conf.cuda_available:
                    targets = targets.cuda(device=conf.cuda_id0, non_blocking=True)
                inputs = Variable(inputs)
                targets = Variable(targets)

                if conf.use_embeddings:
                    embs, outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    
                    if conf.use_quadruplet:
                       
                        batch_hard_loss = conf.quadruplet_alpha * batch_hard_quadruplet_loss(targets, scene_targets, embs,
                                                                                             beta=conf.quadruplet_beta)

                        additional_metrics['QUADRUPLET_LOSS'] = batch_hard_loss.data.cpu() if conf.cuda_available else batch_hard_loss.data
                        additional_metrics['CLASSIFICATION_LOSS'] = loss.data.cpu() if conf.cuda_available else loss.data

                        loss += batch_hard_loss
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                if step == STEP.TRAIN:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if conf.cuda_available:
                    logger.update(loss.data.cpu(), outputs.detach().cpu().numpy(), targets.detach().cpu().numpy(), **additional_metrics)
                else:
                    logger.update(loss.data, outputs.detach().numpy(), targets.detach().numpy(), **additional_metrics)
                t.update(1)
                t.set_postfix(ordered_dict=logger.main_state(), refresh=True)
            logger.update_epoch(epoch)
            logger.reset()

            if epoch % conf.checkpoint == 0:
                save_file_path = os.path.join(conf.result_path,
                                              'save_{}.pth'.format(epoch))
                states = {
                    'epoch': epoch + 1,
                    'arch': conf.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(states, save_file_path)
