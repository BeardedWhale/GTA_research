import os
from comet_ml import Experiment
import json
from torch import nn
from torch import optim
from torch.optim import lr_scheduler

from opts import parse_opts
from model import generate_model
from utils.mean import get_mean, get_std
from utils.spatial_transforms import (
    Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
    MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from utils.temporal_transforms import LoopPadding, TemporalRandomCrop
from utils.target_transforms import ClassLabel, VideoID
from dataset import get_training_set, get_validation_set, get_test_set
from utils import Logger
from train import train_epoch
from validation import val_epoch
from torchsummary import summary
import test
import torch
if __name__ == '__main__':
    opt = parse_opts()
    print(opt.cuda_id)
    print( torch.cuda.device_count() )
    # os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.cuda_id}'
    # os.environ['CUDA_VISIBLE_DEVICE'] = f'{opt.cuda_id}'
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    # torch.device = torch.cud
    # rc = subprocess.check_output(["echo $CUDA_VISIBLE_DEVICE"])
    # print(rc)
    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)

    experiment = Experiment(api_key='Cbyqfs9Z8auN5ivKsbv2Z6Ogi', project_name='GTA-Crime')
    params = {'lr': opt.learning_rate,
              'dampening': opt.dampening,
              'optimizer': opt.optimizer,
              'lr_patience': opt.lr_patience,
              'batch_size': opt.batch_size,
              'n_epochs': opt.n_epochs,
              'begin_epoch': opt.begin_epoch,
              'resume_path': opt.resume_path,
              'pretrain_path': opt.pretrain_path,
              'ft_index': opt.ft_begin_index,
              'cuda_available': opt.cuda_available,
              'cuda_id': opt.cuda_id,
              'model': opt.model,
              'model_type': opt.model_type,
              'model_depth': opt.model_depth,
              'resnet_shortcut': opt.resnet_shortcut}
    experiment.log_parameters(params)
    experiment.add_tag('augmentation')
    experiment.add_tag('multilayer fc module')
    experiment.add_tag('densenext')
    experiment.add_tag(opt.model_type)




    print('Generated')
    if opt.root_path != '':
        opt.video_path = os.path.join(opt.root_path, opt.video_path)
        opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
        opt.result_path = os.path.join(opt.root_path, opt.result_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_path, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_path, opt.pretrain_path)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    opt.arch = '{}-{}'.format(opt.model, opt.model_depth)

    opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
    opt.std = get_std(opt.norm_value, opt.dataset)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)

    torch.manual_seed(opt.manual_seed)

    model, parameters = generate_model(opt)
    print(model)
    summary(model, input_size=(3,32, 112, 112))
    criterion = nn.CrossEntropyLoss()
    if not opt.no_cuda:
        criterion = criterion.cuda(device=opt.cuda_id)
        torch.cuda.device(opt.cuda_id)

    if opt.no_mean_norm and not opt.std_norm:
        norm_method = Normalize([0, 0, 0], [1, 1, 1])
    elif not opt.std_norm:
        norm_method = Normalize(opt.mean, [1, 1, 1])
    else:
        norm_method = Normalize(opt.mean, opt.std)

    if not opt.no_train:
        assert opt.train_crop in ['random', 'corner', 'center']
        if opt.train_crop == 'random':
            crop_method = MultiScaleRandomCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'corner':
            crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        elif opt.train_crop == 'center':
            crop_method = MultiScaleCornerCrop(
                opt.scales, opt.sample_size, crop_positions=['c'])
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = TemporalRandomCrop(opt.sample_duration)
        target_transform = ClassLabel()
        training_data = get_training_set(opt, spatial_transform,
                                         temporal_transform, target_transform)
        train_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])

        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        if opt.optimizer == 'sgd':
            optimizer = optim.SGD(
                parameters,
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=dampening,
                weight_decay=opt.weight_decay,
                nesterov=opt.nesterov)
        else:
            optimizer = optim.Adam(
                parameters,
                lr=opt.learning_rate,
                weight_decay=opt.weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=opt.lr_patience)
    if not opt.no_val:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = ClassLabel()
        validation_data = get_validation_set(
            opt, spatial_transform, temporal_transform, target_transform)
        val_loader = torch.utils.data.DataLoader(
            validation_data,
            batch_size=32,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        val_logger = Logger(
            os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'acc'])

    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])

    print('run')
    val_freq = 10
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger, experiment=experiment)
        if not opt.no_val:
            validation_loss = val_epoch(i, val_loader, model, criterion, opt,
                                    val_logger, experiment=experiment)

        if not opt.no_train and not opt.no_val:
            scheduler.step(validation_loss)

    if opt.test:
        spatial_transform = Compose([
            Scale(int(opt.sample_size / opt.scale_in_test)),
            CornerCrop(opt.sample_size, opt.crop_position_in_test),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = LoopPadding(opt.sample_duration)
        target_transform = VideoID()

        test_data = get_test_set(opt, spatial_transform, temporal_transform,
                                 target_transform)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt, test_data.class_names)
