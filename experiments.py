from collections import namedtuple
from comet_ml import Experiment
from torchsummary import summary

from conf import parse_opts
from model import generate_model
from train import train
import os
from loguru import logger as log
import torch

ModelArch = namedtuple('ModelArchitecture', 'arch type depth path shortcut n_frames fn_block ')


def run_classification_experment(n_epochs=3, blocks=[2]):
    """
    Runs classification experiments to eval various defined architectures
    :return:
    """

    config = parse_opts()
    config.n_epochs = n_epochs
    if config.cuda_available:
        torch.cuda.set_device(config.cuda_id0)
        print(config.cuda_available)
        print(config.cuda_id0)
        print(torch.cuda.device_count())
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())

    architectures = [ModelArch('resnet', '3d', 18, 'pretrained_models/resnet-18-kinetics.pth', 'A', 32, 2),
                     ModelArch('resnet', '3d', 18, 'pretrained_models/resnet-18-kinetics-ucf101_split1.pth', 'A', 16,
                               2),
                     ModelArch('resnet', '3d', 34, 'pretrained_models/resnet-34-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', '3d', 50, 'pretrained_models/resnet-50-kinetics.pth', 'B', 16, 2),
                     ModelArch('resnet', '3d', 101, 'pretrained_models/resnet-101-kinetics.pth', 'B', 16, 2),

                     ModelArch('resnet', 'ir_csn', 18, 'pretrained_models/resnet-18-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', 'ir_csn', 18, 'pretrained_models/resnet-18-kinetics-ucf101_split1.pth', 'A',
                               16, 2),
                     ModelArch('resnet', 'ir_csn', 34, 'pretrained_models/resnet-34-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', 'ir_csn', 50, 'pretrained_models/resnet-50-kinetics.pth', 'B', 16, 2),

                     ModelArch('resnext', '3d', 101, 'pretrained_models/resnext-101-64f-kinetics-hmdb51_split1.pth',
                               'B', 64, 2),
                     ModelArch('resnext', '3d', 101, 'pretrained_models/resnext-101-64f-kinetics-hmdb51_split1.pth',
                               'B', 64, 1)]

    for architecture in architectures:
        assert os.path.exists(architecture.path)
        config.base_model = architecture.arch
        config.model_type = architecture.type
        config.model_depth = architecture.depth
        if architecture.depth < 50:
            config.batch_size = 80
        if config.cuda_available:
            config.pretrain_path = architecture.path
        else:
            config.pretrain_path = ''
        config.resnet_shortcut = architecture.shortcut
        config.sample_duration = architecture.n_frames
        config.finetune_block = architecture.fn_block
        config.learning_rate = 0.0004
        result_path = os.path.join('classification_results', f'{architecture.arch}-{architecture.depth}',
                                   f'{architecture.type}'
                                   f'fn_block_{architecture.fn_block}')
        os.makedirs(result_path, exist_ok=True)
        config.result_path = result_path
        try:
            train(config)
        except Exception as e:
            log.exception(f'Couldn\'t run experiment for model: {architecture}. Error: {e}')

    # learning_rates = [0.005, 0.001,  0.0001]
    # blocks = [1, 2]
    # fn_begin_indexes = [4, 5]
    # train(config)


def run_embeddings_experment(n_epochs=1, blocks=[2]):
    """
    Runs classification experiments to eval various defined architectures
    :return:
    """

    config = parse_opts()
    config.use_embeddings = True
    config.use_quadruplet = True
    config.dataset_path = '/Users/elizavetabatanina/Projects/gta/GTA_dataset'
    config.jpg_dataset_path = '/Users/elizavetabatanina/Projects/GTA_JPG_DATASET_new'
    if config.cuda_available:
        torch.cuda.set_device(config.cuda_id0)
        print(config.cuda_available)
        print(config.cuda_id0)
        print(torch.cuda.device_count())
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())

    architectures = [ModelArch('resnet', '3d', 18, 'pretrained_models/resnet-18-kinetics.pth', 'A', 32, 2),
                     ModelArch('resnet', '3d', 18, 'pretrained_models/resnet-18-kinetics-ucf101_split1.pth', 'A', 16,
                               2),
                     ModelArch('resnet', '3d', 34, 'pretrained_models/resnet-34-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', '3d', 50, 'pretrained_models/resnet-50-kinetics.pth', 'B', 16, 2),
                     ModelArch('resnet', '3d', 101, 'pretrained_models/resnet-101-kinetics.pth', 'B', 16, 2),

                     ModelArch('resnet', 'ir_csn', 18, 'pretrained_models/resnet-18-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', 'ir_csn', 18, 'pretrained_models/resnet-18-kinetics-ucf101_split1.pth', 'A',
                               16, 2),
                     ModelArch('resnet', 'ir_csn', 34, 'pretrained_models/resnet-34-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', 'ir_csn', 50, 'pretrained_models/resnet-50-kinetics.pth', 'B', 16, 2),

                     ModelArch('resnext', '3d', 101, 'pretrained_models/resnext-101-64f-kinetics-hmdb51_split1.pth',
                               'B', 64, 2)]

    for architecture in architectures:

        config.base_model = architecture.arch
        config.model_type = architecture.type
        config.model_depth = architecture.depth
        config.batch_size = 4
        if architecture.depth < 50:
            config.batch_size = 80
        if config.cuda_available:
            assert os.path.exists(architecture.path)
            config.pretrain_path = architecture.path
        else:
            config.pretrain_path = ''
        config.resnet_shortcut = architecture.shortcut
        config.sample_duration = architecture.n_frames
        config.finetune_block = architecture.fn_block
        config.learning_rate = 0.0004

        result_path = os.path.join('embeddings_results', f'{architecture.arch}-{architecture.depth}',
                                   f'{architecture.type}'
                                   f'fn_block_{architecture.fn_block}')
        os.makedirs(result_path, exist_ok=True)
        config.result_path = result_path
        try:
            train(config)
        except Exception as e:
            log.exception(f'Couldn\'t run experiment for model: {architecture}. Error: {e}')


def run_embeddings_experment(n_epochs=1, blocks=[2]):
    """
    Runs classification experiments to eval various defined architectures
    :return:
    """

    config = parse_opts()
    config.use_embeddings = True
    config.use_quadruplet = True
    config.dataset_path = '/Users/elizavetabatanina/Projects/gta/GTA_dataset'
    config.jpg_dataset_path = '/Users/elizavetabatanina/Projects/GTA_JPG_DATASET_new'
    if config.cuda_available:
        torch.cuda.set_device(config.cuda_id0)
        print(config.cuda_available)
        print(config.cuda_id0)
        print(torch.cuda.device_count())
        print(torch.cuda.is_available())
        print(torch.cuda.current_device())

    architectures = [ModelArch('resnet', '3d', 18, 'pretrained_models/resnet-18-kinetics.pth', 'A', 32, 2),
                     ModelArch('resnet', '3d', 18, 'pretrained_models/resnet-18-kinetics-ucf101_split1.pth', 'A', 16,
                               2),
                     ModelArch('resnet', '3d', 34, 'pretrained_models/resnet-34-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', '3d', 50, 'pretrained_models/resnet-50-kinetics.pth', 'B', 16, 2),
                     ModelArch('resnet', '3d', 101, 'pretrained_models/resnet-101-kinetics.pth', 'B', 16, 2),

                     ModelArch('resnet', 'ir_csn', 18, 'pretrained_models/resnet-18-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', 'ir_csn', 18, 'pretrained_models/resnet-18-kinetics-ucf101_split1.pth', 'A',
                               16, 2),
                     ModelArch('resnet', 'ir_csn', 34, 'pretrained_models/resnet-34-kinetics.pth', 'A', 16, 2),
                     ModelArch('resnet', 'ir_csn', 50, 'pretrained_models/resnet-50-kinetics.pth', 'B', 16, 2),

                     ModelArch('resnext', '3d', 101, 'pretrained_models/resnext-101-64f-kinetics-hmdb51_split1.pth',
                               'B', 64, 2)]

    for architecture in architectures:
        config.n_finetune_classes = 5
        config.base_model = architecture.arch
        config.model_type = architecture.type
        config.model_depth = architecture.depth
        config.batch_size = 4
        if architecture.depth < 50:
            config.batch_size = 80
        if config.cuda_available:
            assert os.path.exists(architecture.path)
            config.pretrain_path = architecture.path
        else:
            config.pretrain_path = ''
        config.resnet_shortcut = architecture.shortcut
        config.sample_duration = architecture.n_frames
        config.finetune_block = architecture.fn_block
        config.learning_rate = 0.0004

        result_path = os.path.join('embeddings_results', f'{architecture.arch}-{architecture.depth}',
                                   f'{architecture.type}'
                                   f'fn_block_{architecture.fn_block}')
        os.makedirs(result_path, exist_ok=True)
        config.result_path = result_path
        try:
            train(config)
        except Exception as e:
            log.exception(f'Couldn\'t run experiment for model: {architecture}. Error: {e}')

