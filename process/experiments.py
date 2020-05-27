from collections import namedtuple

from conf import parse_opts
from train import train
import os
from loguru import logger as log

ModelArch = namedtuple('ModelArchitecture', 'arch type depth path shortcut n_frames fn_block ')


def run_classification_experment(n_epochs=3, blocks=[2]):
    """
    Runs classification experiments to eval various defined architectures
    :return:
    """

    config = parse_opts()
    config.n_epochs = n_epochs

    architectures = [ModelArch('resnet', '3d', 18, 'pretrained_models/resnet-18-kinetics.pth', 'A', 16, 2),
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
        assert os.path.exists(architecture.path)
        config.base_model = architecture.arch
        config.model_type = architecture.type
        config.model_depth = architecture.depth
        config.pretrain_path = architecture.path
        config.resnet_shortcut = architecture.shortcut
        config.sample_duration = architecture.n_frames
        config.finetune_block = architecture.fn_block
        config.learning_rate = 0.0004
        try:
            train(config)
        except Exception as e:
            log.exception(f'Couldn\'t run experiment for model: {architecture}. Error: {e}')

    # learning_rates = [0.005, 0.001,  0.0001]
    # blocks = [1, 2]
    # fn_begin_indexes = [4, 5]
    # train(config)
