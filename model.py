from time import sleep

import torch
from torch import nn
import os
from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet
import copy


# https://pytorch.org/hub/pytorch_vision_resnext/

RESNET_DEPTHS = [10, 18, 34, 50, 101, 152,
                 200]

WIDERESNET_DEPTHS = [50]

RESNEXT_DEPTHS = [50, 101, 152]

DENSENET_DEPTHS = [121, 169, 201, 264]

PRERESNET_DEPTHS = [18, 34, 50, 101, 152, 200]


def generate_model(config):
    assert config.base_model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if config.base_model == 'resnet':

        from models.resnet import get_fine_tuning_parameters
        create_resnet_model = get_resnet_model(config.model_depth)
        base_model = create_resnet_model(
            num_classes=config.n_classes,
            shortcut_type=config.resnet_shortcut,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration
        )

    elif config.base_model == 'wideresnet':
        assert config.model_depth in [50], f'Wideresnet model depth {config.model_depth} is not valid.' \
                                           f' Acceptable: {WIDERESNET_DEPTHS}'

        from models.wide_resnet import get_fine_tuning_parameters

        if config.model_depth == 50:
            base_model = wide_resnet.resnet50(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                k=config.wide_resnet_k,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)

    elif config.base_model == 'resnext':

        from models.resnext import get_fine_tuning_parameters
        create_resnext_model = get_resnext_model(config.model_depth)
        base_model = create_resnext_model(
            num_classes=config.n_classes,
            shortcut_type=config.resnet_shortcut,
            cardinality=config.resnext_cardinality,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration
        )

    elif config.base_model == 'preresnet':

        from models.pre_act_resnet import get_fine_tuning_parameters
        create_preresnet_model = get_preresnet_model(config.model_depth)
        base_model = create_preresnet_model(
            num_classes=config.n_classes,
            shortcut_type=config.resnet_shortcut,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration
        )

    elif config.base_model == 'densenet':
        from models.densenet import get_fine_tuning_parameters
        create_densenet_model = get_densenet_model(config.model_depth)
        base_model = create_densenet_model(
            num_classes=config.n_classes,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration)

    if config.cuda_available:
        if config.cuda_id1 != -1:
            base_model = nn.DataParallel(base_model, device_ids=[config.cuda_id0, config.cuda_id1])
        else:
            base_model = nn.DataParallel(base_model, device_ids=[config.cuda_id0])
    if config.pretrain_path:
        base_model = load_pretrained(base_model, config.base_model, config.model_depth, config.pretrain_path)

    if config.use_embeddings:
        model = EmbeddingModel(base_model, config)
    else:
        model = update_with_finetune_block(base_model, config)
    parameters = get_fine_tuning_parameters(model,
                                            config.ft_begin_index)  # should not fail due to assert in the beginning

    return model, parameters


class FineTuneBlock1(nn.Module):
    def __init__(self, input_size, n_finetune_classes, dropout_rate=0.3,
                 use_batch_norm: bool = True):
        super(FineTuneBlock1, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, n_finetune_classes)
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(128)

    def forward(self, x):
        x_prime = nn.Dropout(self.dropout_rate)(nn.ReLU()(self.fc1(x)))
        if self.use_batch_norm:
            x_prime = self.bn1(x_prime)
        x_prime = nn.ReLU()(self.fc2(x_prime))
        if self.use_batch_norm:
            x_prime = self.bn2(x_prime)
        classes = self.fc3(x_prime)
        return classes


class FineTuneBlock2(nn.Module):
    def __init__(self, input_size, n_finetune_classes, dropout_rate=0.3,
                 use_batch_norm: bool = True):
        super(FineTuneBlock2, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256 + input_size, 128)
        self.fc4 = nn.Linear(128, n_finetune_classes)
        self.dropout_rate = dropout_rate

        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm1d(512)  # todo add this param to conf
            self.bn2 = nn.BatchNorm1d(256)
            self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x_prime = nn.Dropout(self.dropout_rate)(nn.ReLU()(self.fc1(x)))
        if self.use_batch_norm:
            x_prime = self.bn1(x_prime)
        x_prime = nn.Dropout(self.dropout_rate)(nn.ReLU()(self.fc2(x_prime)))
        if self.use_batch_norm:
            x_prime = self.bn2(x_prime)

        residual = torch.cat([x_prime, x], dim=1)
        x_prime = nn.ReLU()(self.fc3(residual))
        if self.use_batch_norm:
            x_prime = self.bn3(x_prime)
        classes = self.fc4(x_prime)
        return classes


class EmbeddingModel(nn.Module):
    def __init__(self, base_model, config):
        """
        Creates model that has two parts: encoder and classifier. Encoder returns embeddings of sample and classifier returns class probs
        :param base_model:
        :param config:
        :param cuda:
        :param cuda_id:
        """
        super(EmbeddingModel, self).__init__()

        cuda = config.cuda_available
        if cuda:
            device = torch.device(f"cuda:{config.cuda_id0}")
        else:
            device = torch.device('cpu')
        in_features = base_model.module.fc.in_features if cuda else base_model.fc.in_features  # todo handle densenet
        block_n_features = 512

        fc = nn.Sequential(nn.Dropout(0.4),
                           nn.Linear(in_features,
                                     block_n_features),
                           nn.ReLU6())
        # TODO handle densenet
        if cuda:
            base_model.module.fc = fc
        else:
            base_model.fc = fc
        self.encoder = base_model
        self.classifier = get_block(block_n_features, config)
        if cuda:  # TODO
            self.classifier = self.classifier.to(device)

    def forward(self, x):
        embedding = self.encoder(x)
        y = self.classifier(embedding)
        return embedding, y


def get_resnet_model(model_depth):
    assert model_depth in RESNET_DEPTHS, f'Resnet model depth {model_depth} is not valid.' \
                                         f' Acceptable: {RESNET_DEPTHS}'
    if model_depth == 10:
        return resnet.resnet10
    if model_depth == 18:
        return resnet.resnet18
    if model_depth == 34:
        return resnet.resnet34
    if model_depth == 50:
        return resnet.resnet50
    if model_depth == 101:
        return resnet.resnet101
    if model_depth == 152:
        return resnet.resnet152
    if model_depth == 200:
        return resnet.resnet200


def get_resnext_model(model_depth):
    assert model_depth in RESNEXT_DEPTHS, f'Resnext model depth {model_depth} is not valid.' \
                                          f' Acceptable: {RESNEXT_DEPTHS}'
    if model_depth == 50:
        return resnext.resnet50
    if model_depth == 101:
        return resnext.resnet101
    if model_depth == 152:
        return resnext.resnet152


def get_preresnet_model(model_depth):
    assert model_depth in PRERESNET_DEPTHS, f'Preresnet model depth {model_depth} is not valid.' \
                                            f' Acceptable: {PRERESNET_DEPTHS}'

    if model_depth == 18:
        return pre_act_resnet.resnet18
    if model_depth == 34:
        return pre_act_resnet.resnet34
    if model_depth == 50:
        return pre_act_resnet.resnet50
    if model_depth == 101:
        return pre_act_resnet.resnet101
    if model_depth == 152:
        return pre_act_resnet.resnet152
    if model_depth == 200:
        return pre_act_resnet.resnet200


def get_densenet_model(model_depth):
    assert model_depth in DENSENET_DEPTHS, f'Densenet model depth {model_depth} is not valid.' \
                                           f' Acceptable: {DENSENET_DEPTHS}'
    if model_depth == 121:
        return densenet.densenet121
    if model_depth == 169:
        return densenet.densenet169
    if model_depth == 201:
        return densenet.densenet201
    if model_depth == 264:
        return densenet.densenet264


def get_block(in_features: int, config) -> nn.Module:
    FineTuneBlock = FineTuneBlock1 if config.finetune_block == 1 else FineTuneBlock2
    block = FineTuneBlock(in_features, config.n_finetune_classes,
                          use_batch_norm=config.use_batch_norm,
                          dropout_rate=config.finetune_dropout)
    return block


def update_with_finetune_block(base_model, config):  # TODO fix .module is used only for gpus dataparallel!!
    if config.cuda_available:
        device = torch.device(f"cuda:{config.cuda_id0}")
    else:
        device = torch.device('cpu')

    if config.base_model == 'densenet':
        in_features = base_model.module.classifier.in_features
        block = get_block(in_features, config)
        block = block.to(device)
        base_model.module.classifier = block
    else:
        in_features = base_model.module.fc.in_features
        block = get_block(in_features, config)
        block = block.to(device)
        base_model.module.fc = block

    return base_model


def load_pretrained(model, model_name, model_depth, pretrain_path):
    new_model = copy.deepcopy(model)
    assert os.path.exists(pretrain_path), f'Path to train model {pretrain_path} does not exist'
    pretrain_model = torch.load(pretrain_path)
    arch = f'{model_name}-{model_depth}'
    assert arch == pretrain_model['arch'], f'Incorrectly loaded: {pretrain_model["arch"]} for created model: {arch}'

    new_model.load_state_dict(pretrain_model['state_dict'])

    return new_model
