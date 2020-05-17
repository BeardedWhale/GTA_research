import torch
from torch import nn
import os
from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet
import copy
RESNET_DEPTHS = [10, 18, 34, 50, 101, 152,
                 200]

WIDERESNET_DEPTHS = [50]

RESNEXT_DEPTHS = [50, 101, 152]

DENSENET_DEPTHS = [121, 169, 201, 264]

PRERESNET_DEPTHS = [18, 34, 50, 101, 152, 200]

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

def load_pretrained(model, model_name, model_depth, pretrain_path):
    new_model = copy.deepcopy(model)
    assert os.path.exists(pretrain_path), f'Path to train model {pretrain_path} does not exist'
    pretrain_model = torch.load(pretrain_path)
    arch = f'{model_name}-{model_depth}'
    print()
    assert arch == pretrain_model['arch'], f'Incorrectly loaded: {pretrain_model["arch"]} for created model: {arch}'

    new_model.load_state_dict(pretrain_model['state_dict'])

    return new_model
    # if model_name == 'densenet':
    #     model.module.classifier = nn.Linear(
    #         base_model.module.classifier.in_features, config.n_finetune_classes)
    #     base_model.module.classifier = base_model.module.classifier.cuda(device=config.cuda_id)  # TODO THIS

    # no cuda case
    #  if config.model == 'densenet':
    #                 base_model.classifier = nn.Linear(
    #                     base_model.classifier.in_features, config.n_finetune_classes)
    #             else:
    #                 base_model.fc = nn.Linear(base_model.fc.in_features,
    #                                      config.n_finetune_classes)



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


    if  config.cuda_available:
        import os
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.cuda_id}'
        base_model = nn.DataParallel(base_model, device_ids=[0])  # TODO THIS
        # model = model.cuda(device=opt.cuda_id) # TODO THIS
        if config.pretrain_path:
            base_model = load_pretrained(base_model, config.base_model, config.model_depth, config.pretrain_path)

            # elif opt.use_quadriplet:
            #     model = EmbeddingModel(model, opt.n_finetune_classes, not opt.no_cuda, opt.cuda_id)
            # else:
            #     base_model.module.fc = nn.Sequential(nn.Dropout(0.4),
            #                                     nn.Linear(base_model.module.fc.in_features,
            #                                               512),
            #                                     nn.ReLU6(),
            #                                     nn.Dropout(0.4),
            #                                     nn.Linear(512, 256),
            #                                     nn.ReLU6(),
            #                                     nn.Dropout(0.4),
            #                                     nn.Linear(256, config.n_finetune_classes)).cuda(device=config.cuda_id)
            #     # model.module.fc = nn.Linear(model.module.fc.in_features,
            #     #                             opt.n_finetune_classes)
            #
            #     # model.module.fc = model.module.fc.cuda(device=opt.cuda_id)
            # # model = nn.DataParallel(model, device_ids=[0, 1])
            base_model = base_model.cuda(device=config.cuda_id)
            parameters = get_fine_tuning_parameters(base_model, config.ft_begin_index)
            print(len(list(parameters)), 'params to fine tune', config.ft_begin_index)

            # model = nn.DataParallel(model, device_ids=[0, 1])
            print('Device:', base_model.output_device, base_model.device_ids)
            return base_model, parameters
    else:
        if config.pretrain_path:
            base_model = load_pretrained(base_model, config.base_model, config.model_depth, config.pretrain_path)


            parameters = get_fine_tuning_parameters(base_model, config.ft_begin_index)

            return base_model, parameters

    return base_model, base_model.parameters()


class FineTuneBlock1(nn.Module):
    def __init__(self, input_size, cuda=True, cuda_ids=[0]):
        kek = 0

class EmbeddingModel(nn.Module):
    def __init__(self, model, n_finetune_classes, cuda=True, cuda_id=0):
        super(EmbeddingModel, self).__init__()
        print(model)
        model.module.fc = nn.Linear(model.module.fc.in_features,
                                    512)
        self.model = model
        self.classifier = nn.Sequential(nn.Dropout(0.4),
                                        nn.Linear(512,
                                                  512),

                                        nn.ReLU6(),
                                        nn.Dropout(0.4),
                                        nn.Linear(512,
                                                  128),
                                        nn.ReLU6(),
                                        nn.Linear(128, n_finetune_classes))
        if cuda:
            print('CUDA')
            self.classifier = self.classifier.cuda(device=cuda_id)
            self.model = self.model.cuda(device=cuda_id)

    def forward(self, x):
        embedding = self.model(x)
        print(embedding.shape)
        y = self.classifier(embedding)

        return embedding, y
