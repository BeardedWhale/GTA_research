import torch
from torch import nn

from models import resnet, pre_act_resnet, wide_resnet, resnext, densenet


def generate_model(config):
    assert config.model in [
        'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet'
    ]

    if config.model == 'resnet':
        assert config.model_depth in [10, 18, 34, 50, 101, 152, 200]

        from models.resnet import get_fine_tuning_parameters

        if config.model_depth == 10:
            model = resnet.resnet10(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 18:
            model = resnet.resnet18(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 34:
            model = resnet.resnet34(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 50:
            model = resnet.resnet50(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration,
                model_type=config.model_type)
        elif config.model_depth == 101:
            model = resnet.resnet101(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration,
                model_type=config.model_type)
        elif config.model_depth == 152:
            model = resnet.resnet152(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration,
                model_type=config.model_type)
        elif config.model_depth == 200:
            model = resnet.resnet200(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration,
                model_type=config.model_type)
    elif config.model == 'wideresnet':
        assert config.model_depth in [50]

        from models.wide_resnet import get_fine_tuning_parameters

        if config.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                k=config.wide_resnet_k,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
    elif config.model == 'resnext':
        assert config.model_depth in [50, 101, 152]

        from models.resnext import get_fine_tuning_parameters

        if config.model_depth == 50:
            model = resnext.resnet50(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 101:
            model = resnext.resnet101(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 152:
            model = resnext.resnet152(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
    elif config.model == 'preresnet':
        assert config.model_depth in [18, 34, 50, 101, 152, 200]

        from models.pre_act_resnet import get_fine_tuning_parameters

        if config.model_depth == 18:
            model = pre_act_resnet.resnet18(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 34:
            model = pre_act_resnet.resnet34(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 50:
            model = pre_act_resnet.resnet50(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 101:
            model = pre_act_resnet.resnet101(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 152:
            model = pre_act_resnet.resnet152(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 200:
            model = pre_act_resnet.resnet200(
                num_classes=config.n_classes,
                shortcut_type=config.resnet_shortcut,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
    elif config.model == 'densenet':
        assert config.model_depth in [121, 169, 201, 264]

        from models.densenet import get_fine_tuning_parameters

        if config.model_depth == 121:
            model = densenet.densenet121(
                num_classes=config.n_classes,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 169:
            model = densenet.densenet169(
                num_classes=config.n_classes,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 201:
            model = densenet.densenet201(
                num_classes=config.n_classes,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 264:
            model = densenet.densenet264(
                num_classes=config.n_classes,
                sample_size=config.sample_size,
                sample_duration=config.sample_duration)

    if not config.no_cuda:
        import os
        # os.environ['CUDA_VISIBLE_DEVICES'] = f'{opt.cuda_id}'
        model = nn.DataParallel(model, device_ids=[0]) # TODO THIS
        # model = model.cuda(device=opt.cuda_id) # TODO THIS


        if config.pretrain_path:
            print('loading pretrained model {}'.format(config.pretrain_path))
            pretrain = torch.load(config.pretrain_path)
            print(pretrain['arch'])
            arch = f'{config.model}-{config.model_depth}'
            # arch = opt.model + '-' + opt.model_depth
            print(arch)
            assert arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if config.model == 'densenet':
                model.module.classifier = nn.Linear(
                    model.module.classifier.in_features, config.n_finetune_classes)
                model.module.classifier = model.module.classifier.cuda(device=config.cuda_id) # TODO THIS
            # elif opt.use_quadriplet:
            #     model = EmbeddingModel(model, opt.n_finetune_classes, not opt.no_cuda, opt.cuda_id)
            else:
                model.module.fc = nn.Sequential(nn.Dropout(0.4),
                                                nn.Linear(model.module.fc.in_features,
                                                           512),
                                                nn.ReLU6(),
                                                nn.Dropout(0.4),
                                                nn.Linear(512, 256),
                                                nn.ReLU6(),
                                                nn.Dropout(0.4),
                                                nn.Linear(256, config.n_finetune_classes)).cuda(device=config.cuda_id)
                # model.module.fc = nn.Linear(model.module.fc.in_features,
                #                             opt.n_finetune_classes)

                # model.module.fc = model.module.fc.cuda(device=opt.cuda_id)
            # model = nn.DataParallel(model, device_ids=[0, 1])
            model = model.cuda(device=config.cuda_id)
            parameters = get_fine_tuning_parameters(model, config.ft_begin_index)
            print(len(list(parameters)), 'params to fine tune', config.ft_begin_index)

            # model = nn.DataParallel(model, device_ids=[0, 1])
            print('Device:', model.output_device, model.device_ids)
            return model, parameters
    else:
        if config.pretrain_path:
            print('loading pretrained model {}'.format(config.pretrain_path))
            pretrain = torch.load(config.pretrain_path)
            assert config.arch == pretrain['arch']

            model.load_state_dict(pretrain['state_dict'])

            if config.model == 'densenet':
                model.classifier = nn.Linear(
                    model.classifier.in_features, config.n_finetune_classes)
            else:
                model.fc = nn.Linear(model.fc.in_features,
                                     config.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, config.ft_begin_index)

            return model, parameters

    return model, model.parameters()

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
            self.model  = self.model.cuda(device=cuda_id)


    def forward(self, x):
        embedding = self.model(x)
        print(embedding.shape)
        y = self.classifier(embedding)

        return embedding, y



