import argparse


def parse_opts():
    parser = argparse.ArgumentParser()
    # MODEL CONF
    parser.add_argument(
        '--base_model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_type',
        default='ip_csn',
        type=str,
        help='(3d | ir_csn | ip_csn)',
        choices=['3d', 'ir_csn', 'ip_csn']
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=8,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    parser.add_argument(
        '--finetune_block',
        default=2,
        type=int,
        help=
        'Finetune block to use',
        choices=[1, 2]
    )
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')
    parser.add_argument(
        '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    parser.add_argument(
        '--resnext_cardinality',
        default=32,
        type=int,
        help='ResNeXt cardinality')

    # PATHS
    parser.add_argument(
        '--root_path',
        default='',
        type=str,
        help='Root directory path of data')
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument(
        '--dataset_path',
        default='../GTA_dataset',
        type=str,
        help='Directory path of Videos')
    parser.add_argument(
        '--dataset_conf_path',
        default='',
        type=str,
        help='File for dataset config. Default: class_map.yaml')
    parser.add_argument(
        '--jpg_dataset_path',
        default='../GTA_JPG_DATASET',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='resnext-101',
        type=str,
        help='Result directory path')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path', default='', type=str, help='Pretrained model (.pth)')
    parser.add_argument(
        '--checkpoint',
        default=1,
        type=int,
        help='Trained model is saved at every this epochs.')

    parser.add_argument(
        '--finetune_dropout',
        default=0.3,
        type=float,
        help=
        'Dropout rate for finetuning block'
    )
    parser.add_argument(
        '--use_batch_norm',
        default=False,
        type=bool,
        help=
        'If use batchnorm in funetune block or not'
    )

    # CLASSES CONF
    parser.add_argument(
        '--n_classes',
        default=400,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51, GTA: 8)'
    )



    # SAMPLE PARAMS
    parser.add_argument(
        '--sample_size',
        default=112,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=32,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--initial_scale',
        default=1.0,
        type=float,
        help='Initial scale for multiscale cropping')
    parser.add_argument(
        '--n_scales',
        default=2,
        type=int,
        help='Number of scales for multiscale cropping')
    parser.add_argument(
        '--scale_step',
        default=0.84089641525,
        type=float,
        help='Scale step for multiscale cropping')
    parser.add_argument(
        '--train_crop',
        default='center',
        type=str,
        help=
        'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    )
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument(
        '--norm_value',
        default=1,
        type=int,
        help=
        'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')

    # OPTIMIZATION PARAMS
    parser.add_argument(
        '--learning_rate',
        default=0.0002,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0, type=float, help='Momentum')  # 0.9
    parser.add_argument(
        '--dampening', default=0, type=float, help='dampening of SGD')  # 0.9
    parser.add_argument(
        '--weight_decay', default=0, type=float, help='Weight Decay')  # 1e-3
    parser.add_argument(
        '--mean_dataset',
        default='gta',
        type=str,
        help=
        'dataset for mean values of mean subtraction (activitynet | kinetics)')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--std_norm',
        action='store_false',
        help='If true, inputs are normalized by standard deviation.')
    parser.set_defaults(std_norm=True)
    parser.add_argument(
        '--nesterov', action='store_true', help='Nesterov momentum')
    parser.set_defaults(nesterov=False)
    parser.add_argument(
        '--optimizer',
        default='adam',
        type=str,
        help='Currently only support SGD')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )

    # TRAINING PARAMS
    parser.add_argument(
        '--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=2,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )
    parser.add_argument(
        '--use_embeddings',
        default=False,
        type=bool,
        help='if use embeddings model or not'
    )
    parser.add_argument(
        '--use_quadruplet',
        default=False,
        type=bool,
        help='if use quadruplet loss'
    )
    parser.add_argument(
        '--quadruplet_alpha',
        default=0.5,
        type=float,
        help='quadruplet loss weight'
    )
    parser.add_argument(
        '--quadruplet_beta',
        default=0.4,
        type=float,
        help='quadruplet loss weight'
    )
    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')

    parser.add_argument(
        '--ft_begin_index',
        default=4,
        type=int,
        help='Begin block index of fine-tuning')
    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--no_val',
        action='store_true',
        help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)

    # TEST PARAMS
    parser.add_argument(
        '--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--scale_in_test',
        default=1.0,
        type=float,
        help='Spatial scale in test')
    parser.add_argument(
        '--crop_position_in_test',
        default='c',
        type=str,
        help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')

    # CUDA
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--cuda_available', action='store_true', help='If true, cuda is used.', default=True)
    parser.add_argument(
        '--cuda_id0', default=0, type=int, help='0 or 1 or other number for cuda device'
    )
    parser.add_argument('--cuda_id1', default=-1, type=int, help='0 or 1 or other number for cuda device, -1 if second GPU is not available')
    parser.set_defaults(cuda_available=False)
    parser.add_argument(
        '--n_threads',
        default=0,
        type=int,
        help='Number of threads for multi-thread loading')

    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed')

    args = parser.parse_args()

    return args
