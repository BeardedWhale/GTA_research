import os
from glob import glob
import json
from loguru import logger
import sys
import subprocess
import numpy as np
from tqdm import tqdm
from vidaug import augmentors as va
import math
import copy
from sklearn.model_selection import train_test_split, KFold
import functools
from collections import Counter

import torch
import pandas as pd
import torch.utils.data as data
from PIL import Image

DATASET_PATH = '../GTA_dataset'
JPG_PATH = '../GTA_JPG_DATASET'
logger.add(sys.stdout)


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def dataset_jpg(dataset_path, jpg_path):
    '''
    converts all files to jpg format
    '''
    if not os.path.exists(jpg_path):
        os.mkdir(jpg_path)
    inner_files = glob(os.path.join(dataset_path, '*'))
    dataset_folders = [name for name in inner_files if os.path.isdir(name) and os.path.split(name)[
        -1].istitle()]  # filter out files and non Title names
    for class_folder in dataset_folders:
        curr_class = os.path.split(class_folder)[-1]
        os.mkdir(os.path.join(jpg_path, curr_class))
        print(os.path.join(jpg_path, curr_class), 'created')
        scene_folders = glob(os.path.join(class_folder, '*'))
        scene_folders = [scene_folder for scene_folder in scene_folders if '.DS' not in scene_folder]
        for scene_folder in scene_folders:
            curr_scene = os.path.split(scene_folder)[-1]
            os.mkdir(os.path.join(jpg_path, curr_class, curr_scene))
            print(os.path.join(jpg_path, curr_class, curr_scene), 'created')
            scene_videos = glob(os.path.join(scene_folder, '*.mp4'))
            for video in scene_videos:
                # create folder
                video_name = os.path.split(video)[-1][:-4]
                dest_path = os.path.join(jpg_path, curr_class, curr_scene, video_name)
                os.mkdir(dest_path)
                print(dest_path, 'created')
                cmd = 'ffmpeg -i {} -vf  scale=-1:360  -qscale 1 {}/image_%05d.jpg'.format(video, dest_path)
                print(cmd)
                subprocess.call(cmd, shell=True)
                print('\n')
                print(video, 'done!')


def dataset_to_json(dataset_path, split_type=1):
    '''
    1. get all videos by class
    2. split scenes on train, test, validation
    3. write to one global json

    Dataset requirements:
    - Capital name for class folders

    :param dataset_path:
    :param split_type: 1 for train&test, 2 for train&test&validation
    :return:
    '''
    data = {}
    video_labels = {}
    scene_labels = {}
    scene_map = {}
    inner_files = glob(os.path.join(dataset_path, '*'))
    dataset_folders = [name for name in inner_files if os.path.isdir(name) and os.path.split(name)[
        -1].istitle()]  # filter out files and non Title names
    classes = [os.path.split(folder)[-1] for folder in dataset_folders]
    print(classes)
    class_map = dict(zip(classes, range(0, len(classes))))
    scene_id = 1
    for class_folder in dataset_folders:
        curr_class = os.path.split(class_folder)[-1]
        curr_class_id = class_map[curr_class]
        scene_folders = glob(os.path.join(class_folder, '*'))
        scene_folders = [scene_folder for scene_folder in scene_folders if '.DS' not in scene_folder]
        for scene_folder in scene_folders:
            scene_labels[scene_id] = curr_class_id
            scene_videos = glob(os.path.join(scene_folder, '*.mp4'))
            scene_map[scene_id] = scene_folder
            for video in scene_videos:
                video_labels[video.replace(dataset_path, '')] = (curr_class_id, scene_id)
            scene_id += 1
    X_train, X_test, y_train, y_test = train_test_split(
        list(scene_labels.keys()), list(scene_labels.values()), test_size=0.3, random_state=1)
    scene_labels = {'test': dict(zip(X_test, y_test))}

    if split_type == 2:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=1)
        scene_labels['val'] = dict(zip(X_val, y_val))
    scene_labels['train'] = dict(zip(X_train, y_train))

    data = {'video_labels': video_labels, 'scene_labels': scene_labels, 'scene_map': scene_map, 'class_map': class_map}
    print('done')
    with open(os.path.join(dataset_path, 'gta_dataset.json'), 'w+') as file:
        file.write(json.dumps(data))


def dataset_to_json_kfolds(dataset_path, split_type=1, k=4):
    '''
    1. get all videos by class
    2. split scenes on train, test, validation
    3. write to one global json

    Dataset requirements:
    - Capital name for class folders

    :param dataset_path:
    :param split_type: 1 for train&test, 2 for train&test&validation
    :return:
    '''
    data = {}
    video_labels = {}
    print('KEK')
    scene_labels = {}
    scene_map = {}
    inner_files = glob(os.path.join(dataset_path, '*'))
    dataset_folders = [name for name in inner_files if os.path.isdir(name) and os.path.split(name)[
        -1].istitle()]  # filter out files and non Title names
    classes = [os.path.split(folder)[-1] for folder in dataset_folders]
    print(classes)
    class_map = dict(zip(classes, range(0, len(classes))))
    scene_id = 1
    for class_folder in dataset_folders:
        curr_class = os.path.split(class_folder)[-1]
        curr_class_id = class_map[curr_class]
        scene_folders = glob(os.path.join(class_folder, '*'))
        scene_folders = [scene_folder for scene_folder in scene_folders if '.DS' not in scene_folder]
        for scene_folder in scene_folders:
            scene_labels[scene_id] = curr_class_id
            scene_videos = glob(os.path.join(scene_folder, '*.mp4'))
            scene_map[scene_id] = scene_folder
            for video in scene_videos:
                video_labels[video.replace(dataset_path, '')] = (curr_class_id, scene_id)
            scene_id += 1

    folds = KFold(k, shuffle=True).split(list(scene_labels.keys()), y=list(scene_labels.values()))
    folds_dict = []
    for i, f in enumerate(folds):
        print(f)
        f_train = f[0]
        f_test = f[1]
        x_train = np.array(list(scene_labels.keys()))[f[0]].tolist()
        y_train = np.array(list(scene_labels.values()))[f[0]].tolist()
        x_test = np.array(list(scene_labels.keys()))[f[1]].tolist()
        y_test = np.array(list(scene_labels.values()))[f[1]].tolist()
        folds_dict.append({'train': dict(zip(x_train, y_train)), 'test': dict(zip(x_test, y_test))})

    X_train, X_test, y_train, y_test = train_test_split(
        list(scene_labels.keys()), list(scene_labels.values()), test_size=0.3, random_state=1)
    scene_labels = {'test': dict(zip(X_test, y_test))}

    if split_type == 2:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.3, random_state=1)
        scene_labels['val'] = dict(zip(X_val, y_val))
    scene_labels['train'] = dict(zip(X_train, y_train))

    data = {'video_labels': video_labels, 'scene_labels': scene_labels,
            'scene_map': scene_map, 'class_map': class_map, 'scene_folds': folds_dict}
    print('done')
    print(folds_dict[0])
    with open(os.path.join(dataset_path, 'gta_dataset.json'), 'w+') as file:
        file.write(json.dumps(data))


def make_dataset(dataset_path, jpg_path, subset, n_samples_for_each_video=1, sample_duration=16, fold=1):
    '''
    creates dict of necessary dataset parameters
    :param dataset_path: path to original dataset with videos and json description
    :param jpg_path: path to folder with jpg version of dataset. If there is none, then create
    :param subset: 'train'/'test'/'val' (if use val, divide dataset on train, test, val using dataset_to_json)
    :return: dataset dict
    '''
    time_slots_df = pd.read_csv('time_slots.csv')

    def get_time_slot(video_name: str):
        class_name, id = video_name.split('/')[1:3]
        row = time_slots_df.loc[(time_slots_df['class name'] == class_name) & (time_slots_df['vid id'] == int(id))].iloc[
            0]
        start = row['start second']
        end = row['end second']
        return start, end

    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        logger.exception(f'Dataset Folder {dataset_path} not found or is not a directory')
        return {}, {}, {}, {}
    if not os.path.exists(jpg_path):
        logger.warning(f'Dataset jpg folder {jpg_path} not found. Creating ...')
        dataset_jpg(dataset_path, jpg_path)
    if not os.path.exists(os.path.join(dataset_path, 'gta_dataset.json')):
        logger.warning(
            f'Dataset json description {os.path.join(dataset_path, "gta_dataset.json")} not found. Creating...')
        dataset_to_json(dataset_path, 2)
    with open(os.path.join(dataset_path, 'gta_dataset.json')) as js:
        dataset_json = json.loads(js.read())
    dataset = []
    video_labels = dataset_json['video_labels']
    scene_labels = dataset_json['scene_folds'][fold]
    scene_map = dataset_json['scene_map']
    class_map = dataset_json['class_map']
    scene_labels_split = scene_labels.get(subset, {})
    if len(scene_labels_split) == 0:
        logger.warning(f'Empty {subset} split. Check your json file.')
        return {}, {}, {}, {}

    class_counter = Counter([])
    for video, (class_id, scene_id) in tqdm(video_labels.items(), total=len(video_labels.items())):
        time_slot = get_time_slot(video)  # TODO  multiply by 30
        sample = {}
        n_frames = 0
        if str(scene_id) not in scene_labels_split:  # not required split from: train, test,
            continue
        path_to_video_frames = os.path.join(jpg_path, video[1:-4])
        if not os.path.isdir(path_to_video_frames) or not os.path.exists(path_to_video_frames):
            logger.warning(f'Wrong path to sample frames: {path_to_video_frames}. Expected folder. Skipping ...')
            continue

        frame_indices = glob(os.path.join(path_to_video_frames, '*.jpg'))
        n_frames = len(frame_indices)

        begin_t = time_slot[0] * 30
        end_t = min(max(time_slot[1] * 30, begin_t + sample_duration), n_frames)
        if (end_t - begin_t + 1) < sample_duration:
            new_begin_t = begin_t - (sample_duration - (end_t - begin_t + 1))
            begin_t = max(1, new_begin_t)


        sample = {
            'video': path_to_video_frames,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video[:-4],
            'label': class_id,
            'scene': scene_id,
            'frame_indices': frame_indices,
        }
        class_counter.update([class_id for i in range(n_samples_for_each_video)])
        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)

        else:
            stride = 1
            n_possible_samples = n_frames - sample_duration + 1
            for k in range(min(n_samples_for_each_video, n_possible_samples)):
                sample_k = copy.deepcopy(sample)
                sample_k['frame_indices'] = list(range(k + 1, k+sample_duration+1))
                if len(sample_k['frame_indices']) < sample_duration:
                    raise ValueError(path_to_video_frames, k, begin_t, end_t, sample_k['frame_indices'], time_slot[1] * 30, n_frames)
                dataset.append(sample_k)
            # if n_samples_for_each_video > 1:
            #     step = max(1,
            #                math.ceil((n_frames - 1 - sample_duration) /
            #                          (n_samples_for_each_video - 1)))
            # else:
            #     step = sample_duration
            # for j in range(1, n_frames, step):
            #     sample_j = copy.deepcopy(sample)
            #     sample_j['frame_indices'] = list(
            #         range(j, min(n_frames + 1, j + sample_duration)))
            #     dataset.append(sample_j)
    logger.info(f'Loaded {subset} dataset')
    logger.info(f'Dataset size: {len(dataset)}')
    print(class_counter)
    # print(np.array(list(video_labels.values())))
    video_counter = Counter(np.array(list(video_labels.values()))[:, 0])
    print('DATASET REVIEW')
    print(class_map)
    reverse_class_map = dict(zip(class_map.values(), class_map.keys()))
    for label, c in video_counter.items():
        print(f'{reverse_class_map[label]}: {c}/{len(video_labels)}')

    return dataset, scene_labels, scene_map, class_map


class GTA_crime(data.Dataset):
    def __init__(self,
                 dataset_path,
                 jpg_path,
                 subset,
                 n_samples_for_each_video=15,
                 spatial_transform=None,
                 temporal_transform=None,
                 target_transform=None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        print('n samples:', n_samples_for_each_video)
        self.sample_duration = sample_duration
        self.subset = subset
        self.data, self.scene_labels, self.scene_map, self.class_map = make_dataset(dataset_path, jpg_path, subset,
                                                                                    n_samples_for_each_video,
                                                                                    sample_duration)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        sometimes = lambda aug: va.Sometimes(0.3, aug)  # Used to apply augmentor with 50% probability
        print(subset)
        if self.subset == 'train':
            self.seq = va.Sequential([
                va.RandomRotate(degrees=10),  # randomly rotates the video with a degree randomly choosen from [-10, 10]
                sometimes(va.HorizontalFlip()),  # horizontally flip the video with 50% probability
                sometimes(va.Pepper()),
                sometimes(va.Salt()),
                sometimes(va.RandomTranslate()),
                # sometimes(va.RandomShear()),
                sometimes(va.GaussianBlur(sigma=1)),
                sometimes(va.ElasticTransformation()),
                va.TemporalFit(sample_duration)
            ])
        else:
            self.seq = va.Sequential([va.TemporalFit(sample_duration)])
        self.loader = get_loader()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        sample = self.data[index]
        path = sample['video']
        # print(path)
        frame_indices = sample['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)

        if len(clip) == 0:
            logger.warning(f'Empty clip list: {path}, {frame_indices}')

        clip = self.seq(clip)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]

        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        scene_label = sample['scene']
        return clip, target, scene_label

    def __len__(self):
        return len(self.data)
