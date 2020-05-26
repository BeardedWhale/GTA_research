from typing import Dict, List

import torchnet
from utils import STEPS
from comet_ml import Experiment
import numpy as np
import torch
from sklearn.metrics import confusion_matrix




class PerClassAcc(object):

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.classes_count_d = dict(zip(range(n_classes), [(0, 0) for i in range(n_classes)]))

    def update(self, new_count_d):
        for class_idx in range(self.n_classes):
            new_total, new_predicted = new_count_d[class_idx]
            old_total, old_predicted = self.classes_count_d[class_idx]
            self.classes_count_d[class_idx] = (old_total + new_total), (old_predicted + new_predicted)


class APMeter:
    """
    Measures average precision per class
    """

    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.total_outputs = np.empty((0, n_classes))
        self.total_targets = np.empty((0))

    def add(self, outputs: [np.ndarray, torch.tensor], target: [np.ndarray, torch.tensor]):
        """
        Accepts
        :param outputs: b x N_classes matrix
        :param target: vector of length b
        :return:
        """
        assert np.ndim(outputs) == 2, \
            "Output should be a matrix of prob distribution for each sample"
        assert np.ndim(target) == 1, \
            "Target should be a vector of labels"
        assert outputs.shape[0] == target.shape[0], \
            f"Output should have same number of samples. Expected: {target.shape[0]}"

        self.total_outputs = np.concatenate([self.total_outputs, outputs])
        if self.total_targets.size == 0:
            self.total_targets = target
        else:
            self.total_targets = np.concatenate([self.total_targets, target])

    def __average_accuracy__(self) -> np.ndarray:
        output_labels = np.argmax(self.total_outputs, axis=1)
        cm = confusion_matrix(self.total_targets, output_labels, normalize='true')
        return cm.diagonal()

    def value(self) -> np.ndarray:
        return self.__average_accuracy__()

    def reset(self):
        self.total_outputs = np.empty((0, self.n_classes))
        self.total_targets = np.empty((0))


class Logger:
    def __init__(self, experiment: Experiment, step: STEPS, n_classes=2, topk: [int] = [1], class_map: List[str] = [],
                 metrics=[]):
        """

        :param experiment:
        :param step:
        :param n_classes:
        :param topk:
        :param class_map:
        :param metrics:
        """
        assert len(class_map) == n_classes, \
            f"Class map: {class_map} length is {len(class_map)} and is not equal to n_classes: {n_classes}"
        assert len(topk) != 0, \
            f"topk param should be not empty"
        topk = np.sort(topk)
        assert topk[-1] <= n_classes, \
            f"topk max class is {topk[-1]} which is greater than maximum class number: {n_classes}"

        self.n_classes = n_classes
        self.class_map = class_map
        self.step = step
        self.topk = topk
        self.experiment = experiment
        self.curr_state = {}
        self.metrics = ['ACC', 'LOSS'] + metrics
        self.accuracy_meter = torchnet.meter.ClassErrorMeter(topk=topk, accuracy=True)  # accepts probs + labels
        self.ap_accuracy_meter = APMeter(n_classes)
        self.loss_meter = torchnet.meter.AverageValueMeter()
        self.other_meters: Dict[str, torchnet.meter.AverageValueMeter] = \
            {metric_name: torchnet.meter.AverageValueMeter() for metric_name in metrics}

    def update(self, loss, outputs, target, **kwargs):
        """
        Updates logger state
        :param loss:
        :param outputs:
        :param target: vector of size b
        :return:
        """
        assert np.ndim(outputs) == 2, \
            "Output should be a matrix of prob distribution for each sample"
        assert np.ndim(target) == 1, \
            "Target should be a vector of labels"
        assert outputs.shape[0] == target.shape[0], \
            f"Output should have same number of samples. Expected: {target.shape[0]}"

        self.accuracy_meter.add(outputs, target)
        self.ap_accuracy_meter.add(outputs, target)
        self.loss_meter.add(loss)
        for k, v in kwargs.items():
            if k in self.other_meters:
                self.other_meters[k].add(v)
        self.__update_curr_state__('BATCH')
        for name, value in self.curr_state.items():
            self.experiment.log_metric(name, value)

    def update_epoch(self, epoch: int):
        """
        Updates epoch logging
        :param epoch: epoch number
        :return:
        """
        all_targets = targets_to_one_hot(self.ap_accuracy_meter.total_targets, self.n_classes)
        all_outputs = self.ap_accuracy_meter.total_outputs
        self.experiment.log_confusion_matrix(all_targets, all_outputs, title=f'{self.step.value} matrix EPOCH {epoch}',
                                             step=epoch,
                                             file_name=f"{self.step.value}-confusion-matrix-{epoch}.json")
        self.__update_curr_state__('EPOCH')
        for name, value in self.curr_state.items():
            self.experiment.log_metric(name, value)

    def __update_curr_state__(self, identifier):
        for metric in self.metrics:
            if metric == 'ACC':
                for k in self.topk:
                    self.curr_state[f'{identifier}_{self.step.value}_ACC_TOP_{k}'] = self.accuracy_meter.value(k)
                per_class_acc = self.ap_accuracy_meter.value()
                for class_id in range(self.n_classes):
                    self.curr_state[f'{identifier}_{self.step.value}_ACC_{self.class_map[class_id]}'] = per_class_acc[
                        class_id]

            elif metric == 'LOSS':
                self.curr_state[f'{identifier}_{self.step.value}_LOSS'] = self.loss_meter.value()[0]
            else:
                meter = self.other_meters[metric]
                self.curr_state[f'{identifier}_{self.step.value}_{metric}'] = meter.value()[0]

    def reset(self):
        self.accuracy_meter.reset()
        self.ap_accuracy_meter.reset()
        self.loss_meter.reset()
        for metric, meter in self.other_meters.items():
            meter.reset()
        self.curr_state = {}

    def state_message(self) -> str:
        message = ''
        for k, v in self.curr_state.items():
            message += f' {k}: {v} |'
        return message


def targets_to_one_hot(targets, n_classes) -> torch.tensor:
    # [N, 1] ->[N, n_classes]
    N = targets.shape[0]
    result = torch.zeros((N, n_classes))
    for i in range(N):
        class_id = targets[i]
        result[i][class_id] = 1
    return result
