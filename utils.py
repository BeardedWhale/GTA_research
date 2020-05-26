import torch
from enum import  Enum

class STEPS(Enum):
    TRAIN = 'TRAIN'
    VAL = 'VAL'
    TEST = 'TEST'

def targets_to_one_hot(targets, n_classes) -> torch.tensor:
    # [N, 1] ->[N, n_classes]
    N = targets.shape[0]
    result = torch.zeros((N, n_classes))
    for i in range(N):
        class_id = targets[i]
        result[i][class_id] = 1
    return result


def per_class_accuracies(output, target, n_classes=8):
    classes_count_d = dict(zip(range(n_classes), [(0, 0) for i in range(n_classes)]))
    output_classes = torch.argmax(output, dim=1)
    for class_idx in classes_count_d.keys():
        class_mask = (target == class_idx).nonzero()
        if class_mask.shape[0] > 1:
            class_mask = class_mask.squeeze()

        true_output_class = len((output_classes[class_mask] == class_idx).nonzero())

        classes_count_d[class_idx] = (len(class_mask), true_output_class)
    return classes_count_d
