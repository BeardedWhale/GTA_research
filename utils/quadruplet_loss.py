import torch
import numpy as np
from collections import  Counter

def _pairwise_distances(embeddings, squared=False):
    # Get the dot product between all embeddings
    # shape (batch_size, batch_size)
    dot_product = torch.matmul(embeddings, torch.transpose(embeddings, 0, 1))

    # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
    # This also provides more numerical stability (the diagonal of the result will be exactly 0).
    # shape (batch_size,)
    square_norm = torch.diag(dot_product)

    # Compute L2
    # shape (batch_size, batch_size)
    distances = torch.unsqueeze(square_norm, 1) - 2.0 * dot_product + torch.unsqueeze(square_norm, 0)

    # Because of computation errors, some distances might be negative so we put everything >= 0.0
    distances = torch.max(distances, torch.zeros_like(distances))

    if not squared:
        mask = torch._cast_Float(torch.eq(distances, 0.0))
        distances = distances + mask * 1e-16  # for sqrt(0)

        distances = torch.sqrt(distances)

        # Correct the epsilon added: set the distances on the mask to be exactly 0.0
        distances = distances * (1.0 - mask)

    return distances.cpu()


def _get_anchor_positive_triplet_mask(labels):
    labels = labels.cpu()
    # Check that i and j are distinct
    indices_equal = torch.eye(labels.shape[0]).type(torch.BoolTensor)
    indices_not_equal = torch.logical_not(indices_equal)
    # Check if labels[i] == labels[j]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    # Combine the two masks
    mask = indices_not_equal & labels_equal

    return mask


def _get_anchor_negative_triplet_mask(labels):
    # Check if labels[i] != labels[k]
    # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
    labels = labels.cpu()
    labels_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    mask = torch.logical_not(labels_equal)

    return mask


def batch_hard_triplet_loss(labels, embeddings,  margin=0.3, squared=False):
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)

    # For each anchor, get the hardest positive
    # First, we need to get a mask for every valid positive (they should have same label)
    mask_anchor_positive = _get_anchor_positive_triplet_mask(labels)
    mask_anchor_positive = torch._cast_Float(mask_anchor_positive)

    # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
    anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_dist)

    # shape (batch_size, 1)
    hardest_positive_dist, _ = torch.max(anchor_positive_dist, dim=1, keepdim=True)
    # torch.summary.scalar("hardest_positive_dist", torch.mean(hardest_positive_dist))

    # For each anchor, get the hardest negative
    # First, we need to get a mask for every valid negative (they should have different labels)
    mask_anchor_negative = _get_anchor_negative_triplet_mask(labels)
    mask_anchor_negative = torch._cast_Float(mask_anchor_negative)

    # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
    max_anchor_negative_dist, _ = torch.max(pairwise_dist, dim=1, keepdim=True)
    anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            torch.ones_like(mask_anchor_negative) - mask_anchor_negative)

    # shape (batch_size,)
    hardest_negative_dist, _ = torch.min(anchor_negative_dist, dim=1, keepdim=True)

    # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
    triplet_loss = torch.max(hardest_positive_dist - hardest_negative_dist + margin,
                             torch.zeros_like(hardest_positive_dist))

    # Get final mean triplet loss
    triplet_loss = torch.mean(triplet_loss)

    return triplet_loss


def _get_triplet_mask(labels):
    labels = labels.cpu()
    # Check that i, j and k are distinct
    indices_equal = torch.eye(labels.shape[0]).type(torch.BoolTensor)
    indices_not_equal = torch.logical_not(indices_equal)
    i_not_equal_j = torch.unsqueeze(indices_not_equal, 2)
    i_not_equal_k = torch.unsqueeze(indices_not_equal, 1)
    j_not_equal_k = torch.unsqueeze(indices_not_equal, 0)

    distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

    # Check if labels[i] == labels[j] and labels[i] != labels[k]
    label_equal = torch.eq(torch.unsqueeze(labels, 0), torch.unsqueeze(labels, 1))
    i_equal_j = torch.unsqueeze(label_equal, 2)
    i_equal_k = torch.unsqueeze(label_equal, 1)
    valid_labels = i_equal_j & torch.logical_not(i_equal_k)
    # Combine the two masks
    mask = distinct_indices & valid_labels

    return mask


def accuracy(labels, embeddings, squared=False, T=np.array([0.55, 0.6, 0.7, 0.8, 0.9])):
    labels = labels.cpu()
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    mask = np.array(_get_triplet_mask(labels))
    x, y, z = np.where(mask == True)

    n_triplets = len(x)
    if not n_triplets:
        return np.zeros_like(T)

    true = np.zeros_like(T)
    for i in range(len(x)):
        x_i, y_i, z_i = x[i], y[i], z[i]
        anchor_pos_dist = pairwise_dist[x_i, y_i]
        anchor_neg_dist = pairwise_dist[x_i, z_i]
        if anchor_pos_dist < anchor_neg_dist:
            true += anchor_pos_dist.item() < T

    return true / n_triplets

def batch_hard_quadruplet_loss(class_labels, scene_labels, embeddings,
                               margin_class=0.3, margin_scene=0.05,
                               beta=0.4):
    '''
    Computes quadruplet loss:
    triplet_loss_class + beta*triplet_loss_scene
    :param class_labels:
    :param scene_labels:
    :param embeddings:
    :param margin_class:
    :param margin_scene:
    :param beta:
    :return:
    '''

    loss = batch_hard_triplet_loss(class_labels, embeddings, margin=margin_class)

    if len(np.unique(scene_labels)) != len(scene_labels):
        loss += beta * batch_hard_triplet_loss(scene_labels, embeddings, margin=margin_scene)

    return loss

