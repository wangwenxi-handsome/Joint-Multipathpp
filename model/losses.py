import torch
from torch import nn
import numpy as np
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal


def min_ade_prob_heading_loss(gt_xy, gt_valid, gt_yaw, probas, coordinates, yaws):
    """
    Compute ade loss.
    Args:
        gt_xy.shape is [b, n, t, 2]
        coordinates.shape is [b, n, m, t, 2]
        probas.shape is [b, n, m]
        gt_valid.shape is [b, n, t, 1]
        gt_yaw is [b, n, t, 1]
        yaws is [b, n, m, t, 1]
    Returns:
        a single float number
    """
    gt_xy = torch.unsqueeze(gt_xy, 2)
    gt_yaw = torch.unsqueeze(gt_yaw, 2)
    gt_valid = torch.unsqueeze(gt_valid, 2)
    error = (gt_xy - coordinates) * (gt_xy - coordinates)
    error = error * gt_valid
    error = torch.mean(error, axis=[-1, -2])
    error, idx = torch.min(error, axis=-1)
    distance_loss = torch.mean(error)
    probas = nn.functional.softmax(probas.reshape(-1, probas.shape[-1]))
    confidence_loss = nn.functional.cross_entropy(probas, idx.reshape(-1))
    yaw_error = (gt_yaw - yaws) * (gt_yaw - yaws)
    yaw_error = torch.mean(yaw_error, axis=[-1, -2])
    yaw_loss = torch.mean(torch.gather(yaw_error, dim=-1, index=idx.unsqueeze(-1)))
    return distance_loss, 0 * yaw_loss, 10 * confidence_loss


def mean_ade_loss(gt, predictions, confidences, avails, covariance_matrices):
    """
    Compute ade loss.
    Args:
        gt.shape is [b, n, t, 2]
        predictions.shape is [b, n, m, t, 2]
        confidences.shape is [b, n, m]
        avails.shape is [b, n, t, 1]
        covariance_matrices,shape is [b, n, m, t, 2, 2]
    Returns:
        a single float number
    """
    gt = torch.unsqueeze(gt, 2)
    avails = avails[:, :, None, :, :]
    error = (gt - predictions) * (gt - predictions)
    error = error * avails
    return torch.mean(error), None


def min_ade_loss(gt, predictions, confidences, avails, covariance_matrices):
    """
    Compute ade loss.
    Args:
        gt.shape is [b, n, t, 2]
        predictions.shape is [b, n, m, t, 2]
        confidences.shape is [b, n, m]
        avails.shape is [b, n, t, 1]
        covariance_matrices,shape is [b, n, m, t, 2, 2]
    Returns:
        a single float number
    """
    gt = torch.unsqueeze(gt, 2)
    avails = avails[:, :, None, :, :]
    error = (gt - predictions) * (gt - predictions)
    error = error * avails
    error = torch.mean(error, axis=[-1, -2])
    error, idx = torch.min(error, axis=-1)
    distance_loss = torch.mean(error)
    confidences = nn.functional.softmax(confidences.reshape(-1, confidences.shape[-1]))
    confidence_loss = nn.functional.cross_entropy(confidences, idx.reshape(-1))
    return distance_loss, 0.1 * confidence_loss


def nll_with_covariances(gt, predictions, confidences, avails, covariance_matrices):
    """
    Compute nll loss.
    Args:
        gt.shape is [b, n, t, 2]
        predictions.shape is [b, n, m, t, 2]
        confidences.shape is [b, n, m]
        avails.shape is [b, n, t, 1]
        covariance_matrices,shape is [b, n, m, t, 2, 2]
    Returns:
        a single float number
    """
    precision_matrices = torch.inverse(covariance_matrices)
    gt = torch.unsqueeze(gt, 2)
    avails = avails[:, :, None, :, :]
    coordinates_delta = (gt - predictions).unsqueeze(-1)
    errors = coordinates_delta.permute(0, 1, 2, 3, 5, 4) @ precision_matrices @ coordinates_delta
    assert torch.isfinite(errors).all()
    # print(torch.max(covariance_matrices), torch.min(covariance_matrices))
    errors = avails * (-0.5 * errors.squeeze(-1) - 0.5 * torch.logdet(covariance_matrices).unsqueeze(-1))
    assert torch.isfinite(errors).all()
    with np.errstate(divide="ignore"):
        errors = nn.functional.log_softmax(confidences, dim=2) + \
            torch.sum(errors, dim=[3, 4])
    errors = -torch.logsumexp(errors, dim=-1, keepdim=True)
    return torch.mean(errors)


def neg_multi_log_likelihood_batch(gt, predictions, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        predictions (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords
    error = torch.sum(
        ((gt - predictions) * avails) ** 2, dim=-1
    )  # reduce coords and use availability
    with np.errstate(
        divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time
    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)
    return torch.mean(error)


def get_model_loss(name):
    if name == "mean_ade":
        return mean_ade_loss
    if name == "min_ade":
        return min_ade_loss
    if name == "nll_with_covariances":
        return nll_with_covariances
    if name == "neg_multi_log_likelihood_batch":
        return neg_multi_log_likelihood_batch
    if name == "min_ade_prob_heading":
        return min_ade_prob_heading_loss
    raise ValueError(f"{name} is not supported")