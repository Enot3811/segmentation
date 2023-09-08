from typing import List, Optional

import torch
from torch import FloatTensor, IntTensor


def segmentation_iou(
    pred: FloatTensor,
    target: IntTensor,
    ignore: Optional[List] = None,
    eps: float = 1e-8
) -> FloatTensor:
    """Calculate IoU metric over classes for semantic segmentation.

    Parameters
    ----------
    pred : FloatTensor
        Predicted segmentation logits with shape `(b, c, h, w)`
        where `c` is number of classes.
    target : IntTensor
        Ground truth mask of segmentation with shape `(b, h, w)`.
    ignore : Optional[List]
        Labels that will be ignored. By default is `None`.
    eps : float, optional
        some small value to avoid zero division. By default 1e-8.

    Returns
    -------
    FloatTensor
        IoU per classes with shape `(c,)`.

    Raises
    ------
    ValueError
        "pred" must be 4D float tensor.
    """
    if (len(pred.shape) != 4 or not isinstance(pred, torch.Tensor) or not
            pred.is_floating_point()):
        raise ValueError('"pred" must be 4D float tensor.')
    n_cls = pred.shape[1]
    cls_ious = torch.zeros((n_cls,), dtype=torch.float32)
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    for cls_idx in range(n_cls):
        if ignore is not None and cls_idx in ignore:
            continue
        predict_mask = pred == cls_idx
        gt_mask = target == cls_idx
        intersection = (predict_mask[gt_mask]).sum()
        union = predict_mask.sum() + gt_mask.sum() - intersection
        cls_ious[cls_idx] = intersection / (union + eps)
    return cls_ious
