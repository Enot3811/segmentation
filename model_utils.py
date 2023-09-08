from typing import Dict, Tuple

import numpy as np
import cv2
from numpy.typing import NDArray
import torch
from torch import FloatTensor
from torchvision.models.segmentation import deeplabv3_resnet50


def get_model(n_classes: int, pretrained: bool = False) -> torch.nn.Module:
    """Get deeplabv3 resnet50 and replace its output layers.

    Parameters
    ----------
    n_classes : int
        Number of classes for output layers.
    pretrained : bool, optional
        Whether to load a pretrained model. `False` by default.

    Returns
    -------
    torch.nn.Module
        Prepared model.
    """
    if pretrained:
        model = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
        # Aux loss is only in pretrained model
        model.aux_classifier[4] = torch.nn.Conv2d(
            256, n_classes, kernel_size=1)
    else:
        model = deeplabv3_resnet50()
    model.classifier[4] = torch.nn.Conv2d(256, n_classes, kernel_size=1)
    return model


def mask_to_rgb(
    labels: NDArray, label_map: Dict[str, Tuple[int, int, int]]
) -> NDArray:
    """Convert labels mask to rgb mask.

    Parameters
    ----------
    labels : NDArray
        Label mask with shape `(b, h, w)` or `(h, w)`.
    label_map : Dict[str, Tuple[int, int, int]]
        Label to color dictionary. Each item is int key with rgb tuple.

    Returns
    -------
    NDArray
        Created rgb mask with shape `(b, h, w, 3)` or `(h, w, 3)`.
    """
    if len(labels.shape) == 2:
        axis = 2
    elif len(labels.shape) == 3:
        axis = 3
    else:
        raise ValueError('"labels" must be 2D or 3D NDArray.')
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
  
    for label_num in range(0, len(label_map)):
        mask = labels == label_num
        r, g, b = label_map[label_num]
        red_map[mask] = r
        green_map[mask] = g
        blue_map[mask] = b
    segmentation_map = np.stack([red_map, green_map, blue_map], axis=axis)
    return segmentation_map


def create_segmentation_map(
    predicted_logits: FloatTensor, label_map: Dict[str, Tuple[int, int, int]]
) -> NDArray:
    """Get predicted segmentation logits and create segmentation maps.

    Parameters
    ----------
    predicted_logits : FloatTensor
        Batch of predicted logits from model with shape `(b, c, h, w)`
        when `c` is a number of classes.
    label_map : Dict[str, Tuple[int, int, int]]
        Label to color dictionary. Each item is int key with rgb tuple.

    Returns
    -------
    NDArray
        Batch of created segmentation maps.
    """
    labels = (torch.argmax(predicted_logits, dim=1)
              .detach().cpu().numpy())
    segmentation_map = mask_to_rgb(labels, label_map)
    return segmentation_map


def overlay_segmentation(image: NDArray, segmentation_map: NDArray) -> NDArray:
    """Overlay got segmentation maps on corresponding images.

    Parameters
    ----------
    image : NDArray
        The images to overlay with segmentation maps with shape `(h, w, 3)`.
    segmented_image : NDArray
        Segmentation maps to overlay the images with shape `(h, w)`.

    Returns
    -------
    NDArray
        The Overlaid images with shape `(h, w, 3)`.
    """
    alpha = 1  # transparency for the original image
    beta = 0.3  # transparency for the segmentation map
    gamma = 0  # scalar added to each sum
  
    segmentation_map = cv2.cvtColor(segmentation_map, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
     
    return cv2.cvtColor(
        cv2.addWeighted(image, alpha, segmentation_map, beta, gamma),
        cv2.COLOR_BGR2RGB)
