"""Utility functions for the service."""
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

BASE_SCALING_FACTOR: int = 255


def prepare_bbox(bbox: List[int]) -> Dict[str, int]:
    """Convert bbox format COCO -> MinMax.

    Args:
        bbox (List[int]): list of bbox coords in COCO.

    Returns:
        Dict[str, int]: Dict of bbox coords in MinMax.
    """
    x_min = bbox[0]
    x_max = x_min + bbox[2]
    y_min = bbox[1]
    y_max = y_min + bbox[2]
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def preprocess_image(
    image: NDArray[np.uint8],
    target_image_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Preprocess an image for ImageNet.

    This function takes an RGB image, normalizes it, resizes it to the target
    image size, and transposes it to meet the input requirements for ImageNet
    models.

    Args:
        image (np.ndarray): The input RGB image.
        target_image_size (Tuple[int, int]): The target image size (height, width)

    Returns:
        torch.Tensor: A batch containing a single preprocessed image.

    """
    processed_image = image.astype(np.float32)
    processed_image /= BASE_SCALING_FACTOR

    # Calculate scaling and padding
    height, width = processed_image.shape[:2]
    target_height, target_width = target_image_size
    scale = min(target_width / width, target_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    pad_width = (target_width - new_width) // 2
    pad_height = (target_height - new_height) // 2

    # Resize the image
    processed_image = cv2.resize(processed_image, (new_width, new_height))

    # Add padding
    processed_image = cv2.copyMakeBorder(
        processed_image,
        pad_height,
        pad_height,
        pad_width,
        pad_width,
        cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )

    # Transpose and normalization
    processed_image = np.transpose(processed_image, (2, 0, 1))
    processed_image -= np.array([0.485, 0.456, 0.406])[:, None, None]
    processed_image /= np.array([0.229, 0.224, 0.225])[:, None, None]

    return torch.from_numpy(processed_image)[None]


def resize_mask_back_to_original(mask: torch.Tensor, original_image_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize the predicted mask back to the original size of the input image.

    Args:
        mask (torch.Tensor): The predicted segmentation mask.
        original_image_size (Tuple[int, int]): The size of the original image (height, width).

    Returns:
        np.ndarray: The resized mask.
    """
    # Convert mask from PyTorch tensor to numpy array if it's not already
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    original_height, original_width = original_image_size
    mask_height, mask_width = mask.shape[-2:]

    # Calculate the scale and padding used during preprocessing
    scale = min(mask_width / original_width, mask_height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    pad_width = (mask_width - new_width) // 2
    pad_height = (mask_height - new_height) // 2

    # Crop the padding
    cropped_mask = mask[pad_height : pad_height + new_height, pad_width : pad_width + new_width]

    return cv2.resize(cropped_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
