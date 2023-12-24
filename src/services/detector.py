"""Detector model wrappers."""
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.ndimage import label

from src.services.base import ModelWrapper
from src.utils.processing import preprocess_image, resize_mask_back_to_original

SCALE: int = 255


class SegTorchWrapper(ModelWrapper):
    """
    A wrapper class for loading and running Segmentation PyTorch model.

    This class is used to load a PyTorch model from a given checkpoint path and
    perform predictions on the given input data on the specified device.

    Attributes:
        model (torch.jit.ScriptModule): The loaded PyTorch model.
        device (str): The device on which the model will be run.

    Args:
        checkpoint (str): The path to the PyTorch model checkpoint.
        device (str): The device to run the model on. Defaults to "cpu".
    """

    def __init__(self, checkpoint: str, device: str = "cpu"):
        """
        Initialize the SegTorchWrapper class by loading the model and moving it to the specified device.

        Args:
            checkpoint (str): The path to the PyTorch model checkpoint.
            device (str): The device to run the model on. Defaults to "cpu".
        """
        self.device = device
        self.model = torch.jit.load(checkpoint, map_location=device)  # type: ignore
        self.model.eval()

    @staticmethod
    def masks_to_bboxes(mask: NDArray[np.uint8]) -> List[List[int]]:
        """
        Convert a binary mask with potentially multiple objects to a list of bounding boxes in COCO format.

        Args:
            mask (NDArray[np.uint8]): A binary mask where objects' pixels are 1 and the background is 0.

        Returns:
            List[List[int]]: A list of bounding boxes, each in the format [x_min, y_min, width, height].
        """
        # Label different components (barcodes)
        labeled_array, num_features = label(mask)

        bboxes = []
        for barcode_label in range(1, num_features + 1):
            # Find the bounding box for each labeled component
            barcode_mask = labeled_array == barcode_label
            rows = np.any(barcode_mask, axis=1)
            cols = np.any(barcode_mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            bbox_width = x_max - x_min + 1
            bbox_height = y_max - y_min + 1

            bboxes.append([int(x_min), int(y_min), int(bbox_width), int(bbox_height)])
        return bboxes

    def predict_mask(self, input_data: NDArray[np.uint8]) -> NDArray[np.float32]:
        """
        Perform prediction on the given input data.

        This function takes an input data as a numpy array, converts it to a PyTorch tensor,
        and performs inference using the loaded model. The output is then converted back to
        a numpy array before being returned.

        Args:
            input_data (np.ndarray): The input data as a numpy array.

        Returns:
            np.ndarray: The output data as a numpy array.
        """
        batch = preprocess_image(input_data)
        intial_shape = input_data.shape[:2]

        with torch.no_grad():
            output_data = self.model(batch).cpu().numpy()
            output_data = output_data.squeeze()
            output_data = resize_mask_back_to_original(output_data, intial_shape)  # type: ignore

        return output_data

    def predict(self, input_data: NDArray[np.uint8]) -> List[List[int]]:
        """
        Perform prediction on the given input data and postprocess it.

        This function takes an input data as a numpy array, converts it to a PyTorch tensor,
        and performs inference using the loaded model. The output is then postprocessed to get the bounding boxes

        Args:
            input_data (np.ndarray): The input data as a numpy array.

        Returns:
            List[List[int]]: Predicted bounding boxes in COCO format.
        """
        batch = preprocess_image(input_data)
        intial_shape = input_data.shape[:2]

        with torch.no_grad():
            output_data = self.model(batch).sigmoid().cpu().numpy()
            output_data = (output_data > self.threshold).astype(np.uint8) * SCALE
            output_data = output_data.squeeze()
            output_data = resize_mask_back_to_original(output_data, intial_shape)  # type: ignore
        return self.masks_to_bboxes(output_data)
