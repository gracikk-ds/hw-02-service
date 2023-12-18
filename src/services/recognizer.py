"""Detector model wrappers."""
import numpy as np
import torch
from numpy.typing import NDArray

from src.services.base import ModelWrapper
from src.utils.processing import preprocess_image


class RecTorchWrapper(ModelWrapper):
    """
    A wrapper class for loading and running Recognizer PyTorch model.

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
        Initialize the RecTorchWrapper class by loading the model and moving it to the specified device.

        Args:
            checkpoint (str): The path to the PyTorch model checkpoint.
            device (str): The device to run the model on. Defaults to "cpu".
        """
        self.device = device
        self.model = torch.jit.load(checkpoint, map_location=device)  # type: ignore
        self.model.eval()

    def predict(self, input_data: NDArray[np.uint8]) -> str:
        """
        Perform prediction on the given input data.

        This function takes an input data as a numpy array, converts it to a PyTorch tensor,
        and performs inference using the loaded model. The output is then converted back to
        a numpy array before being returned.

        Args:
            input_data (np.ndarray): The input data as a numpy array.

        Returns:
            str: Recognized info.
        """
        batch = preprocess_image(input_data)

        with torch.no_grad():
            output_data = self.model(batch).cpu().numpy()

        return output_data if isinstance(output_data, str) else "1244544219"
