"""An abstract base class for model wrappers."""
from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class ModelWrapper(ABC):
    """
    An abstract base class for model wrappers.

    This class defines a common interface for loading models and performing predictions.
    All model wrapper classes should inherit from this class and implement its abstract methods.
    """

    threshold: float = 0.5

    @abstractmethod
    def __init__(self, checkpoint: str, device: str = "cpu"):
        """
        Initialize the model wrapper.

        Args:
            checkpoint (str): The path to the model checkpoint.
            device (str): The device to run the model on. Defaults to "cpu".
        """

    @abstractmethod
    def predict(self, input_data: NDArray[Any]) -> Any:
        """
        Perform prediction on the given input data.

        Args:
            input_data (NDArray): The input data as a numpy array.

        Returns:
            Any: The output data.
        """
