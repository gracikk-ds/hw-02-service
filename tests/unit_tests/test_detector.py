"""Unit tests."""

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from src.containers.containers import AppContainer


def test_seg_predicts_not_fail(app_container: AppContainer, sample_image_np: NDArray[np.uint8]):
    """
    Test to ensure the seg model prediction does not fail.

    Args:
        app_container (AppContainer): The application container holding the seg model.
        sample_image_np (NDArray[np.uint8]): The numpy array of the sample image to be tested.

    """
    segmentor = app_container.seg_model()
    segmentor.predict(sample_image_np)


def test_seg_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: NDArray[np.uint8]):
    """
    Test to ensure the initial image is not mutated during seg prediction.

    Args:
        app_container (AppContainer): The application container holding the seg model.
        sample_image_np (NDArray[np.uint8]): The numpy array of the sample image to be tested.

    """
    image_to_compare = deepcopy(sample_image_np)
    model = app_container.seg_model()
    model.predict(sample_image_np)

    assert np.allclose(sample_image_np, image_to_compare)  # noqa: S101
