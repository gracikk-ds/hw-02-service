"""Unit tests."""

from copy import deepcopy

import numpy as np
from numpy.typing import NDArray

from src.containers.containers import AppContainer


def test_rec_predicts_not_fail(app_container: AppContainer, sample_image_np: NDArray[np.uint8]):
    """
    Test to ensure the rec model prediction does not fail.

    Args:
        app_container (AppContainer): The application container holding the rec model.
        sample_image_np (NDArray[np.uint8]): The numpy array of the sample image to be tested.

    """
    segmentor = app_container.rec_model()
    segmentor.predict(sample_image_np)


def test_rec_predict_dont_mutate_initial_image(app_container: AppContainer, sample_image_np: NDArray[np.uint8]):
    """
    Test to ensure the initial image is not mutated during rec prediction.

    Args:
        app_container (AppContainer): The application container holding the rec model.
        sample_image_np (NDArray[np.uint8]): The numpy array of the sample image to be tested.

    """
    image_to_compare = deepcopy(sample_image_np)
    model = app_container.rec_model()
    model.predict(sample_image_np)

    assert np.allclose(sample_image_np, image_to_compare)  # noqa: S101
