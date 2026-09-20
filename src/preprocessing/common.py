import numpy as np
import torch
import torch.nn.functional as F


def mask_aware_area_resize(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: tuple[int, int] = (64, 64),
) -> np.ndarray:
    """
    Resize a 2D image using area interpolation without allowing
    padding positions to influence valid-pixel averages.

    Parameters
    ----------
    image:
        2D numeric array.

    mask:
        2D array with the same shape as image.
        Valid positions are True/1 and padding positions are False/0.

    output_size:
        Desired (height, width).

    Returns
    -------
    np.ndarray
        Float32 image with shape output_size.
    """

    image = np.asarray(image, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D image, got shape {image.shape}"
        )

    if mask.shape != image.shape:
        raise ValueError(
            "Mask and image must have identical shapes"
        )

    if image.size == 0:
        raise ValueError("Cannot resize an empty image")

    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("Output dimensions must be positive")

    image_t = torch.from_numpy(image)[None, None, :, :]
    mask_t = torch.from_numpy(mask)[None, None, :, :]

    weighted_sum = F.interpolate(
        image_t * mask_t,
        size=output_size,
        mode="area",
    )

    valid_weight = F.interpolate(
        mask_t,
        size=output_size,
        mode="area",
    )

    output = torch.zeros_like(weighted_sum)

    valid = valid_weight > 0

    output[valid] = (
        weighted_sum[valid]
        / valid_weight[valid]
    )

    return (
        output[0, 0]
        .numpy()
        .astype(np.float32, copy=False)
    )