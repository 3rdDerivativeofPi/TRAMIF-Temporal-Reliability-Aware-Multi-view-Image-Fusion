import numpy as np

from .common import mask_aware_area_resize


RAW_WIDTH = 256
OUTPUT_SIZE = (64, 64)


def raw_byte_layout(
    data: bytes,
    width: int = RAW_WIDTH,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map file bytes into a row-major byte image.

    Valid byte values are normalized from [0, 255] to [0, 1].

    Padding is represented by zero in the image but marked False
    in the validity mask, so padding does not contribute during
    resizing.
    """

    if width <= 0:
        raise ValueError("width must be positive")

    values = np.frombuffer(
        data,
        dtype=np.uint8,
    )

    n_bytes = len(values)

    if n_bytes == 0:
        raise ValueError(
            "Cannot construct a raw-byte image from an empty file"
        )

    n_rows = (
        n_bytes + width - 1
    ) // width

    image = np.zeros(
        (n_rows, width),
        dtype=np.float32,
    )

    mask = np.zeros(
        (n_rows, width),
        dtype=bool,
    )

    image.flat[:n_bytes] = (
        values.astype(np.float32)
        / 255.0
    )

    mask.flat[:n_bytes] = True

    return image, mask


def raw_byte_image(
    data: bytes,
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> np.ndarray:
    """
    Construct the final fixed-size raw-byte image.
    """

    image, mask = raw_byte_layout(data)

    output = mask_aware_area_resize(
        image=image,
        mask=mask,
        output_size=output_size,
    )

    return output