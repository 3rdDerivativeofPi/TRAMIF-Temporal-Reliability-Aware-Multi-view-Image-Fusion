import numpy as np

from src.preprocessing.common import mask_aware_area_resize


def test_padding_does_not_reduce_average():
    image = np.array(
        [[1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )

    mask = np.array(
        [[1, 0, 0, 0]],
        dtype=bool,
    )

    output = mask_aware_area_resize(
        image,
        mask,
        output_size=(1, 1),
    )

    print(output)

    np.testing.assert_allclose(
        output[0, 0],
        1.0,
        atol=1e-7,
    )

# testing
# invalid / padded pixels should not contribute to the average of valid pixels

def test_all_valid_pixels_are_averaged():
    image = np.array(
        [[1.0, 0.0]],
        dtype=np.float32,
    )

    mask = np.array(
        [[1, 1]],
        dtype=bool,
    )

    output = mask_aware_area_resize(
        image,
        mask,
        output_size=(1, 1),
    )

    print(output)

    np.testing.assert_allclose(
        output[0, 0],
        0.5,
        atol=1e-7,
    )

# testing resizing
# all valid pixels should be averaged when resizing to a smaller size
