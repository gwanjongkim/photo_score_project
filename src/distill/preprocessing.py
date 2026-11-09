from __future__ import annotations

import tensorflow as tf


DEFAULT_INTERPOLATION = "bilinear"


def decode_student_rgb_image(path: tf.Tensor) -> tf.Tensor:
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image.set_shape([None, None, 3])
    return image


def resize_student_image(image: tf.Tensor, image_size: int) -> tf.Tensor:
    image = tf.image.resize(image, [int(image_size), int(image_size)], method=DEFAULT_INTERPOLATION)
    return tf.ensure_shape(image, [int(image_size), int(image_size), 3])


def normalize_student_image(image: tf.Tensor) -> tf.Tensor:
    return tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 1.0)


def decode_resize_normalize_student_image(path: tf.Tensor, image_size: int) -> tf.Tensor:
    image = decode_student_rgb_image(path)
    image = resize_student_image(image, image_size=image_size)
    return normalize_student_image(image)


def student_preprocessing_metadata(image_size: int) -> dict[str, object]:
    image_size = int(image_size)
    return {
        "image_size": [image_size, image_size],
        "decode": {
            "color_format": "RGB",
            "animation_handling": "expand_animations=False",
        },
        "resize": {
            "method": DEFAULT_INTERPOLATION,
            "mode": "direct_resize",
        },
        "normalization": {
            "type": "pixel_value_div_255",
            "range": [0.0, 1.0],
            "keras_preprocess_input": None,
        },
        "layout": "NHWC",
        "dtype": "float32",
    }


def describe_student_preprocessing(image_size: int) -> dict[str, object]:
    return student_preprocessing_metadata(image_size=image_size)
