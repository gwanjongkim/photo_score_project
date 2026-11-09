import tensorflow as tf

def build_aadb_regressor(input_shape=(224, 224, 3), rescale_unit_to_signed: bool = True):
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape,
        include_preprocessing=False,
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = (
        tf.keras.layers.Rescaling(2.0, offset=-1.0, name="unit_to_signed")(inputs)
        if rescale_unit_to_signed
        else inputs
    )
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="aadb_regressor")
    return model
