import tensorflow as tf

def build_technical_regressor(input_shape=(224, 224, 3), rescale_unit_to_signed: bool = True):
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=input_shape
    )

    inputs = tf.keras.Input(shape=input_shape)
    x = (
        tf.keras.layers.Rescaling(2.0, offset=-1.0, name="mobilenetv2_unit_to_signed")(inputs)
        if rescale_unit_to_signed
        else inputs
    )
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs, outputs, name="technical_regressor")
    return model
