# sod_model.py
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = 128

# --- helper conv blocks ---
def conv_block_basic(x, filters, kernel=3, activation='relu'):
    x = layers.Conv2D(filters, kernel, padding='same', activation=activation)(x)
    x = layers.Conv2D(filters, kernel, padding='same', activation=activation)(x)
    return x

def conv_block_bn_dropout(x, filters, kernel=3, dropout_rate=0.2, activation='relu'):
    x = layers.Conv2D(filters, kernel, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    x = layers.Conv2D(filters, kernel, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(activation)(x)

    if dropout_rate and dropout_rate>0:
        x = layers.Dropout(dropout_rate)(x)
    return x

def up_block_basic(x, skip, filters):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same', activation='relu')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block_basic(x, filters)
    return x

def up_block_bn(x, skip, filters, dropout_rate=0.2):
    x = layers.Conv2DTranspose(filters, 2, strides=2, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block_bn_dropout(x, filters, dropout_rate=dropout_rate)
    return x

# --- Baseline (your original) ---
def build_unet_baseline(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = layers.Input(shape=input_shape)
    # encoder
    x1 = conv_block_basic(inputs, 32)
    p1 = layers.MaxPooling2D(2)(x1)
    x2 = conv_block_basic(p1, 64)
    p2 = layers.MaxPooling2D(2)(x2)
    x3 = conv_block_basic(p2, 128)
    p3 = layers.MaxPooling2D(2)(x3)
    x4 = conv_block_basic(p3, 256)
    p4 = layers.MaxPooling2D(2)(x4)
    # bottleneck
    b = layers.Conv2D(512, 3, padding='same', activation='relu')(p4)
    b = layers.Conv2D(512, 3, padding='same', activation='relu')(b)
    # decoder
    u1 = up_block_basic(b, x4, 256)
    u2 = up_block_basic(u1, x3, 128)
    u3 = up_block_basic(u2, x2, 64)
    u4 = up_block_basic(u3, x1, 32)
    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u4)
    return tf.keras.Model(inputs, outputs)

# --- Variant A: BatchNorm + Dropout ---
def build_unet_bn_dropout(input_shape=(IMG_SIZE, IMG_SIZE, 3), base_filters=32, dropout_rate=0.2):
    inputs = layers.Input(shape=input_shape)
    # encoder
    x1 = conv_block_bn_dropout(inputs, base_filters, dropout_rate=0.0)  # small dropout in shallow layers
    p1 = layers.MaxPooling2D(2)(x1)

    x2 = conv_block_bn_dropout(p1, base_filters*2, dropout_rate=0.0)
    p2 = layers.MaxPooling2D(2)(x2)

    x3 = conv_block_bn_dropout(p2, base_filters*4, dropout_rate=0.1)
    p3 = layers.MaxPooling2D(2)(x3)

    x4 = conv_block_bn_dropout(p3, base_filters*8, dropout_rate=0.1)
    p4 = layers.MaxPooling2D(2)(x4)

    # bottleneck
    b = conv_block_bn_dropout(p4, base_filters*16, dropout_rate=0.3)

    # decoder with dropout
    u1 = up_block_bn(b, x4, base_filters*8, dropout_rate=0.2)
    u2 = up_block_bn(u1, x3, base_filters*4, dropout_rate=0.2)
    u3 = up_block_bn(u2, x2, base_filters*2, dropout_rate=0.1)
    u4 = up_block_bn(u3, x1, base_filters, dropout_rate=0.0)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u4)
    return tf.keras.Model(inputs, outputs)

# --- Variant B: Deeper / wider network ---
def build_unet_deep(input_shape=(IMG_SIZE, IMG_SIZE, 3), base_filters=64):
    inputs = layers.Input(shape=input_shape)
    # encoder (one extra stage and more filters)
    x1 = conv_block_basic(inputs, base_filters)
    p1 = layers.MaxPooling2D(2)(x1)

    x2 = conv_block_basic(p1, base_filters*2)
    p2 = layers.MaxPooling2D(2)(x2)

    x3 = conv_block_basic(p2, base_filters*4)
    p3 = layers.MaxPooling2D(2)(x3)

    x4 = conv_block_basic(p3, base_filters*8)
    p4 = layers.MaxPooling2D(2)(x4)

    x5 = conv_block_basic(p4, base_filters*16)  # extra stage
    p5 = layers.MaxPooling2D(2)(x5)

    # bottleneck
    b = layers.Conv2D(base_filters*32, 3, padding='same', activation='relu')(p5)
    b = layers.Conv2D(base_filters*32, 3, padding='same', activation='relu')(b)

    # decoder
    u1 = layers.Conv2DTranspose(base_filters*16, 2, strides=2, padding='same', activation='relu')(b)
    u1 = layers.Concatenate()([u1, x5]); u1 = conv_block_basic(u1, base_filters*16)

    u2 = layers.Conv2DTranspose(base_filters*8, 2, strides=2, padding='same', activation='relu')(u1)
    u2 = layers.Concatenate()([u2, x4]); u2 = conv_block_basic(u2, base_filters*8)

    u3 = layers.Conv2DTranspose(base_filters*4, 2, strides=2, padding='same', activation='relu')(u2)
    u3 = layers.Concatenate()([u3, x3]); u3 = conv_block_basic(u3, base_filters*4)

    u4 = layers.Conv2DTranspose(base_filters*2, 2, strides=2, padding='same', activation='relu')(u3)
    u4 = layers.Concatenate()([u4, x2]); u4 = conv_block_basic(u4, base_filters*2)

    u5 = layers.Conv2DTranspose(base_filters, 2, strides=2, padding='same', activation='relu')(u4)
    u5 = layers.Concatenate()([u5, x1]); u5 = conv_block_basic(u5, base_filters)

    outputs = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(u5)
    return tf.keras.Model(inputs, outputs)


# --- metrics / loss ---
def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def sod_loss(y_true, y_pred):
    return bce(y_true, y_pred) + 0.5*(1 - iou_score(y_true, y_pred))

# convenience
def get_model_by_name(name, **kwargs):
    name = name.lower()
    if name in ['baseline', 'base']:
        return build_unet_baseline(**kwargs)
    if name in ['bn_dropout', 'variant_a']:
        return build_unet_bn_dropout(**kwargs)
    if name in ['deep', 'variant_b']:
        return build_unet_deep(**kwargs)
    raise ValueError("Unknown model name: "+name)
