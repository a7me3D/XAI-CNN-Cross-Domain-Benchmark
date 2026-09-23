import numpy as np
import tensorflow as tf

VALID_RULES = ["basic", "epsilon", "gamma", "alphabeta"]


def fold_conv_bn(conv_layer, bn_layer):
    W = tf.cast(conv_layer.kernel, tf.float32).numpy()
    b = tf.cast(conv_layer.bias, tf.float32).numpy() if conv_layer.use_bias else np.zeros(W.shape[-1])
    gamma = tf.cast(bn_layer.gamma, tf.float32).numpy()
    beta = tf.cast(bn_layer.beta, tf.float32).numpy()
    mean = tf.cast(bn_layer.moving_mean, tf.float32).numpy()
    var = tf.cast(bn_layer.moving_variance, tf.float32).numpy()
    scale = gamma / np.sqrt(var + bn_layer.epsilon)
    W_folded = W * scale[None, None, None, :]
    b_folded = (b - mean) * scale + beta
    return tf.constant(W_folded, dtype=tf.float32), tf.constant(b_folded, dtype=tf.float32)


def build_folded_model_params(model, input_scale=None):
    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    folded_params = {}
    skip_bn = set()
    first_conv = True
    for i, layer in enumerate(layers):
        if not isinstance(layer, tf.keras.layers.Conv2D):
            continue
        if i + 1 < len(layers) and isinstance(layers[i + 1], tf.keras.layers.BatchNormalization):
            W, b = fold_conv_bn(layer, layers[i + 1])
            skip_bn.add(layers[i + 1].name)
        else:
            W = tf.constant(tf.cast(layer.kernel, tf.float32).numpy(), dtype=tf.float32)
            b = tf.constant(tf.cast(layer.bias, tf.float32).numpy() if layer.use_bias
                            else np.zeros(layer.kernel.shape[-1]), dtype=tf.float32)
        if first_conv and input_scale is not None:
            W = W * input_scale
        first_conv = False
        folded_params[layer.name] = {"W": W, "b": b}
    return folded_params, skip_bn


def backprop_conv2d(activation, relevance, layer, rule="epsilon", epsilon=1e-4, gamma=0.25,
                    alpha=1.0, beta=0.0, W_override=None, b_override=None):
    activation = tf.cast(activation, tf.float32)
    relevance = tf.cast(relevance, tf.float32)
    W = W_override if W_override is not None else tf.cast(layer.kernel, tf.float32)
    if b_override is not None:
        b = tf.reshape(b_override, [1, 1, 1, -1])
    elif layer.use_bias:
        b = tf.reshape(tf.cast(layer.bias, tf.float32), [1, 1, 1, -1])
    else:
        b = tf.zeros([1, 1, 1, W.shape[-1]], dtype=tf.float32)
    strides = layer.strides
    padding = layer.padding.upper()

    if rule == "alphabeta":
        neg_frac = float(tf.reduce_mean(tf.cast(activation < 0, tf.float32)).numpy())
        if neg_frac > 0.05:
            z = tf.nn.conv2d(activation, W, strides=strides, padding=padding) + b
            z = z + epsilon * tf.where(z >= 0, tf.ones_like(z), -tf.ones_like(z))
            R = tf.nn.conv2d_transpose(relevance / z, W, output_shape=tf.shape(activation),
                                       strides=strides, padding=padding)
            return activation * R
        Wp = tf.maximum(W, 0.)
        Wn = tf.minimum(W, 0.)
        zp = tf.nn.conv2d(activation, Wp, strides=strides, padding=padding) + tf.maximum(b, 0.)
        zn = tf.nn.conv2d(activation, Wn, strides=strides, padding=padding) + tf.minimum(b, 0.)
        zp_safe = tf.where(tf.abs(zp) < epsilon, tf.ones_like(zp) * epsilon, zp)
        zn_safe = tf.where(tf.abs(zn) < epsilon, -tf.ones_like(zn) * epsilon, zn)
        Rp = tf.nn.conv2d_transpose(relevance / zp_safe, Wp, output_shape=tf.shape(activation),
                                    strides=strides, padding=padding)
        Rn = tf.nn.conv2d_transpose(relevance / zn_safe, Wn, output_shape=tf.shape(activation),
                                    strides=strides, padding=padding)
        return activation * (alpha * Rp - beta * Rn)

    if rule == "gamma":
        W = W + gamma * tf.maximum(W, 0.)
    z = tf.nn.conv2d(activation, W, strides=strides, padding=padding) + b
    if rule in ("epsilon", "basic"):
        z = z + epsilon * tf.where(z >= 0, tf.ones_like(z), -tf.ones_like(z))
    R = tf.nn.conv2d_transpose(relevance / z, W, output_shape=tf.shape(activation),
                               strides=strides, padding=padding)
    return activation * R


def backprop_dense(activation, relevance, layer, rule="epsilon", epsilon=1e-4, gamma=0.25,
                   alpha=1.0, beta=0.0):
    activation = tf.cast(activation, tf.float32)
    relevance = tf.cast(relevance, tf.float32)
    W = tf.cast(layer.kernel, tf.float32)
    b = tf.cast(layer.bias, tf.float32) if layer.use_bias else 0.0

    if rule == "alphabeta":
        neg_frac = float(tf.reduce_mean(tf.cast(activation < 0, tf.float32)).numpy())
        if neg_frac > 0.05 or W.shape[-1] == 1:
            z = tf.matmul(activation, W) + b
            z = z + epsilon * tf.where(z >= 0, tf.ones_like(z), -tf.ones_like(z))
            return activation * tf.matmul(relevance / z, tf.transpose(W))
        Wp = tf.maximum(W, 0.)
        Wn = tf.minimum(W, 0.)
        zp = tf.matmul(activation, Wp) + (tf.maximum(b, 0.) if layer.use_bias else 0.)
        zn = tf.matmul(activation, Wn) + (tf.minimum(b, 0.) if layer.use_bias else 0.)
        zp_safe = tf.where(tf.abs(zp) < epsilon, tf.ones_like(zp) * epsilon, zp)
        zn_safe = tf.where(tf.abs(zn) < epsilon, -tf.ones_like(zn) * epsilon, zn)
        Rp = tf.matmul(relevance / zp_safe, tf.transpose(Wp))
        Rn = tf.matmul(relevance / zn_safe, tf.transpose(Wn))
        return activation * (alpha * Rp - beta * Rn)

    if rule == "gamma":
        W = W + gamma * tf.maximum(W, 0.)
    z = tf.matmul(activation, W) + b
    if rule in ("epsilon", "basic"):
        z = z + epsilon * tf.where(z >= 0, tf.ones_like(z), -tf.ones_like(z))
    return activation * tf.matmul(relevance / z, tf.transpose(W))


def backprop_batchnorm_passthrough(activation, relevance, layer, rule="basic", epsilon=1e-4):
    return relevance


def backprop_globalavgpool(activation, relevance, layer):
    activation = tf.cast(activation, tf.float32)
    relevance = tf.cast(relevance, tf.float32)
    h = tf.cast(tf.shape(activation)[1], tf.float32)
    w = tf.cast(tf.shape(activation)[2], tf.float32)
    r_expanded = tf.expand_dims(tf.expand_dims(relevance, 1), 1)
    return tf.ones_like(activation) * (r_expanded / (h * w))


def backprop_maxpool(activation, relevance, layer, rule="basic", epsilon=1e-4):
    activation = tf.cast(activation, tf.float32)
    relevance = tf.cast(relevance, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(activation)
        pooled = tf.nn.max_pool(
            activation,
            ksize=[1, layer.pool_size[0], layer.pool_size[1], 1],
            strides=[1, layer.strides[0], layer.strides[1], 1],
            padding=layer.padding.upper(),
        )
        weighted = pooled * relevance
    R = activation * tape.gradient(weighted, activation)
    t_in = tf.reduce_sum(tf.abs(relevance), axis=[1, 2, 3], keepdims=True)
    t_out = tf.reduce_sum(tf.abs(R), axis=[1, 2, 3], keepdims=True)
    return R * (t_in / (t_out + epsilon))


def compute_relevance(model, input_tensor, target_class_idx, rule="epsilon", epsilon=1e-4,
                      alpha=1.0, beta=0.0, gamma=0.25, folded_params=None, skip_bn=None,
                      conv_fn=backprop_conv2d, batchnorm_fn=backprop_batchnorm_passthrough,
                      input_scale=None):
    if rule not in VALID_RULES:
        raise ValueError(f"Invalid rule '{rule}'. Choose from {VALID_RULES}.")
    if rule == "alphabeta" and not (alpha - beta == 1.0):
        raise ValueError("alpha - beta must equal 1.0")
    folded_params = folded_params or {}
    skip_bn = skip_bn or set()

    x = tf.cast(tf.convert_to_tensor(input_tensor), tf.float32)
    while len(x.shape) > 4:
        x = tf.squeeze(x, axis=1)
    if len(x.shape) == 3:
        x = tf.expand_dims(x, 0)
    if input_scale is not None:
        x = x / input_scale

    model.trainable = False
    layers = [l for l in model.layers if not isinstance(l, tf.keras.layers.InputLayer)]
    activations = [x]
    for layer in layers:
        activations.append(tf.cast(layer(activations[-1], training=False), tf.float32))

    output = activations[-1]
    if output.shape[-1] == 1:
        relevance = tf.reshape(output if target_class_idx else 1.0 - output, [1, 1])
    else:
        if target_class_idx is None:
            target_class_idx = int(tf.argmax(output[0]).numpy())
        relevance = tf.one_hot([target_class_idx], output.shape[-1], dtype=tf.float32) * output

    relevances = [None] * len(activations)
    relevances[-1] = relevance
    for i in reversed(range(len(layers))):
        layer = layers[i]
        a_prev = activations[i]
        r_curr = relevances[i + 1]
        if isinstance(layer, tf.keras.layers.Conv2D):
            if layer.name in folded_params:
                fp = folded_params[layer.name]
                r = conv_fn(a_prev, r_curr, layer, rule, epsilon, gamma, alpha, beta,
                            W_override=fp["W"], b_override=fp["b"])
            else:
                r = conv_fn(a_prev, r_curr, layer, rule, epsilon, gamma, alpha, beta)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            r = r_curr if layer.name in skip_bn else batchnorm_fn(a_prev, r_curr, layer, rule, epsilon)
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            r = backprop_maxpool(a_prev, r_curr, layer, rule, epsilon)
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            r = backprop_globalavgpool(a_prev, r_curr, layer)
        elif isinstance(layer, tf.keras.layers.Dense):
            r = backprop_dense(a_prev, r_curr, layer, rule, epsilon, gamma, alpha, beta)
        elif isinstance(layer, (tf.keras.layers.Flatten, tf.keras.layers.Reshape)):
            r = tf.reshape(r_curr, tf.shape(a_prev))
        elif isinstance(layer, tf.keras.layers.ReLU):
            r = tf.cast(a_prev > 0, tf.float32) * r_curr
        else:
            r = r_curr
        r = tf.cast(r, tf.float32)
        r = tf.where(tf.math.is_nan(r), tf.zeros_like(r), r)
        r = tf.where(tf.math.is_inf(r), tf.zeros_like(r), r)
        relevances[i] = r
    return relevances[0].numpy()
