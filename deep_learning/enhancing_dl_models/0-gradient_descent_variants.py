from tensorflow import keras


def train_with_gradient_descent_variant(variant, learning_rate, x_train, batch_size):
    """
    Arguments:
    variant: A string — 'batch', 'stochastic', or 'mini_batch'.
    learning_rate: Learning rate for the optimizer.
    x_train: The training dataset (input data).
    batch_size: Integer, batch size to use for mini-batch gradient descent.

    Returns:
    optimizer: A configured SGD optimizer.
    bs: The correct batch size to be passed to model.fit.
    """

    if variant == 'batch':
        bs = len(x_train)
    elif variant == 'stochastic':
        bs = 1
    elif variant == 'mini_batch':
        bs = batch_size

    optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    return optimizer, bs
