"""Custom tensorflow training callbacks."""

import tensorflow as tf


class DelayedStopper(tf.keras.callbacks.EarlyStopping):
    """Early stopper that waits before kicking in."""
    def __init__(self, start_epoch=100, **kwargs):
        super(DelayedStopper, self).__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)
