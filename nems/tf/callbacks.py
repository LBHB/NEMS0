"""Custom tensorflow training callbacks."""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf


class DelayedStopper(tf.keras.callbacks.EarlyStopping):
    """Early stopper that waits before kicking in."""
    def __init__(self, start_epoch=100, **kwargs):
        super(DelayedStopper, self).__init__(**kwargs)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


class SparseProgbarLogger(tf.keras.callbacks.ProgbarLogger):
    """Subclasses ProgbarLogger to not print on each iteration."""
    def __init__(self, count_mode='samples', n_iters=10, **kwargs):
        if count_mode != 'samples':
            raise ValueError('Count mode must be "samples" to use this logger.')

        self.n_iters = n_iters
        super(SparseProgbarLogger, self).__init__(count_mode, **kwargs)

    def set_params(self, params):
        super(SparseProgbarLogger, self).set_params(params)
        self.target = self.epochs
        self.verbose = 2
        self.leading_zeros = int(np.log10(self.target)) + 1

    def should_print(self, epoch):
        """Simple check to see if should print loss."""
        return epoch == 0 or (epoch + 1) % self.n_iters == 0

    def on_epoch_begin(self, epoch, logs=None):
        """Override to avoid printing epoch number"""
        pass

    def on_epoch_end(self, epoch, logs=None):
        """Skips the whole progbar setup and just prints logs to sdout."""
        if not self.should_print(epoch):
            return

        info = 'Epoch {epoch:0>{zeros}}/{total}'.format(epoch=epoch + 1, zeros=self.leading_zeros, total=self.target)

        for k, v in logs.items():
            info += ' - %s:' % k
            if v > 1e-3:
                info += ' %.4f' % v
            else:
                info += ' %.4e' % v
        info += '\n'

        sys.stdout.write(info)
        sys.stdout.flush()


class GradientLogger(tf.keras.callbacks.Callback):
    """Logs gradients."""
    def __init__(self, train_input, filepath, model, **kwargs):
        super(GradientLogger, self).__init__(**kwargs)

        self.train_input = tf.convert_to_tensor(train_input, dtype=tf.float32)

        # dict of dicts: in order to have lines on the same fig, need a writer for each trace
        # top level is weight names, nested is flattened idx with value of file writer
        self.writers = {}

        # pull out the layer names to create all the necessary file writers
        for layer_num, layer in enumerate(model.layers):
            for weight in layer.weights:
                # add in layer num so we can organize the ordering in tensorboard
                self.writers[weight.name] = {'layer_num': layer_num}

                shape = weight.shape
                for idx in range(weight.numpy().size):
                    str_ind = str(np.unravel_index(idx, shape)).replace(', ', '-').strip(')').strip('(')
                    writer_path = Path(filepath) / weight.name.replace(':', '-') / str_ind
                    writer = tf.summary.create_file_writer(str(writer_path))#, name=writer_name)
                    self.writers[weight.name][idx] = writer

    def get_gradients(self):
        """Creates gradient tape."""
        with tf.GradientTape() as tape:
            loss = self.model(self.train_input)

        return tape.gradient(loss, self.model.weights)

    def write_gradient_weight(self, weight, gradient, epoch):
        """Writes the values of a single gradient."""
        name = weight.name

        if name not in self.writers:
            raise KeyError(f'Weight name "{name}" not found in list of writers.')

        tb_name = str(self.writers[name]['layer_num']) + '-' + name.split(':')[0]

        for idx, (gradient_value, weight_value) in enumerate(zip(
                gradient.numpy().flatten(),
                weight.numpy().flatten())):
            writer = self.writers[name][idx]

            with writer.as_default():
                tf.summary.scalar(tb_name + '/gradient', gradient_value, step=epoch)
                tf.summary.scalar(tb_name + '/weight', weight_value, step=epoch)

    def on_epoch_end(self, epoch, logs=None):
        """Writes a histogram for each gradient."""
        gradients = self.get_gradients()

        for gradient, weight in zip(gradients, self.model.weights):
            self.write_gradient_weight(weight, gradient, epoch)


class TerminateOnNaNWeights(tf.keras.callbacks.Callback):
    """Termiantes on NaN weights, or inf. Modelled on tf.keras.callbacks.TerminateOnNan."""
    def on_epoch_end(self, epoch, logs=None):
        """Goes through weights looking for any NaNs."""
        for weight in self.model.weights:
            if tf.math.reduce_any(tf.math.is_nan(weight)) or tf.math.reduce_any(tf.math.is_inf(weight)):
                print('Epoch %d: Invalid weights in "%s", terminating training' % (epoch, weight.name))
                print('Weights %s' % (weight))
                self.model.early_terminated = True
                self.model.stop_training = True
