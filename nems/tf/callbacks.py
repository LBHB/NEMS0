"""Custom tensorflow training callbacks."""
import sys
import time
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
    def __init__(self, count_mode='samples', verbose=2, n_iters=10, **kwargs):
        if count_mode != 'samples':
            raise ValueError('Count mode must be "samples" to use this logger.')

        self.verbose = verbose
        self.n_iters = n_iters
        super(SparseProgbarLogger, self).__init__(count_mode, **kwargs)

    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.leading_zeros = len(str((int(self.epochs))))

    def should_print(self, epoch):
        """Simple check to see if should print loss."""
        return epoch == 0 or (epoch + 1) % self.n_iters == 0

    def on_epoch_begin(self, epoch, logs=None):
        if not self.should_print(epoch):
            return

        self.seen = 0
        if self.use_steps:
            self.target = self.params['steps']
        else:
            self.target = self.params['samples']

        self.progbar = CleanProgbar(
            target=self.target,
            verbose=self.verbose,
            stateful_metrics=self.stateful_metrics,
            unit_name='step' if self.use_steps else 'sample')

    def on_epoch_end(self, epoch, logs=None):
        if not self.should_print(epoch):
            return

        logs = logs or {}
        for k in self.params['metrics']:
            if k in logs:
                self.log_values.append((k, logs[k]))
        if self.verbose:
            pre = 'Epoch {epoch:0>{zeros}}/{total}'.format(epoch=epoch + 1, zeros=self.leading_zeros, total=self.epochs)
            self.progbar.update(self.seen, self.log_values, pre=pre)


class CleanProgbar(tf.keras.utils.Progbar):
    """Sublcasses Progbar for some simpler printing. Assumes verbose == 2"""

    def update(self, current, values=None, pre=None):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                # In the case that progress bar doesn't have a target value in the first
                # epoch, both on_batch_end and on_epoch_end will be called, which will
                # cause 'current' and 'self._seen_so_far' to have the same value. Force
                # the minimal value to 1 here, otherwise stateful_metric will be 0s.
                value_base = max(current - self._seen_so_far, 1)
                if k not in self._values:
                    self._values[k] = [v * value_base, value_base]
                else:
                    self._values[k][0] += v * value_base
                    self._values[k][1] += value_base
            else:
                # Stateful metrics output a numeric value. This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = '' if pre is None else pre

        if self.target is not None and current >= self.target:
            numdigits = int(np.log10(self.target)) + 1
            for k in self._values_order:
                info += ' - %s:' % k
                avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                if avg > 1e-3:
                    info += ' %.4f' % avg
                else:
                    info += ' %.4e' % avg
            info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        self._last_update = now


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
                self.model.early_terminated = True
                self.model.stop_training = True
