"""Custom tensorflow training callbacks."""
import sys
import time

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