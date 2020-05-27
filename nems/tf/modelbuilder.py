import tensorflow as tf
import logging

from nems.tf import loss_functions

log = logging.getLogger(__name__)


class ModelBuilder:
    """Basic wrapper to build tensorflow keras models.

    Not an actual subclass of tf.keras.Model, but instead a set of helper
    methods to prepare layers and then return a model when ready.
    """
    def __init__(self,
                 name='model',
                 layers=None,
                 learning_rate=1e-3,
                 optimizer='adam',
                 loss_fn=loss_functions.loss_se,
                 metrics=None,
                 *args,
                 **kwargs
                 ):
        """Layers don't need to be added at init, only prior to fitting.

        :param name:
        :param args:
        :param kwargs:
        """
        self.optimizer_dict = {
            'adam': tf.keras.optimizers.Adam,
            'sgd': tf.keras.optimizers.SGD,
        }

        if layers is not None:
            self.model_layers = layers
        else:
            self.model_layers = []

        self.name = name
        self.loss_fn = loss_fn
        self.metrics = metrics

        self.learning_rate = learning_rate
        self.optimizer = optimizer

    def add_layer(self, layer, idx=None):
        """Adds a layer to the list of layers that will be part of the model.

        :param layer: Layer to add.
        :param idx: Insertion index.
        """
        if idx is None:
            self.model_layers.append(layer)
            log.debug(f'Added "{layer.ms_name}" to end of model.')
        else:
            self.model_layers.insert(idx, layer)
            log.debug(f'Inserted "{layer.ms_name}" to idx "{idx}".')

    def build_model(self, input_shape):
        """Creates a tf.keras.Model instance. Simple single headed model.

        :param input_shape: Allows summary creation. Shape should be a tuple of (batch, time, channel).
        :return: tf.keras.Model instance.
        """
        inputs = tf.keras.Input(shape=input_shape[1:], name='input_node', batch_size=input_shape[0], dtype='float32')
        outputs = inputs
        for layer in self.model_layers:
            outputs = layer(outputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        log.info('Built model, printing summary.')
        model.summary(print_fn=log.info)

        # build the optimizer
        try:
            optimizer = self.optimizer_dict[self.optimizer](learning_rate=self.learning_rate)
        except KeyError:
            optimizer = self.optimizer

        model.compile(optimizer, loss=self.loss_fn, metrics=self.metrics)
        return model
