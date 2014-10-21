from pylearn2.space import NullSpace
from theano import config
from pylearn2.utils import sharedX
import functools
import numpy as np
from pylearn2.train_extensions import TrainExtension

class DropoutScaler(TrainExtension):
    def __init__(self, estimate_set = "train", proportion_to_avg_act = 1.0):
        self.estimate_set = estimate_set
        self.proportion_to_avg_act = proportion_to_avg_act

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:
            '''
            if layer.layer_name not in algorithm.cost.input_include_probs:
                algorithm.cost.input_include_probs[layer.layer_name] = algorithm.cost.default_input_include_prob
            algorithm.cost.input_include_probs[layer.layer_name] = sharedX(algorithm.cost.input_include_probs[layer.layer_name])

            if layer.layer_name not in algorithm.cost.input_scales:
                algorithm.cost.input_scales[layer.layer_name] = algorithm.cost.default_input_scale
            algorithm.cost.input_scales[layer.layer_name] = sharedX(algorithm.cost.input_scales[layer.layer_name])
            '''

            channel_name = self.estimate_set + "_" + layer_name
            monitor.add_channel(
                name=channel_name + "_dropout_inc_prob",
                ipt=None,
                val=algorithm.cost.input_include_probs[layer_name],
                data_specs=(NullSpace(), ''),
                dataset=dataset)

            monitor.add_channel(
                name=channel_name + "_dropout_scale",
                ipt=None,
                val=algorithm.cost.input_scales[layer_name],
                data_specs=(NullSpace(), ''),
                dataset=dataset)

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        monitor = model.monitor
        channels = monitor.channels

        for i in range(1, len(model.layers)):
            channel_name = self.estimate_set + "_" + model.layers[i - 1].layer_name + "_percentage_activated"
            channel = channels[channel_name]
            val_record = channel.val_record
            last_activation_estimate = val_record[-1]

            dropout_include_prob = 1.0 - last_activation_estimate * self.proportion_to_avg_act
            if dropout_include_prob < 0.2:
                dropout_include_prob = 0.2
            elif dropout_include_prob > 0.8:
                dropout_include_prob = 0.8
            dropout_scale = 1.0 / dropout_include_prob
            #algorithm.cost.input_include_probs[layer.layer_name] = dropout_include_prob
            #algorithm.cost.input_scales[layer.layer_name] = dropout_scale
            algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_include_prob))
            algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_scale))
