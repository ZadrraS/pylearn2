from pylearn2.space import NullSpace
from theano import config
from pylearn2.utils import sharedX
import functools
import numpy as np
from pylearn2.train_extensions import TrainExtension

class DropoutScaler(TrainExtension):
    def __init__(self, estimate_set = "train"):
        self.estimate_set = estimate_set

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

            dropout_include_prob = 0.25 / last_activation_estimate
            if dropout_include_prob < 0.2:
                dropout_include_prob = 0.2
            elif dropout_include_prob > 0.8:
                dropout_include_prob = 0.8
            dropout_scale = 1.0 / dropout_include_prob
            #algorithm.cost.input_include_probs[layer.layer_name] = dropout_include_prob
            #algorithm.cost.input_scales[layer.layer_name] = dropout_scale
            algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_include_prob))
            algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_scale))

class DropoutSwitcher(TrainExtension):
    def __init__(self, trigger_epochs, dropout_prob_values, scale_learning_rate = False):
        self.trigger_epochs = trigger_epochs
        self.dropout_prob_values = dropout_prob_values
        self.scale_learning_rate = scale_learning_rate
        self.count = 0

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

            channel_name = layer_name
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

        if self.scale_learning_rate:
            drop_product = 0.0
            for i in range(0, len(model.layers)):
                drop_product += algorithm.cost.input_include_probs[model.layers[i].layer_name].get_value()

            drop_product /= len(model.layers)
            model.layers[-1].W_lr_scale = drop_product * drop_product
        self.count += 1

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        monitor = model.monitor
        channels = monitor.channels

        for i in range(len(self.trigger_epochs)):
            if self.trigger_epochs[i] == self.count:
                for layer_name in self.dropout_prob_values[i]:
                    dropout_include_prob = self.dropout_prob_values[i][layer_name]
                    dropout_scale = 1.0 / dropout_include_prob
                    algorithm.cost.input_include_probs[layer_name].set_value(np.cast[config.floatX](dropout_include_prob))
                    algorithm.cost.input_scales[layer_name].set_value(np.cast[config.floatX](dropout_scale))

                if self.scale_learning_rate:
                    drop_product = 0.0
                    for i in range(0, len(model.layers)):
                        drop_product += algorithm.cost.input_include_probs[model.layers[i].layer_name].get_value()

                    drop_product /= len(model.layers)
                    model.layers[-1].W_lr_scale = drop_product * drop_product

                break

        self.count += 1

class DropoutActivationLinearScaler(TrainExtension):
    def __init__(self, decay_factor, min_incl_prob, max_incl_prob, target_act_fraction, max_target_act_fraction, estimate_set = "train"):
        self.decay_factor = decay_factor
        self.min_incl_prob = min_incl_prob
        self.max_incl_prob = max_incl_prob
        self.target_act_fraction = sharedX(target_act_fraction)
        self.max_target_act_fraction = max_target_act_fraction
        self.estimate_set = estimate_set

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:
            if layer_name in algorithm.cost.input_include_probs:
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

        monitor.add_channel(
                name="target_act_fraction",
                ipt=None,
                val=self.target_act_fraction,
                data_specs=(NullSpace(), ''),
                dataset=dataset)

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        monitor = model.monitor
        channels = monitor.channels

        droppable_layer_sequence = []
        for i in range(0, len(model.layers)):
            if model.layers[i].layer_name in algorithm.cost.input_include_probs:
                droppable_layer_sequence.append(model.layers[i].layer_name)

        for i in range(1, len(droppable_layer_sequence)):
            if droppable_layer_sequence[i] not in algorithm.cost.constant_layers:
                prev_l_name = droppable_layer_sequence[i - 1]
                channel_name = self.estimate_set + "_" + prev_l_name + "_percentage_activated"
                channel = channels[channel_name]
                val_record = channel.val_record
                last_activation_estimate = val_record[-1]

                dropout_include_prob = self.target_act_fraction.get_value() / last_activation_estimate
                if dropout_include_prob < self.min_incl_prob:
                    dropout_include_prob = self.min_incl_prob
                elif dropout_include_prob > self.max_incl_prob:
                    dropout_include_prob = self.max_incl_prob

                algorithm.cost.set_dropout_value(droppable_layer_sequence[i], dropout_include_prob)

        act_frac_val = self.target_act_fraction.get_value()
        act_frac_val += self.decay_factor
        if act_frac_val > self.max_target_act_fraction:
            act_frac_val = self.max_target_act_fraction

        self.target_act_fraction.set_value(np.cast[config.floatX](act_frac_val))

        if droppable_layer_sequence[0] not in algorithm.cost.constant_layers:
            l0_dropout_include_prob = self.target_act_fraction.get_value() * 2
            if l0_dropout_include_prob > self.max_incl_prob:
                l0_dropout_include_prob = self.max_incl_prob
            elif l0_dropout_include_prob < self.min_incl_prob:
                l0_dropout_include_prob = self.min_incl_prob

            algorithm.cost.set_dropout_value(droppable_layer_sequence[0], l0_dropout_include_prob)

class DropoutActivationSetter(TrainExtension):
    def __init__(self, min_incl_prob, max_incl_prob, target_act_fraction, estimate_set = "train"):
        self.min_incl_prob = min_incl_prob
        self.max_incl_prob = max_incl_prob
        self.target_act_fraction = sharedX(target_act_fraction)
        self.estimate_set = estimate_set

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:

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

        monitor.add_channel(
                name="target_act_fraction",
                ipt=None,
                val=self.target_act_fraction,
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

            dropout_include_prob = self.target_act_fraction.get_value() / last_activation_estimate
            if dropout_include_prob < self.min_incl_prob:
                dropout_include_prob = self.min_incl_prob
            elif dropout_include_prob > self.max_incl_prob:
                dropout_include_prob = self.max_incl_prob
            dropout_scale = 1.0 / dropout_include_prob

            #algorithm.cost.input_include_probs[layer.layer_name] = dropout_include_prob
            #algorithm.cost.input_scales[layer.layer_name] = dropout_scale
            algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_include_prob))
            algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_scale))

        #l0_dropout_include_prob = self.target_act_fraction.get_value() * 2
        #if l0_dropout_include_prob > self.max_incl_prob:
        #    l0_dropout_include_prob = self.max_incl_prob
        #elif l0_dropout_include_prob < self.min_incl_prob:
        #    l0_dropout_include_prob = self.min_incl_prob
        #algorithm.cost.input_include_probs[model.layers[0].layer_name].set_value(np.cast[config.floatX](l0_dropout_include_prob))
        #algorithm.cost.input_scales[model.layers[0].layer_name].set_value(np.cast[config.floatX](1.0 / l0_dropout_include_prob))

class DropoutActivationExponentialScaler(TrainExtension):
    def __init__(self, decay_factor, min_incl_prob, max_incl_prob, target_act_fraction, max_target_act_fraction, estimate_set = "train"):
        self.decay_factor = decay_factor
        self.min_incl_prob = min_incl_prob
        self.max_incl_prob = max_incl_prob
        self.target_act_fraction = sharedX(target_act_fraction)
        self.max_target_act_fraction = max_target_act_fraction
        self.estimate_set = estimate_set

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:

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

        monitor.add_channel(
                name="target_act_fraction",
                ipt=None,
                val=self.target_act_fraction,
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

            dropout_include_prob = self.target_act_fraction.get_value() / last_activation_estimate
            if dropout_include_prob < self.min_incl_prob:
                dropout_include_prob = self.min_incl_prob
            elif dropout_include_prob > self.max_incl_prob:
                dropout_include_prob = self.max_incl_prob
            dropout_scale = 1.0 / dropout_include_prob

            #algorithm.cost.input_include_probs[layer.layer_name] = dropout_include_prob
            #algorithm.cost.input_scales[layer.layer_name] = dropout_scale
            algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_include_prob))
            algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_scale))

        act_frac_val = self.target_act_fraction.get_value()
        act_frac_val *= self.decay_factor
        if act_frac_val > self.max_target_act_fraction:
            act_frac_val = self.max_target_act_fraction

        self.target_act_fraction.set_value(np.cast[config.floatX](act_frac_val))

        l0_dropout_include_prob = algorithm.cost.input_include_probs[model.layers[1].layer_name].get_value()
        algorithm.cost.input_include_probs[model.layers[0].layer_name].set_value(np.cast[config.floatX](l0_dropout_include_prob))
        algorithm.cost.input_scales[model.layers[0].layer_name].set_value(np.cast[config.floatX](1.0 / l0_dropout_include_prob))

class DropoutLinearAnnealer(TrainExtension):
    def __init__(self, decay_factor, max_incl_prob):
        self.decay_factor = decay_factor
        self.max_incl_prob = max_incl_prob

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:
            channel_name = layer_name
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
        for i in range(0, len(model.layers)):
            if model.layers[i].layer_name in algorithm.cost.input_include_probs:
                dropout_prob = algorithm.cost.input_include_probs[model.layers[i].layer_name].get_value()
                dropout_prob += self.decay_factor
                if dropout_prob > self.max_incl_prob:
                    dropout_prob = self.max_incl_prob
                algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_prob))
                algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](1.0 / dropout_prob))

class DropoutLinearMeanActivationSetter(TrainExtension):
    def __init__(self, min_keep_prob = 0.4, max_keep_prob = 0.8, estimate_set = "train"):
        self.min_keep_prob = min_keep_prob
        self.max_keep_prob = max_keep_prob
        self.estimate_set = estimate_set

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:
            channel_name = layer_name
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
            channel_name = self.estimate_set + "_" + model.layers[i - 1].layer_name + "_mean_activation"
            channel = channels[channel_name]
            val_record = channel.val_record
            last_activation_estimate = val_record[-1]

            act_scaled = (last_activation_estimate - 0.01) / (0.4 - 0.01)
            if act_scaled < 0.0:
                act_scaled = 0.0
            elif act_scaled > 1.0:
                act_scaled = 1.0
            dropout_prob = act_scaled * (self.max_keep_prob - self.min_keep_prob) + self.min_keep_prob

            algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_prob))
            algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](1.0 / dropout_prob))
            
class DropoutExponentialAnnealer(TrainExtension):
    def __init__(self, decay_factor, max_incl_prob, scale_learning_rate = False, keep_input_constant = False):
        self.decay_factor = decay_factor
        self.max_incl_prob = max_incl_prob
        self.scale_learning_rate = scale_learning_rate
        self.keep_input_constant = keep_input_constant

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:
            channel_name = layer_name
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
        if self.scale_learning_rate:
            drop_product = 0.0
            for i in range(0, len(model.layers)):
                if model.layers[i].layer_name in algorithm.cost.input_include_probs:
                    drop_product += algorithm.cost.input_include_probs[model.layers[i].layer_name].get_value()
            
            drop_product /= len(model.layers)
            model.layers[-1].W_lr_scale = drop_product * drop_product

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        start_it = 0
        if self.keep_input_constant:
            start_it = 1
        for i in range(start_it, len(model.layers)):
            if model.layers[i].layer_name in algorithm.cost.input_include_probs:
                dropout_prob = algorithm.cost.input_include_probs[model.layers[i].layer_name].get_value()
                dropout_prob *= self.decay_factor
                if dropout_prob > self.max_incl_prob:
                    dropout_prob = self.max_incl_prob
                algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_prob))
                algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](1.0 / dropout_prob))

        if self.scale_learning_rate:
            drop_product = 0.0
            for i in range(0, len(model.layers)):
                if model.layers[i].layer_name in algorithm.cost.input_include_probs:
                    drop_product += algorithm.cost.input_include_probs[model.layers[i].layer_name].get_value()

            drop_product /= len(model.layers)
            model.layers[-1].W_lr_scale = drop_product * drop_product

class DropoutExponentialAnnealerLayerwise(TrainExtension):
    def __init__(self, decay_factor, max_incl_prob):
        self.decay_factor = decay_factor
        self.max_incl_prob = max_incl_prob

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor

        dropout_include_prob = {}
        dropout_scale = {}
        for layer_name in algorithm.cost.input_include_probs:
            channel_name = layer_name
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
        for i in range(0, len(model.layers)):
            dropout_prob = algorithm.cost.input_include_probs[model.layers[i].layer_name].get_value()
            dropout_prob *= self.decay_factor
            layers_done = False
            if dropout_prob > self.max_incl_prob:
                dropout_prob = self.max_incl_prob
                layers_done = True
            algorithm.cost.input_include_probs[model.layers[i].layer_name].set_value(np.cast[config.floatX](dropout_prob))
            algorithm.cost.input_scales[model.layers[i].layer_name].set_value(np.cast[config.floatX](1.0 / dropout_prob))

            if not layers_done:
                break


class LinearizationProbLinearAnnealer(TrainExtension):
    def __init__(self, decay_factor, min_lin_prob):
        self.decay_factor = decay_factor
        self.min_lin_prob = min_lin_prob

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor
        for i in range(0, len(model.layers) - 1):
            channel_name = model.layers[i].layer_name
            monitor.add_channel(
                name=channel_name + "_linearization_prob",
                ipt=None,
                val=model.layers[i].linearize_prob,
                data_specs=(NullSpace(), ''),
                dataset=dataset)

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        for i in range(0, len(model.layers) - 1):
            lin_prob =  model.layers[i].linearize_prob.get_value()
            lin_prob -= self.decay_factor
            if lin_prob < self.min_lin_prob:
                lin_prob = self.min_lin_prob

            model.layers[i].linearize_prob.set_value(np.cast[config.floatX](lin_prob))

class RandomlyRectifyingLinearAnnealer(TrainExtension):
    def __init__(self, decay_factor, min_std = 0.0):
        self.decay_factor = decay_factor
        self.min_std = min_std

    @functools.wraps(TrainExtension.setup)
    def setup(self, model, dataset, algorithm):
        monitor = model.monitor
        for i in range(0, len(model.layers) - 1):
            channel_name = model.layers[i].layer_name
            monitor.add_channel(
                name=channel_name + "_randomfier_std",
                ipt=None,
                val=model.layers[i].std,
                data_specs=(NullSpace(), ''),
                dataset=dataset)

    @functools.wraps(TrainExtension.on_monitor)
    def on_monitor(self, model, dataset, algorithm):
        for i in range(0, len(model.layers) - 1):
            lin_prob =  model.layers[i].std.get_value()
            lin_prob -= self.decay_factor
            if lin_prob < self.min_std:
                lin_prob = self.min_std

            model.layers[i].std.set_value(np.cast[config.floatX](lin_prob))
