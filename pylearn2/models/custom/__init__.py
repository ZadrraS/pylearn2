import numpy as np
import theano.tensor as T
from pylearn2.utils import wraps
from pylearn2.utils import sharedX
from pylearn2.models.mlp import Linear
from pylearn2.models.mlp import Layer
from theano.compat.python2x import OrderedDict

class RandomfiedLinear(Linear):
    """
    Like ReLU, only with randomized lower threshold

    Parameters
    ----------
    kwargs : dict
        Keyword arguments to pass to `Linear` class constructor.
    """

    def __init__(self, rect_stdev = 0.1, **kwargs):
        super(RandomfiedLinear, self).__init__(**kwargs)
        self.rect_stdev = rect_stdev

    @wraps(Layer.fprop)
    def fprop(self, state_below):

        p = self._linear_part(state_below)
        p = p * (p > self.rectifier_thresh) + self.rectifier_thresh * (p < self.rectifier_thresh)
        return p

    @wraps(Layer.cost)
    def cost(self, *args, **kwargs):
        raise NotImplementedError()

    @wraps(Layer.set_input_space)
    def set_input_space(self, space):
        super(RandomfiedLinear, self).set_input_space(space)

        rng = self.mlp.rng
        self.rectifier_thresh = rng.randn(self.dim) * self.rect_stdev
        self.rectifier_thresh = sharedX(self.rectifier_thresh)        

    @wraps(Layer.get_layer_monitoring_channels)
    def get_layer_monitoring_channels(self, state_below=None,
                                    state=None, targets=None):
        #rval = super(RectifiedLinear, self).get_layer_monitoring_channels(state_below = state_below, state = state, targets = targets)
        rval = OrderedDict([])
        rval['percentage_activated'] = T.cast(T.gt(state, self.rectifier_thresh).mean(), dtype = state.dtype)

        return rval