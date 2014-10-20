"""
Callable activation functions for autoencoders and other models.
"""

import theano.tensor as T

class ActivationFunction(object):
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

class RectifiedLinear(ActivationFunction):
    def __init__(self, left_slope = 0.0, rect_point = 0.0):
        self.left_slope = left_slope
        self.rect_point = rect_point

    def __call__(self, inputs):
        return T.switch(T.ge(inputs, self.rect_point), inputs, self.rect_point)
        #return inputs * (inputs > rect_point) + self.left_slope * inputs * (inputs < rect_point)