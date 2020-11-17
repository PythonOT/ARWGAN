"""
From pull request:
https://github.com/tensorflow/tensorflow/pull/21276
"""
import tensorflow as tf
from tensorflow.python.keras.layers import Wrapper
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.ops import nn_impl
from tensorflow.python.keras import initializers

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.eager import context


class WeightNorm(Wrapper):
    """ This wrapper reparameterizes a layer by decoupling the weight's
    magnitude and direction. This speeds up convergence by improving the
    conditioning of the optimization problem.

    Weight Normalization: A Simple Reparameterization to Accelerate
    Training of Deep Neural Networks: https://arxiv.org/abs/1602.07868
    Tim Salimans, Diederik P. Kingma (2016)

    WeightNorm wrapper works for keras and tf layers.

    ```python
      net = WeightNorm(tf.keras.layers.Conv2D(2, 2, activation='relu'),
             input_shape=(32, 32, 3), data_init=True)(x)
      net = WeightNorm(tf.keras.layers.Conv2D(16, 5, activation='relu'),
                       data_init=True)
      net = WeightNorm(tf.keras.layers.Dense(120, activation='relu'),
                       data_init=True)(net)
      net = WeightNorm(tf.keras.layers.Dense(n_classes),
                       data_init=True)(net)
    ```

    Arguments:
      layer: a `Layer` instance.
      data_init: If `True` use data dependent variable initialization

    Raises:
      ValueError: If not initialized with a `Layer` instance.
      ValueError: If `Layer` does not contain a `kernel` of weights
      NotImplementedError: If `data_init` is True and running graph execution
    """
    def __init__(self, layer, data_init=False, **kwargs):
        if not isinstance(layer, Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

        if not context.executing_eagerly() and data_init:
            raise NotImplementedError(
                'Data dependent variable initialization is not available for '
                'graph execution')

        self.initialized = True
        if data_init:
            self.initialized = False

        super(WeightNorm, self).__init__(layer, **kwargs)
        self._track_checkpointable(layer, name='layer')

    def _compute_weights(self):
        """Generate weights by combining the direction of weight vector
         with it's norm """
        with variable_scope.variable_scope('compute_weights'):
            self.layer.kernel = nn_impl.l2_normalize(
                self.layer.v, axis=self.norm_axes) * self.layer.g

    def _init_norm(self, weights):
        """Set the norm of the weight vector"""
        from tensorflow.python.ops.linalg_ops import norm
        with variable_scope.variable_scope('init_norm'):
            flat = array_ops.reshape(weights, [-1, self.layer_depth])
            return array_ops.reshape(norm(flat, axis=0), (self.layer_depth,))

    def _data_dep_init(self, inputs):
        """Data dependent initialization for eager execution"""
        from tensorflow.python.ops.nn import moments
        from tensorflow.python.ops.math_ops import sqrt

        with variable_scope.variable_scope('data_dep_init'):
            # Generate data dependent init values
            activation = self.layer.activation
            self.layer.activation = None
            x_init = self.layer.call(inputs)
            m_init, v_init = moments(x_init, self.norm_axes)
            scale_init = 1. / sqrt(v_init + 1e-10)

        # Assign data dependent init values
        self.layer.g = self.layer.g * scale_init
        self.layer.bias = (-m_init * scale_init)
        self.layer.activation = activation
        self.initialized = True

    def build(self, input_shape):
        """Build `Layer`"""
        input_shape = tensor_shape.TensorShape(input_shape).as_list()
        self.input_spec = InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = False

            if not hasattr(self.layer, 'kernel'):
                raise ValueError(
                    '`WeightNorm` must wrap a layer that'
                    ' contains a `kernel` for weights'
                )

            # The kernel's filter or unit dimension is -1
            self.layer_depth = int(self.layer.kernel.shape[-1])
            self.norm_axes = list(range(self.layer.kernel.shape.ndims - 1))

            self.layer.v = self.layer.kernel
            self.layer.g = self.layer.add_variable(
                name="g",
                shape=(self.layer_depth,),
                initializer=initializers.get('ones'),
                dtype=self.layer.kernel.dtype,
                trainable=True)

            with ops.control_dependencies([self.layer.g.assign(
                    self._init_norm(self.layer.v))]):
                self._compute_weights()

            self.layer.built = True

        super(WeightNorm, self).build()
        self.built = True

    def call(self, inputs):
        """Call `Layer`"""
        if context.executing_eagerly():
            if not self.initialized:
                self._data_dep_init(inputs)
            self._compute_weights()  # Recompute weights for each forward pass

        output = self.layer.call(inputs)
        return output

    def compute_output_shape(self, input_shape):
        return tensor_shape.TensorShape(
            self.layer.compute_output_shape(input_shape).as_list())

def instance_norm(input, name="instance_norm"):
    
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        return scale*normalized + offset
    
def Layernorm(name, norm_axes, inputs):
    mean, var = tf.nn.moments(inputs, norm_axes, keep_dims=True)

    # Assume the 'neurons' axis is the first of norm_axes. This is the case for fully-connected and BCHW conv layers.
    n_neurons = inputs.get_shape().as_list()[norm_axes[0]]

    offset = tf.get_variable("offset", [n_neurons], initializer=tf.constant_initializer(0.0, dtype=tf.float32))
    scale = tf.get_variable("scale", [n_neurons], initializer=tf.constant_initializer(0.0, dtype=tf.float32))

    # Add broadcasting dims to offset and scale (e.g. BCHW conv data)
    offset = tf.reshape(offset, [-1] + [1 for i in range(len(norm_axes)-1)])
    scale = tf.reshape(scale, [-1] + [1 for i in range(len(norm_axes)-1)])

    result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-5)

    return result

def BatchNorm(input_, is_training=False):
    
    def _fused_batch_norm_training():
        return tf.nn.fused_batch_norm(inputs, scale, offset, epsilon=1e-5)
    
    def _fused_batch_norm_inference():
            # Version which blends in the current item's statistics
            batch_size = tf.cast(tf.shape(inputs)[0], 'float32')
            mean, var = tf.nn.moments(inputs, [2,3], keep_dims=True)
            mean = ((1./batch_size)*mean) + (((batch_size-1.)/batch_size)*moving_mean)[None,:,None,None]
            var = ((1./batch_size)*var) + (((batch_size-1.)/batch_size)*moving_variance)[None,:,None,None]
            return tf.nn.batch_normalization(inputs, mean, var, offset[None,:,None,None], scale[None,:,None,None], 1e-5), mean, var
        
    outputs, batch_mean, batch_var = tf.cond(is_training,
                                             _fused_batch_norm_training,
                                             _fused_batch_norm_inference)
    