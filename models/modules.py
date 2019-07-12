import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell


def prenet(inputs, is_training, layer_sizes, scope=None):
  x = inputs
  # drop_rate = 0.5 if is_training else 0.0
  drop_rate = 0.2 if is_training else 0.0
  with tf.variable_scope(scope or 'prenet'):
    for i, size in enumerate(layer_sizes):
      dense = tf.layers.dense(x, units=size, activation=tf.nn.tanh, name='dense_%d' % (i+1))
      x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i+1))
  return x


def encoder_cbhg(inputs, input_lengths, is_training, depth, scope='encoder_cbhg'):
  input_channels = inputs.get_shape()[2]
  return cbhg(
    inputs,
    input_lengths,
    is_training,
    scope=scope,
    K=16,
    projections=[128, input_channels],
    depth=depth)


def post_cbhg(inputs, input_dim, is_training, depth):
  return cbhg(
    inputs,
    None,
    is_training,
    scope='post_cbhg',
    K=8,
    projections=[256, input_dim],
    depth=depth)


def cbhg(inputs, input_lengths, is_training, scope, K, projections, depth):
  with tf.variable_scope(scope):
    with tf.variable_scope('conv_bank'):
      # Convolution bank: concatenate on the last axis to stack channels from all convolutions
      conv_outputs = tf.concat(
        [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K+1)],
        axis=-1
      )

    # Maxpooling:
    maxpool_output = tf.layers.max_pooling1d(
      conv_outputs,
      pool_size=2,
      strides=1,
      padding='same')

    # Two projection layers:
    proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')
    proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')

    # Residual connection:
    highway_input = proj2_output + inputs

    half_depth = depth // 2
    assert half_depth*2 == depth, 'encoder and postnet depths must be even.'

    # Handle dimensionality mismatch:
    if highway_input.shape[2] != half_depth:
      highway_input = tf.layers.dense(highway_input, half_depth)

    # 4-layer HighwayNet:
    for i in range(4):
      highway_input = highwaynet(highway_input, 'highway_%d' % (i+1), half_depth)
    rnn_input = highway_input

    # Bidirectional RNN
    outputs, states = tf.nn.bidirectional_dynamic_rnn(
      GRUCell(half_depth),
      GRUCell(half_depth),
      rnn_input,
      sequence_length=input_lengths,
      dtype=tf.float32)
    return tf.concat(outputs, axis=2)  # Concat forward and backward


def highwaynet(inputs, scope, depth):
  with tf.variable_scope(scope):
    H = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.relu,
      name='H')
    T = tf.layers.dense(
      inputs,
      units=depth,
      activation=tf.nn.sigmoid,
      name='T',
      bias_initializer=tf.constant_initializer(-1.0))
    return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
  with tf.variable_scope(scope):
    conv1d_output = tf.layers.conv1d(
      inputs,
      filters=channels,
      kernel_size=kernel_size,
      activation=activation,
      padding='same')
    return tf.layers.batch_normalization(conv1d_output, training=is_training)


# no dropout in conv2d
def conv2d(inputs, kernel_size, strides, channels, activation, is_training, scope):

  with tf.variable_scope(scope):
    conv2d_output = tf.layers.conv2d(
      inputs=inputs,
      filters=channels,
      kernel_size=kernel_size,
      strides=strides,
      activation=None,
      padding='SAME')
    batched = tf.layers.batch_normalization(conv2d_output, training=is_training)
    activated = activation(batched)
    return activated

def gru(inputs, num_units=None, bidirection=False, scope="gru", reuse=None):
    '''Applies a GRU.
    
    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: An int. The number of hidden units.
      bidirection: A boolean. If True, bidirectional results 
        are concatenated.
      scope: Optional scope for `variable_scope`.  
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.
        
    Returns:
      If bidirection is True, a 3d tensor with shape of [N, T, 2*num_units],
        otherwise [N, T, num_units].
    '''
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = inputs.get_shape().as_list()[-1]
            
        cell = tf.contrib.rnn.GRUCell(num_units)  
        if bidirection: 
            cell_bw = tf.contrib.rnn.GRUCell(num_units)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
            return tf.concat(outputs, 2)  
        else:
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
            return outputs

def encoderRefAudio(inputs, batch_size, is_training, scope='refAudio_emb'):
  """ref audio embedding 
      inputs: [N, T, n_mels] -> [batch, features]
  """
  with tf.variable_scope(scope):
    x = inputs
    x = tf.expand_dims(x, -1) # [N, T, n_mels, 1]

    # 6-Layer Strided Conv2D -> (N, T/64, n_mels/64, 128)
    for i, channel in enumerate([32,32,64,64,128,128]):
      x = conv2d(x, 3, 2, channel, tf.nn.relu, is_training, 'conv2d_{}'.format(i + 1))
    
    # Unroll -> (N, T/64, 128*n_mels/64)
    N, _, W, C = x.get_shape().as_list()
    x = tf.reshape(x, [batch_size, -1, W*C])

    # GRU -> (N, T/64, 128) -> (N, 128)
    x = gru(x, num_units=128, bidirection=False, scope="gru")
    x = x[:, -1, :]

    # FC -> (N, 128)
    x = tf.layers.dense(x, 128, activation=tf.nn.tanh)

    return x

