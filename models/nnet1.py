import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder
from util.infolog import log
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet, encoderRefAudio
from .rnn_wrappers import FrameProjection, StopProjection, TacotronDecoderWrapper
from .attention import LocationSensitiveAttention
from .custom_decoder import CustomDecoder


class NNet1():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, input_lengths, linear_targets, ppgs=None, mel_targets=None, speakers=None):

    with tf.variable_scope('inference') as scope:
      is_training = ppgs is not None
      hp = self._hparams

      # Pre-net: [batch, time, feat]
      # encoder_steps = tf.gather(input_lengths, tf.argmax(input_lengths))
      prenet_outputs = prenet(linear_targets, is_training, hp.prenet_depths)
      post_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training, hp.encoder_depth, 'cbhg_ppgs') # [80->128]
      logits = tf.layers.dense(post_outputs, hp.num_ppgs, name='pred_ppgs') # [128->80]
      pred_ppgs = tf.nn.softmax(logits, name='ppgs')

      self.speakers = speakers
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      self.input_lengths = input_lengths
      self.ppgs = ppgs

      self.logits = logits
      self.pred_ppgs = pred_ppgs

      log('Initialized NNet1 model. Dimensions: ')
      log('  pred_ppgs:               {}'.format(pred_ppgs.shape))


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      ppgs = tf.to_int32(tf.argmax(self.ppgs, axis=-1))
      istarget = tf.sign(tf.abs(tf.reduce_sum(tf.abs(self.linear_targets), -1)))
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=ppgs) # logits: [batch, 201, 218]
      loss *= istarget
      self.loss = tf.reduce_mean(loss)


  def add_optimizer(self, global_step):
    '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

    Args:
      global_step: int32 scalar Tensor representing current global step in training
    '''
    with tf.variable_scope('optimizer') as scope:
      hp = self._hparams
      if hp.decay_learning_rate:
        self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
      else:
        self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
      optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
      gradients, variables = zip(*optimizer.compute_gradients(self.loss))
      self.gradients = gradients
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

      # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
      # https://github.com/tensorflow/tensorflow/issues/1122
      with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
  # Noam scheme from tensor2tensor:
  warmup_steps = 4000.0
  step = tf.cast(global_step + 1, dtype=tf.float32)
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
