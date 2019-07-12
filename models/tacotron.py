import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder
from util.infolog import log
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet, encoderRefAudio
from .rnn_wrappers import FrameProjection, StopProjection, TacotronDecoderWrapper
from .attention import LocationSensitiveAttention
from .custom_decoder import CustomDecoder


class Tacotron():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, input_lengths, ppgs, mel_targets=None, linear_targets=None, speakers=None):

    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      # batch_size = tf.shape(ppgs)[0]
      hp = self._hparams

      # Pre-net: [batch, time, feat]
      encoder_steps = tf.gather(input_lengths, tf.argmax(input_lengths))
      if hp.addition_vector == 'None':
        prenet_outputs = prenet(ppgs, is_training, hp.prenet_depths)
      elif hp.addition_vector in ['onehot'] and hp.speaker_emb > 0:
        prenet_outputs = prenet(ppgs, is_training, hp.prenet_depths)
        encoder_speakers = tf.layers.dense(speakers, units=hp.speaker_emb, activation=tf.nn.tanh,name='dense')
        encoder_speakers = tf.expand_dims(encoder_speakers, 1) # [batch_size, 1, speaker_emb]
        encoder_speakers = tf.tile(encoder_speakers, (1, encoder_steps, 1))
        prenet_outputs = tf.concat([encoder_speakers, prenet_outputs], axis=2)
        

      if hp.pred_step == 2:
        # CBHG1: mel-scale [batch, time, 80]
        pred_mel = encoder_cbhg(prenet_outputs, input_lengths, is_training, hp.encoder_depth, 'cbhg_mel') # [80->128]
        pred_mel = tf.layers.dense(pred_mel, hp.num_mels, name='pred_mel') # [128->80]

        # CBHG2: spec-scale [batch, time, 257]
        post_outputs = post_cbhg(pred_mel, hp.num_mels, is_training, hp.postnet_depth)  # [80->128]
        pred_spec = tf.layers.dense(post_outputs, hp.num_freq) # [128->257]
      elif hp.pred_step == 1:
        # CBHG1: mel-scale [batch, time, 80]
        post_outputs = encoder_cbhg(prenet_outputs, input_lengths, is_training, hp.encoder_depth, 'cbhg_spec') # [80->128]
        pred_spec = tf.layers.dense(post_outputs, hp.num_freq, name='pred_spec') # [128->80]


      self.speakers = speakers
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      self.input_lengths = input_lengths
      self.ppgs = ppgs

      if hp.pred_step == 2: self.pred_mel = pred_mel
      self.pred_spec = pred_spec
      log('Initialized Tacotron model. Dimensions: ')
      log('  pred_spec:               {}'.format(pred_spec.shape))
      if hp.pred_step == 2: log('  pred_mel:                {}'.format(pred_mel.shape))


  def add_loss(self):
    '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
    with tf.variable_scope('loss') as scope:
      hp = self._hparams
      if hp.pred_step == 2:
        self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.pred_mel))
        l1 = tf.abs(self.linear_targets - self.pred_spec)
        self.linear_loss = tf.reduce_mean(l1)
        self.loss = self.mel_loss + self.linear_loss
      elif hp.pred_step == 1:
        l1 = tf.abs(self.linear_targets - self.pred_spec)
        self.linear_loss = tf.reduce_mean(l1)
        self.loss = self.linear_loss



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
