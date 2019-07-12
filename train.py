import argparse
from datetime import datetime
import math
import os
import subprocess
import time
import tensorflow as tf
import traceback
import numpy as np

from datasets.datafeeder import DataFeeder
from hparams import hparams, hparams_debug_string
from models import create_model
from util import audio, infolog, plot, ValueWindow
log = infolog.log

# os.environ['CUDA_VISIBLE_DEVICES']='0'


def add_stats(model):
  with tf.variable_scope('stats') as scope:
    tf.summary.scalar('learning_rate', model.learning_rate)
    tf.summary.scalar('loss', model.loss)
    return tf.summary.merge_all()


def time_string():
  return datetime.now().strftime('%Y-%m-%d %H:%M')


def train(log_dir, pretrain_log_dir, args):
  checkpoint_path = os.path.join(log_dir, 'model.ckpt')
  input_path = os.path.join(args.base_dir, args.input)
  log('Checkpoint path: %s' % checkpoint_path)
  log('Loading training data from: %s' % input_path)
  log('Using model: %s' % args.model)
  log(hparams_debug_string())

  # Set up DataFeeder:
  ### input_path: linear, mel, frame_num, ppgs
  coord = tf.train.Coordinator()
  with tf.variable_scope('datafeeder') as scope:
    feeder = DataFeeder(coord, input_path, hparams)

  # Set up model:
  global_step = tf.Variable(0, name='global_step', trainable=False)
  with tf.variable_scope('model') as scope:
    model = create_model(args.model, hparams)
    model.initialize(input_lengths=feeder.input_lengths, mel_targets=feeder.mel_targets,
                     linear_targets=feeder.linear_targets, ppgs=feeder.ppgs, speakers=feeder.speakers)
    model.add_loss()
    model.add_optimizer(global_step)
    stats = add_stats(model)

  # Bookkeeping:
  step = 0
  time_window = ValueWindow(100)
  loss_window = ValueWindow(100)
  acc_window = ValueWindow(100)
  saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

  # Train!
  with tf.Session() as sess:
    try:
      summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
      sess.run(tf.global_variables_initializer())
      
      try:
        if pretrain_log_dir != None:
          checkpoint_state = tf.train.get_checkpoint_state(pretrain_log_dir)
        else:
          checkpoint_state = tf.train.get_checkpoint_state(log_dir)
        if (checkpoint_state and checkpoint_state.model_checkpoint_path):
          log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
          saver.restore(sess, checkpoint_state.model_checkpoint_path)
        else:
          log('No model to load at {}'.format(log_dir), slack=True)
          saver.save(sess, checkpoint_path, global_step=global_step)
      except tf.errors.OutOfRangeError as e:
        log('Cannot restore checkpoint: {}'.format(e), slack=True)

      feeder.start_in_session(sess)

      while not coord.should_stop():
        start_time = time.time()

        ### how to run training
        if args.model == 'tacotron':
          step, loss, opt = sess.run([global_step, model.loss, model.optimize])
          time_window.append(time.time() - start_time)
          loss_window.append(loss)
          message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
            step, time_window.average, loss, loss_window.average)
          log(message, slack=(step % args.checkpoint_interval == 0))
        elif args.model == 'nnet1':
          step, loss, opt, ppgs, logits = sess.run([global_step, model.loss, model.optimize, model.ppgs, model.logits])
          ## cal acc
          ppgs = np.argmax(ppgs, axis=-1) # (N, 201, )
          logits = np.argmax(logits, axis=-1) # (N, 201, )
          num_hits = np.sum(np.equal(ppgs, logits))
          num_targets = np.shape(ppgs)[0] * np.shape(ppgs)[1]
          acc = num_hits / num_targets
          ## summerize
          time_window.append(time.time() - start_time)
          loss_window.append(loss)
          acc_window.append(acc)
          message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f, acc=%.05f, avg_acc=%.05f]' % (
            step, time_window.average, loss, loss_window.average, acc, acc_window.average)
          log(message, slack=(step % args.checkpoint_interval == 0))
        else:
          print ('input error!!')
          assert 1==0

        ### save model and logs
        if loss > 100 or math.isnan(loss):
          log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
          raise Exception('Loss Exploded')

        if step % args.summary_interval == 0:
          log('Writing summary at step: %d' % step)
          summary_writer.add_summary(sess.run(stats), step)

        if step % args.checkpoint_interval == 0:
          log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
          saver.save(sess, checkpoint_path, global_step=step)

    except Exception as e:
      log('Exiting due to exception: %s' % e, slack=True)
      traceback.print_exc()
      coord.request_stop(e)

# python train.py --input='LJspeech-training-world.txt' --name='VC_LJspeech_world_lr0.0003' --hparams=initial_learning_rate=0.0003,num_freq=62,pred_step=1
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--base_dir', default=os.path.expanduser('./'))
  parser.add_argument('--input', default='training/train.txt')
  parser.add_argument('--model', default='tacotron')
  parser.add_argument('--name', help='Name of the run. Used for logging. Defaults to model name.')
  parser.add_argument('--pretrain_name', default='', help='where to gain pretrained model')
  parser.add_argument('--hparams', default='',
    help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  parser.add_argument('--restore_step', type=int, help='Global step to restore from checkpoint.')
  parser.add_argument('--summary_interval', type=int, default=100,
    help='Steps between running summary ops.')
  parser.add_argument('--checkpoint_interval', type=int, default=1000,
    help='Steps between writing checkpoints.')
  parser.add_argument('--slack_url', help='Slack webhook URL to get periodic reports.')
  parser.add_argument('--tf_log_level', type=int, default=1, help='Tensorflow C++ log level.')
  parser.add_argument('--git', action='store_true', help='If set, verify that the client is clean.')
  args = parser.parse_args()
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
  run_name = args.name or args.model
  log_dir = os.path.join(args.base_dir, 'logs-%s' % run_name)
  pretrain_log_dir = os.path.join(args.base_dir, 'logs-%s' % args.pretrain_name) if args.pretrain_name!='' else None
  os.makedirs(log_dir, exist_ok=True)
  infolog.init(os.path.join(log_dir, 'train.log'), run_name, args.slack_url)
  hparams.parse(args.hparams)
  train(log_dir, pretrain_log_dir, args)

if __name__ == '__main__':
  main()
