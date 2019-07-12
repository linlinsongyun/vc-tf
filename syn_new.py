import io
import sys
import os
import glob
import math
import numpy as np
import argparse
import tensorflow as tf
from hparams import hparams
import librosa
from models import create_model
from util import audio


np.set_printoptions(threshold=np.inf)

class Synthesizer:

  def speaker2id(self, metadata_filename):
    speakers = []
    with open(metadata_filename, encoding='utf-8') as f:
      _metadata = [line.strip().split('|') for line in f]
      speakers = set([os.path.basename(x[-1]).split('_')[0] for x in _metadata]) # for 108 speakers
      speakers = [item for item in speakers]
      speakers.sort()

    speaker2vec = {}
    for ii, speaker in enumerate(speakers):
      temp = np.zeros((len(speakers), ))
      temp[ii] = 1.0
      speaker2vec[speaker] = temp
    self.speaker2vec=speaker2vec


  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    self.hop_length = hparams.sample_rate * hparams.frame_shift_ms // 1000
    self.n_timesteps = (hparams.duration * hparams.sample_rate) // self.hop_length + 1
    ppgs = tf.placeholder(tf.float32, [1, self.n_timesteps, hparams.num_ppgs], 'ppgs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    if hparams.addition_vector=='onehot':
      speakers = tf.placeholder(tf.float32, [1, len(self.speaker2vec)], 'speakers')
    else:
      speakers = None

    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(input_lengths=input_lengths, ppgs=ppgs, speakers=speakers)
      self.wav_output = self.model.pred_spec[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)

  ## input ppgs is [201, 176]
  def synthesize(self, ppgs):
    #####################################
    #####################################
    # speaker_emb = np.zeros((108, ))
    # speaker_emb[int(hparams.speaker_id)] = 1.0
    ppgs = librosa.util.fix_length(ppgs, self.n_timesteps, axis=0)

    feed_dict = {
      self.model.ppgs: np.asarray(ppgs[np.newaxis,:,:], dtype=np.float32),
      self.model.input_lengths: np.asarray([len(ppgs)], dtype=np.int32),
    }
    if hparams.addition_vector=='onehot':
      assert hparams.speaker_id in self.speaker2vec
      speaker_emb = self.speaker2vec[hparams.speaker_id]
      feed_dict[self.model.speakers] = np.asarray(speaker_emb[np.newaxis,:], dtype=np.float32)

    spec_pred = self.session.run(self.wav_output, feed_dict=feed_dict) # shape: [201, 257]
    assert len(spec_pred) == len(ppgs)

    return spec_pred

    
# python syn_new.py --checkpoint='logs-arcticBdl_vc_lr0.0001/model.ckpt-98000' --ppgs_root='arcticBdl/ppgs_176'
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint', required=True, help='Full path to model checkpoint')
  parser.add_argument('--ppgs_root', default='arcticBdl/ppgs_176', help='Full path to source ppgs')
  parser.add_argument('--save_dir', default='./', help='Save generate wavs')
  parser.add_argument('--lang', default='eng', help='eng or chinese')
  parser.add_argument('--targetSpk', default='ts', help='target speaker')
  parser.add_argument('--input', default='VCTKAlignLijie-training-lpc.txt', help='input path for speaker id')
  parser.add_argument('--max_process_test', default=10, type=int, help='Max prceossed test number')
  parser.add_argument('--hparams', default='',help='Hyperparameter overrides as a comma-separated list of name=value pairs')
  args = parser.parse_args()
  hparams.parse(args.hparams)

  if not os.path.exists(args.save_dir): os.makedirs(args.save_dir)

  # load model
  model_name1 = args.checkpoint.split('/')[-2] # logs-baoweinosil1
  model_name2 = args.checkpoint.split('/')[-1]  # model.ckpt.12892
  model_name = model_name1+'_'+model_name2
  synthesizer = Synthesizer()
  synthesizer.speaker2id(args.input)
  synthesizer.load(args.checkpoint)
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

  # load ppgs for synthesis
  for ii, ppgs_path in enumerate(glob.glob(args.ppgs_root + '/*.npy')):
    print ('%d: %s' %(ii, ppgs_path))
    if ii >= args.max_process_test: break
    ppgs = np.load(ppgs_path)
    #################################
    # ### testing, only for chinese
    # if args.lang == 'chi':
    #   source_mean_lf0 = np.load('speakerInfo/%s.npz' % 'baowei')
    #   target_mean_lf0 = np.load('speakerInfo/%s.npz' % 'ts')
    #   assert np.shape(ppgs)[-1] == 203
    #   source_lf0 = ppgs[:, -1]
    #   target_lf0 = ((source_lf0-source_mean_lf0['mean'])/source_mean_lf0['var'])*target_mean_lf0['var']+target_mean_lf0['mean']
    #   ppgs[:, -1] = target_lf0
    #################################
    #################################
    # ### testing, only for chinese
    # if args.lang == 'chi':
    #   target_mean_lf0 = np.load('speakerInfo/%s.npz' % args.targetSpk)
    #   assert np.shape(ppgs)[-1] == 203
    #   source_lf0 = ppgs[:, -1]
    #   source_vuv = ppgs[:, -2]
    #   source_lf0_nonzero = [item for item in source_lf0 if item != 0]
    #   source_mean_lf0 = np.mean(source_lf0_nonzero)
    #   source_var_lf0 = np.std(source_lf0_nonzero)
    #   target_lf0 = ((source_lf0-source_mean_lf0)/source_var_lf0)*target_mean_lf0['var']+target_mean_lf0['mean']
    #   target_lf0 = target_lf0 * source_vuv
    #   ppgs[:, -1] = target_lf0
    #################################
    #################################
    # ### testing, for chinese and english
    if hparams.speaker_id != '0': args.targetSpk = hparams.speaker_id
    target_mean_lf0 = np.load('speakerInfo/%s.npz' % args.targetSpk)
    assert np.shape(ppgs)[-1] in [203, 178, 220, 194, 378, 930, 355]
    source_lf0 = ppgs[:, -1]
    source_vuv = ppgs[:, -2]
    source_lf0_nonzero = [item for item in source_lf0 if item != 0]
    source_mean_lf0 = np.mean(source_lf0_nonzero)
    source_var_lf0 = np.std(source_lf0_nonzero)
    target_lf0 = ((source_lf0-source_mean_lf0)/source_var_lf0)*target_mean_lf0['var']+target_mean_lf0['mean']
    target_lf0 = target_lf0 * source_vuv
    ppgs[:, -1] = target_lf0
    #################################
    spec_pred = synthesizer.synthesize(ppgs)
    ppgs_name = os.path.basename(ppgs_path)
    ppgs_name = ppgs_name.split('.')[0]
    save_path = '%s/%s-%s-%s.npy' %(args.save_dir, ppgs_name, model_name, hparams.speaker_id)
    np.save(save_path, spec_pred)
    #audio.save_wav(wav, save_path)
