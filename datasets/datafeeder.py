import os
import random
import threading
import time
import traceback
import glob
import numpy as np
import tensorflow as tf
import librosa
from util.infolog import log

_batches_per_group = 32
_pad = 0
_stop_token_pad = 1


class DataFeeder(threading.Thread):
    '''Feeds batches of data into a queue on a background thread.'''

    def __init__(self, coordinator, metadata_filename, hparams):
        super(DataFeeder, self).__init__()
        self._coord = coordinator
        self._hparams = hparams
        self._offset = 0
        self.speaker2vec = {}

        # Load metadata:
        speakers = []
        self._datadir = os.path.dirname(metadata_filename)
        with open(metadata_filename, encoding='utf-8') as f:
            self._metadata = [line.strip().split('|') for line in f]
            hours = sum((int(x[2]) for x in self._metadata)) * hparams.frame_shift_ms / (3600 * 1000)
            if hparams.addition_vector=='None':
                speakers = set([os.path.basename(x[-1]).split('.')[-1] for x in self._metadata]) # only save 'wav' or 'npy'
            else:
                speakers = set([os.path.basename(x[-1]).split('_')[0] for x in self._metadata]) # for 108 speakers
            speakers = [item for item in speakers] # change to list
            speakers.sort() # sort
            log('Loaded metadata for %d examples (%.2f hours)' % (len(self._metadata), hours))
            log('Speaker num %d' % len(speakers))


        speaker2vec = {}
        for ii, speaker in enumerate(speakers):
            temp = np.zeros((len(speakers), ))
            temp[ii] = 1.0
            speaker2vec[speaker] = temp
        self.speaker2vec=speaker2vec


        # Create placeholders for inputs and targets. Don't specify batch size because we want to
        # be able to feed different sized batches at eval time.
        self.hop_length = hparams.sample_rate * hparams.frame_shift_ms // 1000
        self.n_timesteps = (hparams.duration * hparams.sample_rate) // self.hop_length + 1
        self._placeholders = [
            tf.placeholder(tf.int32, [None], 'input_lengths'),
            tf.placeholder(tf.float32, [None, self.n_timesteps, hparams.num_mels], 'mel_targets'),
            tf.placeholder(tf.float32, [None, self.n_timesteps, hparams.num_freq], 'linear_targets'),
            tf.placeholder(tf.float32, [None, self.n_timesteps, hparams.num_ppgs], 'ppgs'),
            tf.placeholder(tf.float32, [None, len(speakers)], 'speakers'),
        ]

        # Create queue for buffering data:
        queue = tf.FIFOQueue(8, [tf.int32, tf.float32, tf.float32, tf.float32, tf.float32], name='input_queue')
        self._enqueue_op = queue.enqueue(self._placeholders)
        self.input_lengths, self.mel_targets, self.linear_targets, self.ppgs, self.speakers = queue.dequeue()
        self.input_lengths.set_shape(self._placeholders[0].shape)
        self.mel_targets.set_shape(self._placeholders[1].shape)
        self.linear_targets.set_shape(self._placeholders[2].shape)
        self.ppgs.set_shape(self._placeholders[3].shape)
        self.speakers.set_shape(self._placeholders[4].shape)


    def start_in_session(self, session):
        self._session = session
        self.start()

    def run(self):
        try:
            while not self._coord.should_stop():
                self._enqueue_next_group()
        except Exception as e:
            traceback.print_exc()
            self._coord.request_stop(e)

    def _enqueue_next_group(self):
        start = time.time()

        # Read a group of examples:
        n = self._hparams.batch_size
        r = 1

        # 32*32
        examples = [self._get_next_example() for i in range(n * _batches_per_group)]

        # Bucket examples based on similar output sequence length for efficiency:
        examples.sort(key=lambda x: x[-1])
        batches = [examples[i:i + n] for i in range(0, len(examples), n)]
        random.shuffle(batches)

        log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
        for batch in batches:
            feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
            self._session.run(self._enqueue_op, feed_dict=feed_dict)

    def _get_next_example(self):
        '''Loads a single example (input, mel_target, linear_target, stop_token_target) from disk'''
        while 1:
            if self._offset >= len(self._metadata):
                self._offset = 0
                random.shuffle(self._metadata)
            meta = self._metadata[self._offset]
            self._offset += 1

            # read mel_target
            mel_target = np.zeros((1, self._hparams.num_mels))
            if self._hparams.pred_step != 1: ## '1' -> mel is not needed
                mel_path = os.path.join(self._datadir, meta[1])
                if not os.path.exists(mel_path): continue
                mel_target = np.load(mel_path)

            # read linear_target
            linear_path = os.path.join(self._datadir, meta[0])
            if not os.path.exists(linear_path): continue
            linear_target = np.load(linear_path)

            # read ppgs
            ppgs_path = os.path.join(self._datadir, meta[3])
            if not os.path.exists(ppgs_path): continue
            ppgs = np.load(ppgs_path)
            if abs(len(ppgs) - len(linear_target)) <= 3: break
        
        # make ppgs and linear_target have the same len
        min_len = min(len(ppgs), len(linear_target))
        ppgs = ppgs[:min_len]
        linear_target = linear_target[:min_len]
        assert self._hparams.random_crop in [True, False]
        if self._hparams.random_crop==True:
            start = np.random.choice(range(np.maximum(1, len(linear_target) - self.n_timesteps)), 1)[0]
            end = start + self.n_timesteps
            ppgs = ppgs[start:end]
            linear_target = linear_target[start:end]

        ppgs = librosa.util.fix_length(ppgs, self.n_timesteps, axis=0)
        mel_target = librosa.util.fix_length(mel_target, self.n_timesteps, axis=0)
        linear_target = librosa.util.fix_length(linear_target, self.n_timesteps, axis=0)
        if self._hparams.addition_vector=='None':
            speaker_id = os.path.basename(meta[3]).split('.')[-1]
        else:
            speaker_id = os.path.basename(meta[3]).split('_')[0]
        speaker_emb = self.speaker2vec[speaker_id]
        assert len(ppgs) == len(linear_target)
        return (linear_target, mel_target, ppgs, speaker_emb, len(linear_target))


def _prepare_batch(batch, outputs_per_step):
    random.shuffle(batch)
    linear_targets = np.stack([x[0] for x in batch]) # [N, timestep, feat]
    mel_targets = np.stack([x[1] for x in batch])
    ppgs = np.stack([x[2] for x in batch])
    speakers = np.stack([x[3] for x in batch]).astype('float32')
    input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
    return (input_lengths, mel_targets, linear_targets, ppgs, speakers)


def _prepare_inputs(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _prepare_stop_token_targets(targets, alignment):
    max_len = max((len(t) for t in targets)) + 1
    return np.stack([_pad_stop_token_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)


def _pad_stop_token_target(t, length):
    return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=_stop_token_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder
