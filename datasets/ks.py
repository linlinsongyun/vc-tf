from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import glob
import librosa
from util import audio
from hparams import hparams


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the LJ Speech dataset from a given input path into a given output directory.

    Args:
      in_dir: The directory where you have downloaded the LJ Speech dataset
      out_dir: The directory to write the output into
      num_workers: Optional number of worker processes to parallelize across
      tqdm: You can optionally pass tqdm to get a nice progress bar

    Returns:
      A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    ## process on wav directly
    for wav_path in glob.glob(in_dir+'/*.wav'):
        futures.append(
            executor.submit(
                partial(_process_utterance, out_dir, index, wav_path)))
        index += 1
    return [future.result() for future in tqdm(futures)]



def _process_utterance(out_dir, index, wav_path):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
        out_dir: The directory to write the spectrograms into
        index: The numeric index to use in the spectrogram filenames.
        wav_path: Path to the audio file containing the speech input
        text: The text spoken in the input audio file

    Returns:
        A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # cut or pad wav into 2s
    length = hparams.sample_rate * hparams.duration
    wav = librosa.util.fix_length(wav, length)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Computer mfcc
    # mfcc = audio.mfcc(wav).astype(np.float32)

    # Write the spectrograms to disk:
    wav_name = os.path.basename(wav_path)
    wav_name = wav_name.split('.')[0]
    spectrogram_filename = 'spec-%s.npy' % wav_name
    mel_filename = 'mel-%s.npy' % wav_name
    mfcc_filename = 'mfcc-%s.npy' % wav_name
    np.save(
        os.path.join(out_dir, spectrogram_filename),
        spectrogram.T,
        allow_pickle=False)
    np.save(
        os.path.join(out_dir, mel_filename),
        mel_spectrogram.T,
        allow_pickle=False)
    # np.save(
    #     os.path.join(out_dir, mfcc_filename),
    #     mfcc.T,
    #     allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames)
