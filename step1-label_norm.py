# -*- coding: utf-8 -*-
import numpy as np
import glob
import os
import tqdm


# change wav into promise sampling rate
def norm_wav(wav_dir, sample_rate):
    new_wav_dir = wav_dir + '-' + str(sample_rate)
    if os.path.exists(new_wav_dir):
        print ('%s is exists, you can utilzied it right now!' %(new_wav_dir))
        return
    else:
        print ('Start changing %s !' %(new_wav_dir))
        os.makedirs(new_wav_dir)

    for wav_path in tqdm.tqdm(glob.glob(wav_dir + '/*')):
        wav_basename = os.path.basename(wav_path)
        wav_name, wav_type = wav_basename.split('.')
        if wav_type != 'wav':
            print ('%s is not the right wav' %(wav_path))
            continue

        # change all wav into corresponded sample rate
        new_wav_path = os.path.join(new_wav_dir, wav_basename)
        cmd = 'sox %s -r %d -c 1 %s' %(wav_path, sample_rate, new_wav_path)
        os.system(cmd)


if __name__ == "__main__":
    import fire
    fire.Fire()

