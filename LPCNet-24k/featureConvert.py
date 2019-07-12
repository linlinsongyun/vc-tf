import numpy as np
import glob
import tqdm
import os
import librosa

sample_rate=24000
##############################################################################
#### read feat from wav
##############################################################################
def read_feat_one(feat_path, save_path):

    # read origin features
    features = np.fromfile(feat_path, dtype='float32')
    features = features.reshape((-1, 49))

    # extract 32dim feat
    feat = features[:,:32]

    # save feat
    np.save(save_path, feat)


def read_feat_all(wav_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)

    for wav_path in tqdm.tqdm(glob.glob(wav_root+'/*')):

        ## pad wav
        # wav = librosa.core.load(wav_path, sr=sample_rate)[0]
        # wav = np.pad(wav, (sample_rate*1,0), mode='constant') # add 1s in the begin
        # wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        # librosa.output.write_wav('test_wav.wav', wav.astype(np.int16), sample_rate)

        ## extract feat
        # cmd = "sox test_wav.wav -r %d -c 1 -t sw -> input.s16" % (sample_rate)
        cmd = "sox %s -r %d -c 1 -t sw -> input.s16" % (wav_path, sample_rate)
        os.system(cmd)
        cmd = "./dump_data -test input.s16 features.f32"
        os.system(cmd)

        ## change feat into 20dim
        wav_name = os.path.basename(wav_path)
        wav_name = wav_name.split('.')[0]
        save_path = os.path.join(save_root, wav_name+'.npy')
        read_feat_one(feat_path='features.f32',save_path=save_path)



##############################################################################
#### convert feat to wav
##############################################################################
def feat_to_wav(feat_path, save_wav_path):
    feat = np.load(feat_path) # [len, 20]
    feat_55dim = np.zeros((len(feat), 49))
    feat_55dim[:,:32] = feat
    feat_55dim = feat_55dim.astype('float32')

    features = feat_55dim.reshape((-1))
    features.tofile('test_features_new.f32')

    cmd = "./test_lpcnet test_features_new.f32 test_input_new.s16"
    os.system(cmd)
    cmd = "ffmpeg -f s16le -ar 24k -ac 1 -i test_input_new.s16 %s" %(save_wav_path)
    os.system(cmd)

def feat_to_wav_folder(feat_root, save_root):
    if not os.path.exists(save_root): os.makedirs(save_root)
    for feat_path in glob.glob(feat_root+'/*.npy'):
        feat_name = os.path.basename(feat_path).rsplit('.', 1)[0]
        save_path = os.path.join(save_root, feat_name+'.wav')
        feat_to_wav(feat_path, save_path)


##############################################################################
#### convert wav to feat to wav
##############################################################################
def wav_to_feat_to_wav(wav_path):

    ## wav to feat
    cmd = "sox %s -r %s -c 1 -t sw -> test_input.s16" %(wav_path, sample_rate)
    os.system(cmd)
    cmd = "./dump_data -test test_input.s16 test_features.f32"
    os.system(cmd)

    # feat to 20-dim feat
    read_feat_one(feat_path='test_features.f32',save_path='test_features32.npy')

    ## 20-dim feat to wav
    feat_to_wav('test_features32.npy', 'test_wavGenerate.wav')



# Extract features: python featureConvert.py read_feat_all wav_root save_root
# python featureConvert.py feat_to_wav feat_path save_wav_path
# python featureConvert.py wav_to_feat_to_wav /home/lianzheng03/backup/ts/wav-44100/ts00001.wav
if __name__ == '__main__':
    import fire
    fire.Fire()

