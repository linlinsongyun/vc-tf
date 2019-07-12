import os
import copy
import tqdm
import glob
import shutil
import numpy as np

def read_binfile(filename, dim=60, dtype=np.float64):
    '''
    Reads binary file into numpy array.
    '''
    fid = open(filename, 'rb')
    v_data = np.fromfile(fid, dtype=dtype)
    fid.close()
    if np.mod(v_data.size, dim) != 0:
        raise ValueError('Dimension provided not compatible with file size.')
    m_data = v_data.reshape((-1, dim)).astype('float64') # This is to keep compatibility with numpy default dtype.
    m_data = np.squeeze(m_data)
    return  m_data


def merge_ppgs_lf0_uvu(ppgs_root, f0_root, save_root):

    if not os.path.exists(save_root): os.makedirs(save_root)
    for ppgs_path in tqdm.tqdm(glob.glob(ppgs_root+'/*.npy')):
        wav_name = os.path.basename(ppgs_path).split('.')[0]
        f0_path = os.path.join(f0_root, wav_name+'.f0')
        ppgs_feat = np.load(ppgs_path) ## read ppgs
        if not os.path.exists(f0_path): continue
        f0_feat = read_binfile(f0_path, dim=1, dtype=np.float64) # read f0
        if np.max(np.abs(f0_feat)) < 1e-5: continue
        vuv_feat = [] # read vuv
        for item in f0_feat:
            item = 0 if item==0 else 1
            vuv_feat.append(item)
        lf0_feat = np.log10(np.maximum(1e-10, f0_feat)) # read lf0
        lf0_feat = lf0_feat * vuv_feat

        if abs(len(ppgs_feat)-len(f0_feat))>3:
            print (abs(len(ppgs_feat)-len(f0_feat)))
            continue
        min_len = min(len(ppgs_feat), len(f0_feat))
        ppgs_feat = ppgs_feat[:min_len, :]
        lf0_feat = np.array(lf0_feat[:min_len])[:, np.newaxis]
        vuv_feat = np.array(vuv_feat[:min_len])[:, np.newaxis]

        # merge ppgs, vuv and lf0
        feat_new = np.concatenate((ppgs_feat, vuv_feat, lf0_feat), axis=1)

        # save
        save_path = os.path.join(save_root, wav_name+'.npy')
        np.save(save_path, feat_new)


def calcu_lf0_mean_var(ppgslf0vuv_root, save_path='speakerInfo/baowei.npz'):
    mean_lf0 = []
    var_lf0 = []
    for feat_path in tqdm.tqdm(glob.glob(ppgslf0vuv_root)):
        lf0_feat = np.load(feat_path)[:, -1]
        lf0_feat = [item for item in lf0_feat if item != 0] # only non zero process
        if len(lf0_feat) < 10:
            print (feat_path)
            continue
        mean_lf0.append(np.mean(lf0_feat))
        var_lf0.append(np.std(lf0_feat))
    print (np.mean(mean_lf0), np.mean(var_lf0))
    np.savez_compressed(save_path,
                        mean=np.mean(mean_lf0),
                        var=np.mean(var_lf0))



# python step2-ppgsAddf0.py merge_ppgs_lf0_uvu(ppgs_root, world_187_root, save_root)
# python -u step2-ppgsAddf0.py calcu_lf0_mean_var 'TS/ppgsf0vuv/*.npy' speakerInfo/ts.npz
if __name__ == "__main__":
    import fire
    fire.Fire()
