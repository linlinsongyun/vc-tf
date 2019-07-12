import os
import copy
import tqdm
import glob
import shutil
import numpy as np


def get_trainingTxt(ppgs_root, save_path, lpc_root=None, world_root=None, mel_root=None):
    ## lpc_root and spec_root should has atleast one exists
    assert lpc_root!=None or world_root!=None

    lines_new = []
    for ppgs_path in tqdm.tqdm(glob.glob(ppgs_root+'/*.npy')):
        ppgs_name = os.path.basename(ppgs_path)
        ppgs_name = ppgs_name.split('.')[0]
        mel_path = 'None'
        if mel_root!=None:mel_path = os.path.join(mel_root, '%s.mel.npy'%(ppgs_name))
        if world_root!=None: spec_path = os.path.join(world_root, '%s.62.npy'%(ppgs_name))
        if lpc_root!=None: spec_path = os.path.join(lpc_root, '%s.npy'%(ppgs_name))
        if not os.path.exists(spec_path): continue
        frames = 501 # len(np.load(ppgs_path))
        line = '%s|%s|%d|%s' %(spec_path, mel_path, frames, ppgs_path)
        lines_new.append(line)

    ## save new lines
    output = open(save_path, 'w', encoding='utf8')
    for ii, line in enumerate(lines_new):
        output.write('%s\n' %(line))
    output.close()



def get_trainingTxt_Net1(ppgs_root, mfcc_root, save_path):
   
    lines_new = []
    for ppgs_path in tqdm.tqdm(glob.glob(ppgs_root+'/*.npy')):
        ppgs_name = os.path.basename(ppgs_path)
        ppgs_name = ppgs_name.split('.')[0]
        mfcc_path = os.path.join(mfcc_root, 'mfcc-%s.npy'%(ppgs_name))
        if not os.path.exists(mfcc_path): continue
        # len_distance = abs(len(np.load(ppgs_path)) - len(np.load(mfcc_path)))
        # if  len_distance > 3:
        #     print (len_distance)
        #     continue
        frames = 501 # len(np.load(ppgs_path))
        line = '%s|%s|%d|%s' %(mfcc_path, mfcc_path, frames, ppgs_path)
        lines_new.append(line)

    ## save new lines
    output = open(save_path, 'w', encoding='utf8')
    for ii, line in enumerate(lines_new):
        output.write('%s\n' %(line))
    output.close()


# python step3-trainingTxt.py get_trainingTxt 'LJspeech/ppgs' 'LJspeech-training-lpc.txt' 'LJspeech/lpc' None None
# python step3-trainingTxt.py get_trainingTxt 'LJspeech/ppgs' 'LJspeech-training-world.txt' None 'LJspeech/world/feat_62dim' None
if __name__ == "__main__":
    import fire
    fire.Fire()
