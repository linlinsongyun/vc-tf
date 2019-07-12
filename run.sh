#############################################################################
######################## feature extraction #################################
#############################################################################
#### Step1: Extaract ppgs
# put ppgs into TS/ppgs


#### Step2: Extaract lpcnet params
# put lpc params into TS/lpc


#### Step3: Extract f0 using 'world'
# change wav to 16k: python -u step1-label_norm.py norm_wav /home/lianzheng03/backup/VCTK-Corpus/wavtrimSilAlignLijie-22050 16000
# cd world_feat_187dim
# chmod -R 777 bin/
# python -u mgcExtraction.py world_feature_extraction '/home/lianzheng03/backup/VCTK-Corpus/wavtrimSilAlignLijie-22050-16000' '../TS/world'


#### Step4: Merge ppgs and lf0 and uvu
# python step2-ppgsAddf0.py merge_ppgs_lf0_uvu TS/ppgs TS/world/f0 TS/ppgsf0vuv


#### Step5: Cal target speakers mean/var lf0, saved in speakerInfo/ts.npz
# python -u step2-ppgsAddf0.py calcu_lf0_mean_var 'TS/ppgsf0vuv/*.npy' speakerInfo/ts.npz




#############################################################################
######################## training #################################
#############################################################################
#### gain training.txt
# python step3-trainingTxt.py get_trainingTxt 'TS/ppgsf0vuv' 'TS-training-ppgsf0vuv-lpc.txt' 'TS/lpc' None None

#### training
## num_ppgs: input-dim, for example, ppgs has 201dim, then add lf0 and vuv, we get 203dim input
## num_freq: output-dim, for example, 24k lpcnet need 32 dim params, then num_freq=32
# python train.py --name='VC_TS_ppgsf0vuv_lpc_lr0.001' --input='TS-training-ppgsf0vuv-lpc.txt' --hparams=initial_learning_rate=0.001,num_ppgs=203,num_freq=32,pred_step=1




#############################################################################
######################## synthesis for test.wav ####
#############################################################################
### step1: extract ppgs from test.wav
### step2: gain f0 from test.wav 
# python -u mgcExtraction.py world_feature_extraction '/home/lianzheng03/backup/testWav' '../testWav/world'
### step3: Merge ppgs and lf0 and uvu
# python step2-ppgsAddf0.py merge_ppgs_lf0_uvu testWav/ppgs testWav/world/f0 testWav/ppgsf0vuv


### step2: generate predicted ppgs
# python syn_new.py --checkpoint='logs-VC_TS_ppgsf0vuv_lpc_lr0.001/model.ckpt-550000' --lang='chi' --ppgs_root='testWav_ppgs_chi' --save_dir='testWav_feat20_pred_chi' --hparams=num_ppgs=203,num_freq=32,pred_step=1,addition_vector='None',speaker_id='ts'

#### step3: gain audio
# python featureConvert.py feat_to_wav_folder ../testWav_feat20_pred_chi ../testWav_lpc_vc_chi

