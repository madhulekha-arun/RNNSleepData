#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from collections import Counter
import numpy as np
from scipy.signal import detrend


seg_mask_explanation = [
     'normal',
     'around sleep stage change point',
     'marked by multiple sleep stages in the s3=ghrsame epoch',
     'NaN in sleep stage',
     'NaN in EEG',
     'overly high/low amplitude',
     'more than 5s is 0',
     'single-frequency oscilation']


def segment_EEG(EEG, sleep_stage, epoch_duration, Fs, start_end_remove_epoch_num=0, changepoint_mark_epoch_num=0, amplitude_thres=1000, to_remove_mean=False):#, different_stage_in_epoch='discard'
    """Load data from SleepEEGs_200 folder.

    Arguments:
    EEG -- np.ndarray, size=(sample_num, channel_num)
    sleep_stage -- np.ndarray, size=(sample_num,)
    epoch_size -- number of sample points per epoch (=second x Fs)

    Keyword arguments:
    start_end_remove_epoch_num -- default 0, number of epochs removed at the beginning and the end of the EEG signal
    changepoint_mark_epoch_num -- default 0, number of epochs to be marked around each sleep stage change point
    amplitude_thres -- default 1000, mark all segments with np.any(EEG_seg>=amplitude_thres)=True
    ##different_stage_in_epoch -- default 'discard', method to deal with epochs containing different stages. It can be 'discard' or 'dominate'.
    to_remove_mean -- default False, whether to remove the mean of EEG signal from each channel

    Outputs:
    EEG segments -- a list of np.ndarray, each has size=(epoch_size, channel_num)
    sleep stage segments --  a list of sleep stages corresponding to each epoch
    segment start positions -- a list of starting positions corresponding to each epoch
    """
    #start_end_remove_length = start_end_remove_epoch_num*epoch_size
    #if EEG_length<=start_end_remove_length*2:
    #    raise Exception('Overly short EEG signal with %d sample points (should be at least %d).'%(EEG_length,start_end_remove_length*2+1))

    #different_stage_in_epoch = different_stage_in_epoch.lower()
    #if different_stage_in_epoch!='discard' and different_stage_in_epoch!='dominate':
    #    raise Exception('Undefined keyword argument different_stage_in_epoch="%s".'%different_stage_in_epoch)
    #to_discard = different_stage_in_epoch=='discard'

    #EEG = EEG[start_end_remove_length:-start_end_remove_length,:]
    #sleep_stage = sleep_stage[start_end_remove_length:-start_end_remove_length]
    if to_remove_mean:
        EEG = EEG - np.mean(EEG,axis=0)
    EEG_length, channel_num = EEG.shape
    epoch_size = int(round(epoch_duration*Fs))
    seg_num = EEG_length//epoch_size
    print('number of segements',seg_num)

    # segment EEG and deal with epochs with different sleep stages and containing NaNs
    EEG_segs = []
    sleep_stages = []
    seg_time = []
    seg_masks = []
    #ccc = []
    for i in range(seg_num):
        ssp = i*epoch_size
        ss = sleep_stage[ssp:ssp+epoch_size]
        eeg_seg = EEG[ssp:ssp+epoch_size,:]
        seg_mask = seg_mask_explanation[0]  # normal

        # mark epochs with nan in sleep stage
        if np.any(np.isnan(ss)):
            seg_mask = seg_mask_explanation[3]
            most_common_ss = np.nan
        else:
            ct = Counter(ss)
            ck = ct.keys()
            most_common_ss = ck[np.argmax([ct[j] for j in ck])]

            # multiple sleep stages in one epoch
            if len(ck)!=1 and np.max(ct.values())<epoch_size-10:
                seg_mask = seg_mask_explanation[2]

            # mark epochs with overly high/low amplitude
            elif np.any(np.abs(eeg_seg)>amplitude_thres):
                seg_mask = seg_mask_explanation[5]

            # mark epochs with "0"s longer than 5s
            elif np.any(np.sum(np.abs(eeg_seg)<1,axis=0)>5*Fs):
                seg_mask = seg_mask_explanation[6]

            # mark epochs with nan in EEG
            elif np.any(np.isnan(eeg_seg)):
                seg_mask = seg_mask_explanation[4]

            # single-frequency oscillation
            # maximum power > 10x median power
            else:
                psd = np.abs(np.fft.fft(detrend(eeg_seg,axis=0),axis=0))
                #ccc.extend(np.max(psd,axis=0)*1./np.median(psd,axis=0))
                if np.any(np.max(psd,axis=0)>3000*np.median(psd,axis=0)):########## 3000 times?
                    seg_mask = seg_mask_explanation[7]

        sleep_stages.append(most_common_ss)
        EEG_segs.append(eeg_seg)
        seg_time.append(i*epoch_duration)
        seg_masks.append(seg_mask)

    seg_num = len(EEG_segs)

    # mark epochs around the sleep stage change point
    if changepoint_mark_epoch_num>0:
        tomark_epoch_ids = []
        for i in range(seg_num-1):
            if not np.isnan(sleep_stages[i]) and not np.isnan(sleep_stages[i+1]) and sleep_stages[i]!=sleep_stages[i+1]:
                tomark_epoch_ids.extend(range(max(0,i+1-changepoint_mark_epoch_num),i+1))
                tomark_epoch_ids.extend(range(i+1,min(seg_num,i+1+changepoint_mark_epoch_num)))
        tomark_epoch_ids = set(tomark_epoch_ids)
        for i in tomark_epoch_ids:
            seg_masks[i] =  seg_mask_explanation[1]

    if start_end_remove_epoch_num>0:
        EEG_segs = EEG_segs[start_end_remove_epoch_num:-start_end_remove_epoch_num]
        sleep_stages = sleep_stages[start_end_remove_epoch_num:-start_end_remove_epoch_num]
        seg_time = seg_time[start_end_remove_epoch_num:-start_end_remove_epoch_num]
        seg_masks = seg_masks[start_end_remove_epoch_num:-start_end_remove_epoch_num]

    return EEG_segs, np.array(sleep_stages), np.array(seg_time), seg_masks


"""
if __name__=='__main__':
    import pdb
    from load_dataset import *

    EEG_channels = ['F3M2','F4M1','C3M2','C4M1','O1M2','O2M1']
    EEG, sleep_stage, params = check_load_Yvonne_dataset(r'D:\dropbox\Dropbox\SleepScoringProject\YvonneDataset_MATFILES\YvonneDataSet_Exported_1.mat',
            r'D:\dropbox\Dropbox\SleepScoringProject\YvonneAndApneaDataSetLabels\YvonneDataSet_Labels_1.mat', channels=EEG_channels)

    segs, sleep_stages, seg_times = segment_EEG(EEG, sleep_stage, 30*200,
            start_end_remove_epoch_num=2, changepoint_mark_epoch_num=1, amplitude_thres=500, to_remove_mean=False)#, different_stage_in_epoch='discard'
    print('\nObtained %d segments.'%len(segs))
"""

