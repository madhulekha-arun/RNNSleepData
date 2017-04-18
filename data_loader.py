
# create segments of 30 sec for spect, label data 
import os.path
import re
import sys
from datetime import datetime, timedelta
import numpy as np
import scipy.io as sio
import h5py
import glob
import matplotlib.pyplot as plt
from spectrum import pow2db

def check_load_Yvonne_dataset(data_path, label_path, channels=None, report_and_actual_time_tol=300):
    """Check and load data from Yvonne dataset.

    Arguments:
    data_path -- string
    label_path -- string, default None

    Keyword arguments:
    channels -- default None, a list of channels names, if None, use all.
    report_and_actual_time_tol -- default 300, the maximum allowed difference between reported time and actual signal length in second

    Outputs:
    EEG signal -- np.ndarray, size=(sample_num, channel_num)
    sleep_stage -- np.ndarray, size=(sample_num,)
    a dict of relevant parameters
    """
    # load data
    if not os.path.isfile(data_path):
        raise Exception('%s is not found.'%data_path)
    if not os.path.isfile(label_path):
        raise Exception('%s is not found.'%label_path)

    try:
        ff = h5py.File(data_path)
        EEG = ff['s'][()]
        channel_names = ff['hdr']['label'][()].ravel()
        channel_names = [''.join(chr(ff[channel_names[i]][()].ravel()[j]) for j in range(ff[channel_names[i]][()].ravel().shape[0])).upper() for i in range(len(channel_names))]
        physicalMin = ff['hdr']['physicalMin'][()].ravel()
        physicalMax = ff['hdr']['physicalMax'][()].ravel()
        digitalMin = ff['hdr']['digitalMin'][()].ravel()
        digitalMax = ff['hdr']['digitalMax'][()].ravel()
        # check sample frequency
        if 'sampleFrequency' in ff['hdr']:
            Fs = ff['hdr']['sampleFrequency'][()][0,0]
        else:
            Fs = None
    except:
        ff = sio.loadmat(data_path)
        EEG = ff['s']
        channel_names = ff['hdr']['label'][0,0]
        channel_names = [channel_names[i,0][0] for i in range(channel_names.shape[0])]
        physicalMin = ff['hdr']['physicalMin'][0,0].ravel()
        physicalMax = ff['hdr']['physicalMax'][0,0].ravel()
        digitalMin = ff['hdr']['digitalMin'][0,0].ravel()
        digitalMax = ff['hdr']['digitalMax'][0,0].ravel()
        try:
            Fs = ff['hdr']['sampleFrequency'][0,0][0,0]
        except:
            Fs = None
    if Fs is None:
        #print('\nNo sampleFrequency in %s. Use default 200Hz.'%data_path)
        Fs = 200
    if EEG.shape[0]<EEG.shape[1]:  # make it sample_num x channel_num
        EEG = EEG.T
    data_path = os.path.basename(data_path)

    # load labels
    try:
        ffl = h5py.File(label_path)
        sleep_stage = ffl['stage'][()].ravel()
        time_str_elements = ffl['features']['StartTime'][()].ravel()
        start_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
        time_str_elements = ffl['features']['EndTime'][()].ravel()
        end_time = ''.join(chr(time_str_elements[j]) for j in range(time_str_elements.shape[0]))
    except:
        ffl = sio.loadmat(label_path)
        sleep_stage = ffl['stage'].ravel()
        start_time = ffl['features']['StartTime'][0,0][0,0]
        end_time = ffl['features']['EndTime'][0,0][0,0]

    start_time = start_time.split(':')
    second_elements = start_time[-1].split('.')
    start_time = datetime(1990,1,1,hour=int(float(start_time[0])), minute=int(float(start_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))
    end_time = end_time.split(':')
    second_elements = end_time[-1].split('.')
    end_time = datetime(1990,1,1,hour=int(float(end_time[0])), minute=int(float(end_time[1])),
        second=int(float(second_elements[0])), microsecond=int(float('0.'+second_elements[1])*1000000))

    # check signal length = sleep stage length
    if sleep_stage.shape[0]!=EEG.shape[0]:
        raise Exception('\nInconsistent sleep stage length (%d) and signal length (%d) in %s'%(sleep_stage.shape[0],EEG.shape[0],data_path))

    # check end_time - start_time = signal signal_duration
    reported_time_diff = (end_time-start_time).seconds
    signal_duration = EEG.shape[0]*1.0/Fs
    time_diff = abs(reported_time_diff-signal_duration)
    if time_diff>report_and_actual_time_tol:
        raise Exception('\nend_time-start_time= %ds, EEG signal= %ds, difference= %ds in %s'%(reported_time_diff,signal_duration,time_diff,data_path))

    # check channel number
    if not EEG.shape[1]==len(channel_names)==physicalMin.shape[0]==\
            physicalMax.shape[0]==digitalMin.shape[0]==digitalMax.shape[0]:
        raise Exception('\nInconsistent channel number in %s'%data_path)

    # only take EEG channels to study
    if channels is None:
        EEG_channel_ids = list(range(len(channel_names)))
    else:
        EEG_channel_ids = []
        for i in range(len(channels)):
            channel_name_pattern = re.compile(channels[i][:2].upper()+'-*'+channels[i][-2:].upper())
            found = False
            for j in range(len(channel_names)):
                if channel_name_pattern.match(channel_names[j].upper()):
                    EEG_channel_ids.append(j)
                    found = True
                    break
            if not found:
                raise Exception('Channel %s is not found.'%channels[i])
        EEG = EEG[:,EEG_channel_ids]
        physicalMin = physicalMin[EEG_channel_ids]
        physicalMax = physicalMax[EEG_channel_ids]
        digitalMin = digitalMin[EEG_channel_ids]
        digitalMax = digitalMax[EEG_channel_ids]

    # check whether the EEG signal contains NaN
    if np.any(np.isnan(EEG)):
        raise Exception('\nFound Nan in EEG signal in %s'%data_path)

    # check signal min/max
    s_min = np.nanmin(EEG,axis=0)
    s_max = np.nanmax(EEG,axis=0)
    if np.any(s_min<physicalMin) or np.any(s_max>physicalMax):
        raise Exception('\nSignal exceeds physical min/max in %s'%data_path)
    if np.any(s_min<digitalMin) or np.any(s_max>digitalMax):
        raise Exception('\nSignal exceeds digital min/max in %s'%data_path)

    # check whether sleep_stage contains NaN
    #if np.any(np.isnan(sleep_stage)):
    #    raise Exception('\nFound Nan in sleep stages in %s'%data_path)

    # check whether sleep_stage contains all 5 stages
    stages = np.unique(sleep_stage[np.logical_not(np.isnan(sleep_stage))]).astype(int).tolist()
    if len(stages)<=2:
        raise Exception('\n#sleep stage <= 2: %s in %s'%(stages,data_path))

    return EEG, sleep_stage, {'Fs':Fs, 'EEG_channel_ids':EEG_channel_ids}