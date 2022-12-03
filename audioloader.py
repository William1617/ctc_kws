from pydoc import TextRepr
import numpy as np
import os
from torch.utils import data
import torchaudio.compliance.kaldi as kaldi
import torchaudio
import torch

class audiodataset(data.Dataset):
    def __init__(self,data_dir):
        self.wave_info=self.getaudioinfo(data_dir)

    def __getitem__(self, index) :
        wav,sample_rate,label=self.wave_info[index]
        return wav,sample_rate,label

    def __len__(self):
        return len(self.wave_info)


    def getaudioinfo(self,data_dir):
        file_paths=[]
        for root, directories, files in os.walk(data_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                file_paths.append(filepath)  # Add it to the list.
        wavlist=file_paths
        wav_info=[]
        for path in wavlist:
           
            label0=path.split('//')[-1]
        # Get class name
            label1=label0.split("\\")[0]
            waveform, sample_rate = torchaudio.load(path)
           # wav_len=waveform.shape[0]
            #label_len=len(label)
            wav_info.append((waveform,sample_rate,label1)) 

        return wav_info
if __name__=='__main__':
    test_dir = './/data//testwav//'
    test_set=audiodataset(test_dir)
