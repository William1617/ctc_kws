from tkinter.messagebox import NO
import torch
import torchaudio
from torch import nn, autograd, utils
import torchaudio.transforms as T
import numpy as np
import warnings
import torchaudio.compliance.kaldi as kaldi
warnings.simplefilter("ignore", UserWarning)
index_map = {}
class TextTransform:
    """Maps characters to integers and vice versa"""
    def __init__(self):
       with open('./data/conf/wordid.txt','r') as char_map_str:
            char_map_str = char_map_str
            self.char_map = {}
            self.index_map = {}
            for line in char_map_str:
                ch, index = line.split()
                self.char_map[ch] = int(index)
                self.index_map[int(index)] = ch
       with open('./data/conf/wordmap.txt','r') as word_map_str:
            word_map_str=word_map_str
            self.word_map={}
            for line in word_map_str:
                word,phones=line.split()
                self.word_map[word]=str(phones)
    def text_to_int(self, text):
        """ Create target label sequence """
        int_sequence = []
        phones = self.word_map.get(text)
        if(phones is None):print(text)
        # Split target word label
        chindex=phones.split('-')
       
        for c in chindex:
            ch = self.char_map.get(c)
            if(ch is None):print(c)
            int_sequence.append(int(ch))
      
        return int_sequence

text_transform=TextTransform()

# Out length after subsamplng4
def cal_length(audio_length):
    k=audio_length%4
    if(k<3):
        out_length=int(audio_length/4)-1
    else:
        out_length=int(audio_length/4)
    return out_length

def data_processing(data):
    mfccs = []
    labels = []
    output_lengths = []
    label_lengths = []

    for (waveform, sample_rate, utterance) in data:
        mfcc=kaldi.fbank(waveform,num_mel_bins=80,
                        dither=0.0,energy_floor=0.0,
                        sample_frequency=sample_rate)   
       
        mfccs.append(mfcc)
    #Get target sequence
        label = torch.Tensor(text_transform.text_to_int(utterance.lower()))
        labels.append(label)
    # Audio length after subsampling4
        output_lengths.append(cal_length(mfcc.shape[0]))
        label_lengths.append(len(label))

    mfccs = nn.utils.rnn.pad_sequence(mfccs, batch_first=True).unsqueeze(1).transpose(2, 3)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)
    output_lengths=torch.IntTensor(output_lengths)
    return mfccs, labels, output_lengths, label_lengths

def calcmvn(feats,cmvn_mean,cmvn_var):
    m,n=feats.size()
    feats_norm=feats
    for idx in range(m):
        for j in range(n):
            feats_norm[idx][j]=(feats[idx][j]-cmvn_mean[j])/cmvn_var[j]
    return feats_norm

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

