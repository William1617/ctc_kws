
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import os

import numpy as np

def gready_search(index,textlist,decode_cout):
    preindex=-1
    result=[]
    frame_num=[]
    for i in range(len(index)):
        wordidx=int(index[i])
        if(wordidx!=preindex and wordidx>2):
            decodeword=textlist[wordidx]
            decode_cout[decodeword] +=1
            
            result.append(decodeword)
            frame_num.append(i*4)
        preindex=wordidx
    print(result)
    print(frame_num)
  

model=torch.jit.load('./conformermodel.zip')
# store word id
word_index={}
# Count decode result
decode_count={}
with open('.//data//conf//wordid.txt') as char_map:
    for line in char_map:
        word,index=line.split()
        word_index[int(index)]=word
        decode_count[word]=0

wavdir='.//data/testwav//left//'

file_paths=[]

for root, directories, files in os.walk(wavdir):
    for filename in files:
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)


for wave in file_paths:
    audio_data,samplerate=torchaudio.load(wave)
    print(wave)
   
    mfdata=kaldi.fbank(audio_data,num_mel_bins=80,frame_length=25,frame_shift=10,dither=0.0,energy_floor=0.0,sample_frequency=samplerate)
    m,n=mfdata.size()
    
    inputdata=mfdata.reshape(1,m,n)
    outputdata=model(inputdata)[0].squeeze()
    # Get max index
    max_index=outputdata.max(1,keepdim=False)[1]
    prob=outputdata.max(1)[0]
    gready_search(max_index,word_index,decode_count)
        
print(decode_count)
