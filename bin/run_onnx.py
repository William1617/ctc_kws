import onnxruntime
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
        if(wordidx!=preindex and wordidx>1):
            decodeword=textlist[wordidx]
            decode_cout[decodeword] +=1
            result.append(textlist[wordidx])
            frame_num.append(i*4)
        preindex=wordidx
    print(result)
    print(frame_num)

index_word={}
decode_cout={}
with open('./data//conf//wordid.txt') as char_map:
    for line in char_map:
        word,index=line.split()
        index_word[int(index)]=word
        decode_cout[word]=0


wavdir="./data/testwav/left/"
vocab_size=15

file_paths=[]
model_path="./conformemodel.onnx"
interprter=onnxruntime.InferenceSession(model_path)

for root, directories, files in os.walk(wavdir):
    for filename in files:
        filepath = os.path.join(root, filename)
        file_paths.append(filepath)

for wave in file_paths:
    audio_data,samplerate=torchaudio.load(wave)
   
    mfdata=kaldi.fbank(audio_data,num_mel_bins=80,frame_length=25,frame_shift=10,dither=0.0,energy_floor=0.0,sample_frequency=samplerate)
    m,n=mfdata.size()
    inputdata=np.reshape(mfdata,(1,m,n))
    outputdata=interprter.run(None,{'input_feature':inputdata.numpy()})
    ctc_log=np.squeeze(outputdata)
    ctc_log=np.array(ctc_log)
    max_index=np.argmax(ctc_log,axis=1)
    gready_search(max_index,index_word,decode_cout)
print(decode_cout)

