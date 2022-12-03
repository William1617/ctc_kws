
from conformer import conformermodel
import torch
import numpy as np
#cmvn_mean=torch.Tensor(np.loadtxt('cmvn_mean.txt'))
#cmvn_var=torch.Tensor(np.loadtxt('cmvn_var.txt'))
#istd=1/cmvn_var
model = conformermodel(80, 15, 3,80,train_flag=False)
model.load_state_dict(torch.load('./conformer_out1.pt',map_location=torch.device('cpu')),strict=False)
model.eval()
audio_feature=torch.zeros((1,200,80))
torch.onnx.export(model,
                 (audio_feature),
                 './conformemodel.onnx',
                 export_params=True,
                 input_names=['input_feature'],
                 output_names=['ctc_out'],
                 dynamic_axes={
                    'input_feature':{1:'T'},
                    'ctc_out':{1:'T_out'},
                 },
                 verbose=False)