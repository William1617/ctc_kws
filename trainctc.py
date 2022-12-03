
import time
import random
import logging
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.autograd import Variable
import torch.nn.functional as F
from conformer import conformermodel
from audioloader import audiodataset
from Processor import data_processing

import numpy as np

def train(train_loader,dev_loader,out_dir,vocab_size):

    logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
    random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
    patience=20
    #cmvn_mean=torch.Tensor(np.loadtxt('cmvn_mean.txt'))
    # #cmvn_var=torch.Tensor(np.loadtxt('cmvn_var.txt'))
    # #istd=1/cmvn_var

    model = conformermodel(80, vocab_size, 3,80,train_flag=True)
    model.cuda()
    device = torch.device('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity =True).to(device)
    prev_loss = 10000
    lr = 1e-5
    model_path=out_dir
    for epoch in range(1, 500):
        losses = []
        start_time = time.time()
        for i, (xs, ys, xlen, ylen) in enumerate(train_loader):
            if i == 5000:
                break
            x = Variable(torch.FloatTensor(xs))
            x = torch.squeeze(x,1).transpose(1,2)
            x = x.cuda()
            x_len=xlen.cuda()
            ys = np.hstack([ys[i, :j] for i, j in enumerate(ylen)])
            y = Variable(torch.IntTensor(ys)) 
            model.train()
            optimizer.zero_grad()
            out = model(x,x_len)
          
            xl = Variable(torch.IntTensor(xlen))
            yl = Variable(torch.IntTensor(ylen))
            out = F.log_softmax(out, dim=2)
            loss = criterion(out.transpose(0,1).contiguous(), y, xl, yl)
            loss.backward()
            loss = float(loss.data) * len(xlen) # batch size
            losses.append(loss)
            
            optimizer.step()

        tol_loss = sum(losses) /len(train_loader)
       
        val_losses = []
        with torch.no_grad():
            for i, (xs, ys, xlen, ylen) in enumerate(dev_loader):
                x = Variable(torch.FloatTensor(xs), volatile=True)
                x = torch.squeeze(x,1).transpose(1,2)
                x=x.cuda()
                x_len=xlen.cuda()
                ys = np.hstack([ys[i, :j] for i, j in enumerate(ylen)])
                y = Variable(torch.IntTensor(ys))
                model.eval()
                out = model(x,x_len)
                xl = Variable(xlen)
                yl = Variable(torch.IntTensor(ylen))
                out = F.log_softmax(out, dim=2)
                val_loss = criterion(out.transpose(0,1).contiguous(), y, xl, yl)
                val_loss = float(val_loss.data) * len(xlen) # batch size
                val_losses.append(val_loss)

        tol_valoss=sum(val_losses)/len(dev_loader)
        logging.info('[Epoch %d] time cost %.2fs, train loss %.2f; cv loss %.2f,  lr %.3e'%(
            epoch, time.time()-start_time, tol_loss, tol_valoss, lr
        ))
        # Save checkpoint
        checkpoint = {"epoch": epoch + 1, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
      
        # Early stopping
        if tol_valoss < prev_loss:
            prev_loss = tol_valoss
            torch.save(model.state_dict(), model_path)  
            torch.save(checkpoint,'conformercheckpoint')   
        else:
            patience=patience-1
        if patience ==0:
            break
   
if __name__ == '__main__':
    train_dir = './/data//trainwav//'
    dev_dir = './/data//testwav//'
    train_set=audiodataset(train_dir)
    dev_set=audiodataset(dev_dir)
    train_loader = data.DataLoader(dataset=train_set, pin_memory=False,
                                batch_size=8,
                                shuffle=True,
                                num_workers=0,
                                collate_fn=lambda x: data_processing(x))
    dev_loader = data.DataLoader(dataset=dev_set, pin_memory=False,
                                batch_size=8,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=lambda x: data_processing(x))
    train(train_loader,dev_loader,'conformer_out1.pt',15)