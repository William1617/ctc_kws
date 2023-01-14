import math
from typing import Tuple

import torch
from torch import nn
from typeguard import check_argument_types


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, n_head: int, n_feat: int, dropout_rate: float):
        """Construct an MultiHeadedAttention object."""
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
        )-> torch.Tensor:

        n_batch = value.size(0)
        # for training
        if mask.size(2) > 0 :  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)
        p_attn = self.dropout(attn)
        p_attn=attn
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (x.transpose(1, 2).contiguous().view(n_batch,-1,
                                                 self.h * self.d_k)
             )  # (batch, 1,time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
                ) -> torch.Tensor:
       
        q, k, v = self.forward_qkv(query, key, value)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores,mask)
   
class GroupedMultiHeadedAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate, group_size=3):
        """Construct a GroupedMultiHeadedAttention object."""
        super().__init__()
        # linear transformation for positional encoding
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(dropout_rate)
      
        self.group_size = group_size
        self.d_k = n_feat // n_head  # for GroupedAttention
        self.n_feat = n_feat
        self.h=n_head

    def pad4group(self, Q, K, V, mask, group_size: int = 3):
        """
        q: (#batch, time1, size) -> (#batch, head, time1, size/head)
        k,v: (#batch, time2, size) -> (#batch, head, time2, size/head)
        p: (#batch, time2, size)
        """
        # Compute Overflows
        overflow_Q = Q.size(2) % group_size
        overflow_KV = K.size(2) % group_size

        padding_Q = (group_size - overflow_Q) * int(
            overflow_Q // (overflow_Q + 0.00000000000000001))
        padding_KV = (group_size - overflow_KV) * int(
            overflow_KV // (overflow_KV + 0.00000000000000001))

        batch_size, _, seq_len_KV, _ = K.size()

        # Input Padding (B, T, D) -> (B, T + P, D)
        Q = F.pad(Q, (0, 0, 0, padding_Q), value=0.0)
        K = F.pad(K, (0, 0, 0, padding_KV), value=0.0)
        V = F.pad(V, (0, 0, 0, padding_KV), value=0.0)

        if mask is not None and mask.size(2) > 0 :  # time2 > 0:
            mask = mask[:, ::group_size, ::group_size]

        Q = Q.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        K = K.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)
        V = V.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h, self.d_k * group_size).transpose(1, 2)

        return Q, K, V, mask, padding_Q

    def forward_attention(
        self, value: torch.Tensor, scores: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        padding_q: Optional[int] = None
    ) -> torch.Tensor:
       
        n_batch = value.size(0)
      
        if mask.size(2) > 0 :  # time2 > 0
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            # For last chunk, time2 might be larger than scores.size(-1)
            mask = mask[:, :, :, :scores.size(-1)]  # (batch, 1, *, time2)
            scores = scores.masked_fill(mask, -float('inf'))
            attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0)  # (batch, head, time1, time2)
    
        else:
            attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)

        # n_feat!=h*d_k may be happened in GroupAttention
        x = (x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_feat)
             )  # (batch, time1, d_model)
        if padding_q is not None:
            # for GroupedAttention in efficent conformer
            x = x[:, :x.size(1) - padding_q]
        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor,
                mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
                ) -> torch.Tensor:
    
        q = self.linear_q(query)
        k = self.linear_k(key)          # (#batch, time2, size)
        v = self.linear_v(value)

        batch_size, seq_len_KV, _ = k.size()  # seq_len_KV = time2

        # (#batch, time2, size) -> (#batch, head, time2, size/head)
        q = q.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
    
        # May be k and p does not match.  eg. time2=18+18/2=27 > mask=36/2=18
        if mask is not None and mask.size(2) > 0:
            time2 = mask.size(2)
            k = k[:, :, -time2:, :]
            v = v[:, :, -time2:, :]

        # q k v p: (batch, head, time1, d_k)
        q, k, v,  mask, padding_q = self.pad4group(q, k, v, mask, self.group_size)
 
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.d_k * self.group_size)  # (batch, head, time1, time2)

        return self.forward_attention(v, scores, mask, padding_q)


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """
    def __init__(self,
                 idim: int,
                 hidden_units: int,
                 dropout_rate: float,
                 ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))

class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 bias: bool = True,
                 stride :int=1):
        assert check_argument_types()
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            groups=channels,
            bias=bias,
        )

        self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation
        self.stride=stride

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) ->  Tuple[torch.Tensor, torch.Tensor]:
        x = x.transpose(1, 2)  # (#batch, channels, 1,time)
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)


        x = self.pointwise_conv1(x)  # (batch, 2*channel, 1,time)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, 1,time)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            if mask_pad.size(2) != x.size(2):
                mask_pad = mask_pad[:, :, ::self.stride]
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2),mask_pad
  
class Conv2dSubsampling2(nn.Module):
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(1,odim,3,2),
            nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * ((idim - 1) // 2), odim))
    
    def forward(
            self,
            x: torch.Tensor,
            mask :torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b,t,c * f)) #(b,t,odim)
        return x,mask[:, :, 2::2]


class Conv2dSubsampling4(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """
    def __init__(self, idim: int, odim: int, dropout_rate: float):
        """Construct an Conv2dSubsampling4 object."""
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim))

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        x = x.unsqueeze(1)  # (b, c=1, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b,t,c * f)) #(b,t,odim)
        return x,mask[:,:,2::2][:,:,2::2]


class GlobalCMVN(torch.nn.Module):
    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)
        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x
    
    class TimeReductionLayer1D(nn.Module):
    """
    Squeezeformer Time Reduction procedure.
    Downsamples the audio by `stride` in the time dimension.
    Args:
        channel (int): input dimension of
                       MultiheadAttentionMechanism and PositionwiseFeedForward
        out_dim (int): Output dimension of the module.
        kernel_size (int): Conv kernel size for
                           depthwise convolution in convolution module
        stride (int): Downsampling factor in time dimension.
    """

    def __init__(self, channel: int, out_dim: int,
                 kernel_size: int = 5, stride: int = 2):
        super(TimeReductionLayer1D, self).__init__()

        self.channel = channel
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
    
        self.dw_conv = nn.Conv1d(
            in_channels=channel,
            out_channels=channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2,
            groups=channel,
        )

        self.pw_conv = nn.Conv1d(
            in_channels=channel, out_channels=out_dim,
            kernel_size=1, stride=1, padding=0, groups=1,
        )
    def forward(self, xs :torch.Tensor, mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)-> Tuple[torch.Tensor, torch.Tensor],
                ):
        xs = xs.transpose(1, 2)  # [B, C, T]
        if mask.size(2) > 0:  # time > 0
            xs.masked_fill_(~mask, 0.0)

        xs = self.dw_conv(xs)
        xs = self.pw_conv(xs)

       
        mask = mask[:, :,::self.stride]
       
        if(mask.size(2)>0):
            xs.masked_fill_(~mask, 0.0)
        xs = xs.transpose(1, 2)  # [B, T, C]

        return xs, mask

class TimeRecoverlayer1D(nn.Module):

    def __init__(self,in_size,out_size):
        super().__init__()
        self.linear=nn.Linear(in_features=in_size,out_features=out_size)
    def forward(self,x :torch.Tensor,mask :torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        xs=x.unsqueeze(2).repeat(1, 1, 2, 1).flatten(1, 2)
        xs=self.linear(xs)
        out_mask=mask.transpose(1,2)
        out_mask=out_mask.unsqueeze(2).repeat(1, 1, 2, 1).flatten(1, 2)
        out_mask=out_mask.transpose(1,2)

        return xs,out_mask
