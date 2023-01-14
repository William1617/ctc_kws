import torch
import torch.nn as nn
from typing import Tuple
import math
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
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.dropout = nn.Dropout(p=dropout_rate)
     
        self.ada_scale = nn.Parameter(
            torch.ones([1, 1, n_feat]), requires_grad=True)
        self.ada_bias = nn.Parameter(
            torch.zeros([1, 1, n_feat]), requires_grad=True)

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
        
        query = self.ada_scale * query + self.ada_bias
        key = self.ada_scale * key + self.ada_bias
        value = self.ada_scale * value + self.ada_bias
       
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
      

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        scores = matrix_ac  / math.sqrt(
            self.d_k)  # (batch, head, time1, time2)
        return self.forward_attention(v, scores,mask)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""
    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 bias: bool = True):
        assert check_argument_types()
        super().__init__()

        self.ada_scale = torch.nn.Parameter(
            torch.ones([1, 1, channels]), requires_grad=True)
        self.ada_bias = torch.nn.Parameter(
            torch.zeros([1, 1, channels]), requires_grad=True)

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
            stride=1,
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

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool)
    ) -> torch.Tensor:
        x = self.ada_scale * x + self.ada_bias
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
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2)

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
        self.ada_scale = torch.nn.Parameter(
            torch.ones([1, 1, idim]), requires_grad=True)
        self.ada_bias = torch.nn.Parameter(
            torch.zeros([1, 1, idim]), requires_grad=True)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        xs = self.ada_scale * xs + self.ada_bias
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))


