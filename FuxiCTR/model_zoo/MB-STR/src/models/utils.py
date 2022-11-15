
from torch import nn as nn
import torch
import math
import torch.nn.functional as F

class GradMultiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        res = x.new(x)
        return res

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.scale, None

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.apply(self._init_weights)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

class SublayerConnection(nn.Module):
    """
    sublayer connection with behavior specific layer norm
    """
    def __init__(self, size, dropout=0):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))

class BehaviorSpecificPFF(nn.Module):
    """
    Behavior specific pointwise feedforward network.
    """
    def __init__(self, d_model, d_ff, n_b, bpff=False, dropout=0.1):
        super().__init__()
        self.n_b = n_b
        self.bpff = bpff
        if bpff and n_b > 1:
            self.pff = nn.ModuleList([PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout) for i in range(n_b)])
        else:
            self.pff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def multi_behavior_pff(self, x, b_seq):
        """
        x: B x T x H
        b_seq: B x T, 0 means padding.
        """
        outputs = [torch.zeros_like(x)]
        for i in range(self.n_b):
            outputs.append(self.pff[i](x))
        return torch.einsum('nBTh, BTn -> BTh', torch.stack(outputs, dim=0), F.one_hot(b_seq, num_classes=self.n_b+1).float())
    
    def forward(self, x, b_seq=None):
        if self.bpff and self.n_b > 1:
            return self.multi_behavior_pff(x, b_seq)
        else:
            return self.pff(x)

class MMoE(nn.Module):
    def __init__(self, d_model, d_ff, n_b, n_e=1, bmmoe=False, dropout=0.1):
        super(MMoE, self).__init__()
        self.n_b = n_b
        self.n_e = n_e
        self.bmmoe = bmmoe
        if self.bmmoe and n_e > 1:
            self.softmax = nn.Softmax(dim=-1)
            self.experts = nn.ModuleList([PositionwiseFeedForward(d_model, d_ff, dropout) for i in range(self.n_e)])
            self.w_gates = nn.Parameter(torch.randn(self.n_b, d_model, self.n_e), requires_grad=True)
        else:
            self.pff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

    def forward(self, x, b_seq):
        if self.bmmoe and self.n_e > 1:
            experts_o = [e(x) for e in self.experts]
            experts_o_tensor = torch.stack(experts_o)
            gates_o = self.softmax(torch.einsum('bnd,tde->tbne', x, self.w_gates))
            output = torch.einsum('ebnd,tbne->tbnd', experts_o_tensor, gates_o)
            outputs = torch.cat([torch.zeros_like(x).unsqueeze(0), output])
            return torch.einsum('tbnd, bnt -> bnd', outputs, F.one_hot(b_seq, num_classes=self.n_b+1).float())
        else:
             return self.pff(x)