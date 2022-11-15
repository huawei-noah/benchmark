

import torch
import torch.nn as nn
import torch.nn.functional as F

# head used for bert4rec
class DotProductPredictionHead(nn.Module):
    """share embedding parameters"""
    def __init__(self, d_model, num_items, token_embeddings):
        super().__init__()
        self.token_embeddings = token_embeddings
        self.vocab_size = num_items + 1
        self.out = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
            )
        self.bias = nn.Parameter(torch.zeros(1, self.vocab_size))

    def forward(self, x, b_seq, candidates=None):
        x = self.out(x)  # B x H or M x H
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  # x : M x H
            emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
            logits += self.bias
        return logits


class CGCDotProductPredictionHead(nn.Module):
    """
    model with shared expert and behavior specific expert
    3 shared expert,
    1 specific expert per behavior.
    """
    def __init__(self, d_model, n_b, n_e_sh, n_e_sp, num_items, token_embeddings):
        super().__init__()
        self.n_b = n_b
        self.n_e_sh = n_e_sh
        self.n_e_sp = n_e_sp
        self.vocab_size = num_items + 1
        self.softmax = nn.Softmax(dim=-1)
        self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model)) for i in range(self.n_e_sh)])
        self.specific_experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model)) for i in range(self.n_b * self.n_e_sp)])
        # self.shared_experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),nn.Linear(d_model, d_model)) for i in range(self.n_e_sh)])
        # self.specific_experts = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(),nn.Linear(d_model, d_model)) for i in range(self.n_b * self.n_e_sp)])
        self.w_gates = nn.Parameter(torch.randn(self.n_b, d_model, self.n_e_sh + self.n_e_sp), requires_grad=True)
        self.token_embeddings = token_embeddings
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, b_seq, candidates=None):
        x = self.mmoe_process(x, b_seq)
        if candidates is not None:  # x : B x H
            emb = self.token_embeddings(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
        else:  # x : M x H
            emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
            logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
        return logits
       
    def mmoe_process(self, x, b_seq):
        shared_experts_o = [e(x) for e in self.shared_experts]
        specific_experts_o = [e(x) for e in self.specific_experts]
        gates_o = self.softmax(torch.einsum('nd,tde->tne', x, self.w_gates))
        # rearange
        experts_o_tensor = torch.stack([torch.stack(shared_experts_o+specific_experts_o[i*self.n_e_sp:(i+1)*self.n_e_sp]) for i in range(self.n_b)])
        # torch.stack([torch.stack(shared_experts_o+specific_experts_o[i*2: (i+1)*2]) for i in range(4)])
        output = torch.einsum('tend,tne->tnd', experts_o_tensor, gates_o)
        outputs = torch.cat([torch.zeros_like(x).unsqueeze(0), output])
        return x + self.ln(torch.einsum('tnd, nt -> nd', outputs, F.one_hot(b_seq, num_classes=self.n_b+1).float()))

# class DotProductPredictionHead(nn.Module):
#     """share embedding parameters"""
#     def __init__(self, d_model, num_items, token_embeddings):
#         super().__init__()
#         self.token_embeddings = token_embeddings
#         self.vocab_size = num_items + 1

#     def forward(self, x, b_seq, candidates=None):
#         if candidates is not None:  # x : B x H
#             emb = self.token_embeddings(candidates)  # B x C x H
#             logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
#         else:  # x : M x H
#             emb = self.token_embeddings.weight[:self.vocab_size]  # V x H
#             logits = torch.matmul(x, emb.transpose(0, 1))  # M x V
#         return logits