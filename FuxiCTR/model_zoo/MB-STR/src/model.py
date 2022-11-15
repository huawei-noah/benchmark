
import torch
import pytorch_lightning as pl
from .models import CGCDotProductPredictionHead, DotProductPredictionHead
from .models.bert4rec import BERT
from .utils import recalls_and_ndcgs_for_ks


class RecModel(pl.LightningModule):
    def __init__(self,
            backbone: BERT,
            b_head: bool = False,
        ):
        super().__init__()
        self.backbone = backbone
        self.n_b = backbone.n_b
        if b_head:
            self.head = CGCDotProductPredictionHead(backbone.d_model, self.n_b, 3, 1, backbone.num_items, self.backbone.embedding.token)
        else:
            self.head = DotProductPredictionHead(backbone.d_model, backbone.num_items, self.backbone.embedding.token)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, input_ids, b_seq):
        return self.backbone(input_ids, b_seq)
        

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        outputs = self(input_ids, b_seq)
        outputs = outputs.view(-1, outputs.size(-1))  # BT x H
        labels = batch['labels']
        labels = labels.view(-1)  # BT

        valid = labels>0
        valid_index = valid.nonzero().squeeze()  # M
        valid_outputs = outputs[valid_index]
        valid_b_seq = b_seq.view(-1)[valid_index] # M
        valid_labels = labels[valid_index]
        valid_logits = self.head(valid_outputs, valid_b_seq) # M

        loss = self.loss(valid_logits, valid_labels)
        loss = loss.unsqueeze(0)
        return {'loss':loss}

        
    def training_epoch_end(self, training_step_outputs):
        loss = torch.cat([o['loss'] for o in training_step_outputs], 0).mean()
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        b_seq = batch['behaviors']
        outputs = self(input_ids, b_seq)

        # get scores (B x C) for evaluation
        last_outputs = outputs[:, -1, :]
        last_b_seq = b_seq[:,-1]
        candidates = batch['candidates'].squeeze() # B x C
        logits = self.head(last_outputs, last_b_seq, candidates)
        labels = batch['labels'].squeeze()
        metrics = recalls_and_ndcgs_for_ks(logits, labels, [1, 5, 10, 20, 50])

        return metrics
    
    def validation_epoch_end(self, validation_step_outputs):
        keys = validation_step_outputs[0].keys()
        for k in keys:
            tmp = []
            for o in validation_step_outputs:
                tmp.append(o[k])
            self.log(f'Val:{k}', torch.Tensor(tmp).mean())