import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig
from transformers.tokenization_utils_base import BatchEncoding
from transformers.modeling_outputs import SequenceClassifierOutput

from tqdm import trange, tqdm
from tqdm.notebook import tqdm_notebook
from pathlib import Path
import joblib
import os
import gc

def default_collator(batch, pad_id=1):
    """
    Pads tokens to correct size and turns labels into a tensor.
    """
    out = {}
    out['labels'] = torch.tensor([sample['labels'] for sample in batch])
    out['input_ids'] = [torch.tensor(sample['input_ids']) for sample in batch]
    out['input_ids'] = pad_sequence(out['input_ids'], padding_value=pad_id, batch_first=True)[:, :200]
    out['attention_mask'] = [torch.tensor(sample['attention_mask']) for sample in batch]
    out['attention_mask'] = pad_sequence(out['attention_mask'], padding_value=0, batch_first=True)[:, :200]
    return BatchEncoding(out)

def bilingual_collator(batch, pad_id=1):
    """
    Pads tokens to correct size and turns labels into a tensor.
    """
    out = {}
    out['labels'] = torch.tensor([sample['labels'] for sample in batch])
    for lang in ['tl', 'en']:
        out[f'{lang}_input_ids'] = [torch.tensor(sample[f'{lang}_input_ids']) for sample in batch]
        out[f'{lang}_input_ids'] = pad_sequence(out[f'{lang}_input_ids'], padding_value=pad_id, batch_first=True)[:, :200]
        out[f'{lang}_attention_mask'] = [torch.tensor(sample[f'{lang}_attention_mask']) for sample in batch]
        out[f'{lang}_attention_mask'] = pad_sequence(out[f'{lang}_attention_mask'], padding_value=0, batch_first=True)[:, :200]
    return BatchEncoding(out)

def train(model: torch.nn.Module, dataset: Dataset, epochs: int, collator=default_collator, 
          device=torch.device('cuda'), parameters=None, lr=1e-3, batch_size=4, model_dir=None,
          weight_decay=0) -> None:

    train_loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loader = DataLoader(dataset['test'], batch_size=batch_size, shuffle=True, collate_fn=collator)

    if isinstance(model_dir, str):
        model_dir = Path(model_dir)

    model.to(device)
    if parameters is None:
        parameters = model.parameters()
    opt = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, epochs=epochs, steps_per_epoch=len(train_loader), max_lr=lr)
    scaler = GradScaler()

    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    best_eval_loss = 1e9
    losses = []
    for epoch in trange(epochs, position=0):

        train_loss = 0.0
        model.train()
        with tqdm(train_loader, unit='batch', total=len(train_loader), position=1) as train_iter:
            for i, batch in enumerate(train_iter, start=1):
                try:
                    opt.zero_grad()
                    batch = batch.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out = model(**batch)
                    
                    train_loss += out['loss'].detach().item()

                    scaler.scale(out['loss']).backward()
                    scaler.step(opt)
                    scaler.update()

                    # out['loss'].backward()
                    # opt.step()

                    scheduler.step()
                    train_iter.set_postfix(loss=train_loss/i)
                    
                except RuntimeError as e:
                    print(batch['input_ids'].shape)
                    print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
                    raise e
                    
        eval_loss = 0.0
        model.eval()
        with tqdm_notebook(eval_loader, unit='batch', total=len(eval_loader), position=1) as eval_iter:
            with torch.no_grad():
                for i, batch in enumerate(eval_iter, start=1):
                    opt.zero_grad()
                    batch = batch.to(device)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out = model(**batch)
                    eval_loss += out['loss'].item()
                    eval_iter.set_postfix(loss=eval_loss/i)
        
        losses.append((train_loss / len(train_loader), eval_loss / len(eval_loader)))
        if model_dir:
            torch.save(model.state_dict(), model_dir / 'model.pt')
            joblib.dump(losses, model_dir / 'losses.pkl')
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
        
        gc.collect()
        torch.cuda.empty_cache()
    
    model.load_state_dict(torch.load(model_dir / 'model.pt'))
    return losses

def evaluate(model: torch.nn.Module, eval_dataset: Dataset, collator=default_collator, 
          device=torch.device('cuda'), batch_size=16):
    
    model.eval()
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collator)
    eval_loss = 0.0
    with tqdm_notebook(eval_loader, unit='batch', total=len(eval_loader)) as eval_iter:
        with torch.no_grad():
            for i, batch in enumerate(eval_iter, start=1):
                batch = batch.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    out = model(**batch)
                eval_loss += out['loss'].item()
                eval_iter.set_postfix(loss=eval_loss/i)
    
    gc.collect()
    torch.cuda.empty_cache()
    return eval_loss / i

def get_best_epoch(losses):
    losses = [l[1] for l in losses]
    best_epoch = min(list(range(len(losses))), key=lambda i: losses[i])
    return (best_epoch, losses[best_epoch])

class RegularizedCLSModel(torch.nn.Module):
    def __init__(self, name, alpha=0.1, dropout=0.1):
        super().__init__()
        config = AutoConfig.from_pretrained(name)
        config.hidden_dropout_prob = dropout
        config.attention_probs_dropout_prob = dropout
        self.model = AutoModelForSequenceClassification.from_pretrained(name, 
            hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
        self.default_parameters = [p.detach().clone() for p in self.get_parameters()]
        self.alpha = alpha
    
    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        if 'loss' in out.keys() and self.training:
            difference = self.get_difference()
            out['loss'] = out['loss'] + (self.alpha * difference)
        return out
    
    def get_difference(self):
        diff = sum(torch.sqrt((x - y).pow(2).mean() + 1e-16)
                for x, y in zip(self.get_parameters(), self.default_parameters)) 
        return diff
    
    def get_parameters(self):
        return list(self.model.parameters())
    
    def get_non_cls_parameters(self):
        return list(list(self.model.modules())[1].parameters())
    
    def get_cls_parameters(self):
        return list(self.model.classifier.parameters())
    
    def to(self, device):
        super().to(device)
        self.default_parameters = [p.to(device) for p in self.default_parameters]

class BilingualCLSModel(torch.nn.Module):
    """
    Encodes both Filipino and English representations of a text, concatenates them, then classifies them.
    """
    def __init__(self, tl_name, en_name, alpha=0.1, dropout=0.1):
        super().__init__()
        self.tl_model = RegularizedCLSModel(tl_name, alpha, dropout)
        self.en_model = RegularizedCLSModel(en_name, alpha, dropout)
        self.tl_encoder = self.tl_model.model.roberta
        self.en_encoder = self.en_model.model.bert
        
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(768, eps=1e-12)
        self.hidden_1 = nn.Linear(1536, 768)
        self.classifier = nn.Linear(768, 2)
        
        self.loss_func = nn.CrossEntropyLoss()
    
    def forward(self, tl_input_ids, tl_attention_mask, en_input_ids, en_attention_mask, labels):
        tl_out = self.tl_encoder(input_ids = tl_input_ids, attention_mask = tl_attention_mask).last_hidden_state[:, 0]
        en_out = self.en_encoder(input_ids = en_input_ids, attention_mask = en_attention_mask).last_hidden_state[:, 0]
        out = torch.cat([tl_out, en_out], axis=1)
        
        out = self.dropout(self.layer_norm(F.gelu(self.hidden_1(out))))
        logits = self.classifier(out)
        
        loss = self.loss_func(logits, labels) + self.alpha * (self.tl_model.get_difference() + self.en_model.get_difference())
        
        return SequenceClassifierOutput(
            loss = loss,
            logits = logits
        )
    
    def get_cls_parameters(self):
        return list(self.hidden_1.parameters()) + list(self.classifier.parameters())
    
    def get_non_cls_parameters(self):
        return self.tl_model.get_non_cls_parameters() + self.en_model.get_non_cls_parameters()
    
    def get_parameters(self):
        return self.tl_model.get_parameters() + self.en_model.get_parameters()
    
    def to(self, device):
        super().to(device)
        self.tl_model.to(device)
        self.en_model.to(device)
