from pathlib import Path
import torch
from datasets import load_from_disk
from proj_utils import train, default_collator, bilingual_collator, RegularizedCLSModel, BilingualCLSModel, get_best_epoch
import gc
import sys
import joblib

proj_dir = Path('.')
model_dir = proj_dir / 'models'
data_dir = proj_dir / 'data'

tl_model_name = "jcblaise/roberta-tagalog-base"
xlm_model_name = "xlm-roberta-base"
en_model_name = "bert-base-uncased"

tl_model_args = {"name": tl_model_name}
xlm_model_args = {"name": xlm_model_name}
en_model_args = {"name": en_model_name}

bilingual_model_args = {
    "tl_name": tl_model_name,
    "en_name": en_model_name
}

def train_model_pipeline(n, ModelClass, model_args, collator=default_collator):
    dataset = load_from_disk(data_dir / f'dataset_{n}')
    model = ModelClass(**model_args)
    
    best_lr = 1e-5
    best_alpha = 0
    best_epoch = 0
    best_loss = 1e9
    for lr in [1e-5, 1e-6]:
        for alpha in [0, 0.1, 1, 10]:
            model.load_state_dict(torch.load(f"models/model_{n}/model.pt"))
            torch.save(model.state_dict(), "model.pt")
            model.alpha = alpha
            losses = train(model, dataset, epochs=10, model_dir=model_dir/f'model_{n}/full_{lr}_alpha_{alpha}', collator=collator, lr=lr)
            epoch, loss = get_best_epoch(losses)
            if loss < best_loss:
                best_loss = loss
                best_alpha = alpha
                best_lr = lr
                best_epoch = epoch 
            gc.collect()
            torch.cuda.empty_cache()
    
    del dataset
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "best_loss": best_loss, 
        "best_lr": best_lr, 
        "best_alpha": best_alpha, 
        "best_epoch": best_epoch
    }


train_args = {
    1: [1, RegularizedCLSModel, tl_model_args],
    2: [2, RegularizedCLSModel, xlm_model_args],
    3: [3, RegularizedCLSModel, en_model_args],
    4: [4, BilingualCLSModel, bilingual_model_args, bilingual_collator],
    5: [5, BilingualCLSModel, bilingual_model_args, bilingual_collator],
    6: [6, BilingualCLSModel, bilingual_model_args, bilingual_collator]
}

def main(n):
    best_model_stats = train_model_pipeline(*train_args[n])
    joblib.dump(best_model_stats, model_dir / f"model_{n}/results.pkl")
    print(f"Model {n}: {best_model_stats}")
    
if __name__ == "__main__":
    n = sys.argv[1]
    n = int(n)
    main(n)
