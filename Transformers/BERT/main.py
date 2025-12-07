import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
from config import ModelConfig
from dataset import Creating_BertPretraining_Dataset,BertDataset
from model import Model

ds = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
device = "cuda" if torch.cuda.is_available() else "cpu"

config = ModelConfig()
config.vocab_size = tokenizer.vocab_size

model = Model(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-6)

cbd = Creating_BertPretraining_Dataset(ds['train'])
dataset_list = cbd.get()
dataset = BertDataset(dataset_list, tokenizer)
loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=True if device == "cuda" else False)

total_steps = len(loader) * 10
warmup_steps = int(0.1 * total_steps)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=5e-5,
    total_steps=total_steps,
    pct_start=0.1,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)

print(f"Device: {device}")
print(f"Total samples: {len(dataset)}")
print(f"Batches per epoch: {len(loader)}")
print(f"Total steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}\n")
all_batch_losses = []


for epoch in range(10):
    print("model training started...")
    model.train()
    total_loss = 0
    total_mlm_loss = 0
    total_nsp_loss = 0

    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/10")

    for batch_idx, batch in enumerate(progress_bar):
        input_ids, attention_mask, token_type_ids, nsp_labels, mlm_labels = batch

        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        token_type_ids = token_type_ids.to(device)
        nsp_labels = nsp_labels.to(device)
        mlm_labels = mlm_labels.to(device)

        logits_mlm, logits_nsp = model(input_ids, token_type_ids, attention_mask)

        mlm_loss = F.cross_entropy(
            logits_mlm.view(-1, logits_mlm.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )
        nsp_loss = F.cross_entropy(logits_nsp, nsp_labels)
        loss = mlm_loss + nsp_loss

        all_batch_losses.append({
            'step': epoch * len(loader) + batch_idx,
            'epoch': epoch,
            'total_loss': loss.item(),
            'mlm_loss': mlm_loss.item(),
            'nsp_loss': nsp_loss.item(),
            'lr': optimizer.param_groups[0]["lr"]
        })

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_mlm_loss += mlm_loss.item()
        total_nsp_loss += nsp_loss.item()

        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mlm': f'{mlm_loss.item():.4f}',
            'nsp': f'{nsp_loss.item():.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })

    avg_loss = total_loss / len(loader)
    avg_mlm_loss = total_mlm_loss / len(loader)
    avg_nsp_loss = total_nsp_loss / len(loader)

    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Avg Loss: {avg_loss:.4f}")
    print(f"  Avg MLM Loss: {avg_mlm_loss:.4f}")
    print(f"  Avg NSP Loss: {avg_nsp_loss:.4f}")
    print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}\n")

print("Training completed!")

torch.save(model.state_dict(), "bert_pretrain_weights.pt")
print("Model weights saved to bert_pretrain_weights.pt")

import json
with open("bert_pretrain_losses.json", "w") as f:
    json.dump(all_batch_losses, f)
print("Loss history saved to bert_pretrain_losses.json")