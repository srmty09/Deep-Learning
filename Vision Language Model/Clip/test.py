"""
Stage-1: contrastive alignment of ViT-CLS to frozen BERT-CLS via sigmoid loss.

  image   -> ViT-base + LoRA (DoRA + rsLoRA) -> CLS (768) -> L2 norm -> z_img
  caption -> BERT-base (frozen)              -> CLS (768) -> L2 norm -> z_txt
  loss    -> -F.logsigmoid(labels * (z_img @ z_txt.T) * temp).mean()

This is the same recipe that converged in the notebook (val 0.12 by epoch 8):
single learnable temperature, no bias term, .mean() over the BxB grid,
OneCycleLR. The production plumbing (PEFT save_pretrained, AMP skip-check,
worker_init_fn, etc.) is added on top.
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    ViTModel,
    ViTImageProcessor,
    BertModel,
    BertTokenizerFast,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm


VISION_MODEL_NAME = "google/vit-base-patch16-224"
TEXT_MODEL_NAME   = "google-bert/bert-base-uncased"

MAX_SEQ_LEN  = 32
BATCH_SIZE   = 32           # matches notebook recipe
NUM_EPOCHS   = 10
LR           = 3e-4
WEIGHT_DECAY = 1e-2
PCT_START    = 0.1          # OneCycleLR warmup fraction
GRAD_CLIP    = 1.0
SEED         = 42
SAVE_DIR     = "checkpoints"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA     = DEVICE == "cuda"
NUM_WORKERS  = min(2, os.cpu_count() or 1)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

class COCOCaptionDataset(Dataset):
    def __init__(self, hf_split, processor, tokenizer, max_length=MAX_SEQ_LEN):
        self.ds         = hf_split
        self.processor  = processor
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img    = sample["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")

        caps    = sample["answer"]
        caption = (random.choice(caps) if isinstance(caps, list) else caps).strip().lower()

        tok = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        img_t = self.processor(images=img, return_tensors="pt")
        return {
            "pixel_values":   img_t["pixel_values"].squeeze(0),
            "input_ids":      tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }


def worker_init_fn(worker_id):
    # random.choice on captions isn't correlated across workers / epochs
    random.seed((torch.initial_seed() + worker_id) % (2**32))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class SigLIPModel(nn.Module):
    """ViT (with LoRA, trainable) + BERT (frozen).
    Loss: -F.logsigmoid(labels * (z_img @ z_txt.T) * temp).mean()
    Single learnable temperature, no bias term — same as the notebook."""

    def __init__(self, vit, bert):
        super().__init__()
        self.vit  = vit
        self.bert = bert
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.bert.eval()

        # scalar log_temp -> temp starts at exp(log(10)) = 10
        self.log_temp = nn.Parameter(torch.ones([]) * math.log(10.0))

    def train(self, mode=True):
        super().train(mode)
        self.bert.eval()                       # never let BERT dropout activate
        return self

    def encode_image(self, pixel_values):
        cls = self.vit(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        return F.normalize(cls, dim=-1)

    @torch.no_grad()
    def encode_text(self, input_ids, attention_mask):
        cls = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state[:, 0, :]
        return F.normalize(cls, dim=-1)

    def sigmoid_loss(self, img, txt):
        # fp32 for numerical stability under AMP.
        img  = img.float()
        txt  = txt.float()
        temp = self.log_temp.float().exp().clamp(min=1.0, max=100.0)

        logits = (img @ txt.T) * temp
        n      = logits.size(0)
        labels = 2.0 * torch.eye(n, device=logits.device, dtype=logits.dtype) - 1.0
        return -F.logsigmoid(labels * logits).mean()

    def forward(self, pixel_values, input_ids, attention_mask):
        img = self.encode_image(pixel_values)
        txt = self.encode_text(input_ids, attention_mask)
        return self.sigmoid_loss(img, txt)


# ----------------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------------

def save_checkpoint(model, save_dir, tag):
    """Saves the LoRA adapter via PEFT (so stage-2 can do
    `PeftModel.from_pretrained(base_vit, '<adapter_dir>')`) and the
    SigLIP scalar separately."""
    model.vit.save_pretrained(os.path.join(save_dir, f"vit_lora_{tag}"))
    torch.save({
        "log_temp": model.log_temp.detach().cpu(),
    }, os.path.join(save_dir, f"siglip_scalars_{tag}.pt"))


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    if USE_CUDA:
        torch.cuda.manual_seed_all(SEED)

    processor = ViTImageProcessor.from_pretrained(VISION_MODEL_NAME)
    tokenizer = BertTokenizerFast.from_pretrained(TEXT_MODEL_NAME)

    raw   = load_dataset("lmms-lab/COCO-Caption")
    base  = raw["val"] if "val" in raw else raw[next(iter(raw))]
    split = base.train_test_split(test_size=0.05, seed=SEED)
    print(f"train: {len(split['train'])}  |  val: {len(split['test'])}")

    train_loader = DataLoader(
        COCOCaptionDataset(split["train"], processor, tokenizer),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA, drop_last=True,
        worker_init_fn=worker_init_fn, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        COCOCaptionDataset(split["test"], processor, tokenizer),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=USE_CUDA, drop_last=True,
        worker_init_fn=worker_init_fn, persistent_workers=NUM_WORKERS > 0,
    )

    bert = BertModel.from_pretrained(TEXT_MODEL_NAME)
    vit  = ViTModel.from_pretrained(VISION_MODEL_NAME)
    # "dense" matches attention.output.dense, intermediate.dense, AND
    # output.dense, so we cover Q/K/V + attn-out + both FFN projections.
    vit  = get_peft_model(vit, LoraConfig(
        r              = 32,
        lora_alpha     = 64,
        target_modules = ["query", "key", "value", "dense"],
        lora_dropout   = 0.1,
        bias           = "none",
        use_rslora     = True,
        use_dora       = True,
    ))
    vit.print_trainable_parameters()

    model = SigLIPModel(vit, bert).to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable, lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=PCT_START,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=USE_CUDA)

    os.makedirs(SAVE_DIR, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        # ---- train ----
        model.train()
        train_loss = 0.0
        n_steps    = 0
        bar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}", leave=False)
        for batch in bar:
            pv  = batch["pixel_values"].to(DEVICE, non_blocking=True)
            ids = batch["input_ids"].to(DEVICE, non_blocking=True)
            am  = batch["attention_mask"].to(DEVICE, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=USE_CUDA):
                loss = model(pv, ids, am)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP,
            )
            prev = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= prev:
                scheduler.step()

            l = loss.item()
            train_loss += l
            n_steps    += 1
            bar.set_postfix(
                loss=f"{l:.4f}",
                temp=f"{model.log_temp.detach().exp().item():.3f}",
                lr=f"{scheduler.get_last_lr()[0]:.6f}",
            )

        train_loss /= max(1, n_steps)

        # ---- val ----
        model.eval()
        val_loss = 0.0
        n_steps  = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                pv  = batch["pixel_values"].to(DEVICE, non_blocking=True)
                ids = batch["input_ids"].to(DEVICE, non_blocking=True)
                am  = batch["attention_mask"].to(DEVICE, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=USE_CUDA):
                    val_loss += model(pv, ids, am).item()
                n_steps += 1
        val_loss /= max(1, n_steps)

        print(
            f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
            f"| Temp: {model.log_temp.exp().item():.3f}"
        )

        # ---- save ----
        save_checkpoint(model, SAVE_DIR, f"epoch{epoch:02d}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, SAVE_DIR, "best")
            print(f"  Best model saved at epoch {epoch}  (val {best_val:.4f})")


if __name__ == "__main__":
    main()
