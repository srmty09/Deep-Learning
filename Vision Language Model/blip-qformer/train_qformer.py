import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTImageProcessor, ViTModel, DistilBertTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from vit import VIT
from q_former import Qformer, QformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
pretrained_vit = ViTModel.from_pretrained(model_name)
pretrained_vit.eval()
vit = VIT(pretrained_vit).to(device)
cfg = QformerConfig()
qformer = Qformer(cfg).to(device)
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

ds = load_dataset("lmms-lab/COCO-Caption")
train_ds = ds['val']
test_ds = ds['test']


class CaptionDataset(Dataset):
    def __init__(self, ds, image_processor, text_tokenizer, max_length=60):
        self.ds = ds
        self.processor = image_processor
        self.tokenizer = text_tokenizer
        self.max_length = max_length

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image']
        caption = item['answer'][0]
        if image.mode != 'RGB':
            image = image.convert('RGB')
        pixel_values = self.processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        tokenized = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        return {
            'pixel_values': pixel_values,
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
        }


train_dataset = CaptionDataset(train_ds, processor, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    pin_memory=True,
    num_workers=2,
)


def itc_features(qformer, image_emb, input_ids, attention_mask):
    """Run the two uni-modal passes and return L2-normalised pooled features
    plus the raw similarity matrix used for hard-negative mining in ITM."""
    img_query_out = qformer.encode_image(image_emb)              # (B, Nq, d)
    text_out      = qformer.encode_text(input_ids, attention_mask)  # (B, L, d)

    image_feat = F.normalize(img_query_out.mean(dim=1), dim=-1)
    text_feat  = F.normalize(text_out[:, 0, :], dim=-1)
    return image_feat, text_feat


def loss_itc(image_feat, text_feat, logit_scale):
    B = image_feat.shape[0]
    scale = logit_scale.exp().clamp(max=100.0)
    sim_matrix = (image_feat @ text_feat.T) * scale
    labels = torch.arange(B, device=image_feat.device)

    itc_acc_i2t = (sim_matrix.argmax(dim=1) == labels).float().mean()
    itc_acc_t2i = (sim_matrix.T.argmax(dim=1) == labels).float().mean()
    itc_accuracy = (itc_acc_i2t + itc_acc_t2i) / 2

    loss = (F.cross_entropy(sim_matrix, labels) + F.cross_entropy(sim_matrix.T, labels)) / 2
    return loss, itc_accuracy


def loss_itm(qformer, image_emb, input_ids, attention_mask,
             image_feat, text_feat, itm_head):
    """Real BLIP-2 style ITM:
       1. mine hard negatives from the (no-grad) ITC similarity matrix
       2. build a 3*B batch of (positive, hard-neg-text, hard-neg-image) pairs
       3. run a *second* multimodal forward pass through the Q-Former so the
          queries actually attend to the candidate text — this is what gives
          the head a real signal of "match vs not match"
       4. classify the per-query outputs (averaged over queries) with `itm_head`.

    The 3B layout has a 1:2 positive:negative ratio. We compensate with a
    class-weighted cross-entropy ([1, 2] for [neg, pos]) so that the trivial
    "always predict 0" baseline is no longer a local minimum (it would otherwise
    achieve 2/3 accuracy at log(2) loss before the head ever learns anything).
    """
    B = image_emb.shape[0]
    if B < 2:
        zero = sum(p.sum() for p in qformer.parameters() if p.requires_grad) * 0.0
        return zero, torch.tensor(0.0, device=image_emb.device)

    with torch.no_grad():
        sim = (image_feat @ text_feat.T).clone()
        sim.fill_diagonal_(-float('inf'))
        hard_neg_text_idx  = sim.argmax(dim=1)
        hard_neg_image_idx = sim.argmax(dim=0)

    image_emb_pos = image_emb
    image_emb_neg = image_emb[hard_neg_image_idx]
    input_ids_neg = input_ids[hard_neg_text_idx]
    attn_mask_neg = attention_mask[hard_neg_text_idx]

    all_image_emb      = torch.cat([image_emb_pos, image_emb_pos, image_emb_neg], dim=0)
    all_input_ids      = torch.cat([input_ids,     input_ids_neg, input_ids],      dim=0)
    all_attention_mask = torch.cat([attention_mask, attn_mask_neg, attention_mask], dim=0)

    query_out_mm = qformer.forward_multimodal(all_image_emb, all_input_ids, all_attention_mask)
    # (3B, Nq, d) -> per-query 2-class logits -> average over queries -> (3B, 2)
    logits = itm_head(query_out_mm).mean(dim=1)

    labels = torch.cat([
        torch.ones(B,  dtype=torch.long, device=image_emb.device),
        torch.zeros(B, dtype=torch.long, device=image_emb.device),
        torch.zeros(B, dtype=torch.long, device=image_emb.device),
    ])

    # Class weights: [neg=1, pos=2] cancels the 1:2 sampling imbalance.
    class_weight = torch.tensor([1.0, 2.0], device=image_emb.device)
    loss = F.cross_entropy(logits, labels, weight=class_weight)

    predicted = logits.argmax(dim=1)
    itm_accuracy = (predicted == labels).float().mean()
    return loss, itm_accuracy


class Trainer:
    def __init__(self, vit, qformer, train_loader, epochs=10, lr=1e-4,
                 grad_clip=1.0, itm_warmup_steps=500):
        self.device = device
        self.vit = vit
        self.qformer = qformer
        self.train_loader = train_loader
        self.epochs = epochs
        self.grad_clip = grad_clip
        # Linearly ramp the ITM loss weight from 0 to 1 over the first
        # `itm_warmup_steps` steps. This avoids wasting capacity on a head that
        # cannot yet discriminate anything, and lets ITC bootstrap modality
        # alignment before ITM starts demanding fine-grained matching.
        self.itm_warmup_steps = itm_warmup_steps
        self._global_step = 0
        # ITM head: per-query d_model vector -> (match, no-match) logits, averaged
        # over queries. We initialise the bias to the log-class-prior of the 3B
        # layout (1B positives, 2B negatives) so the head starts at the sample
        # distribution instead of having to descend into "always predict 0" first.
        self.itm_head = nn.Linear(qformer.cfg.d_model, 2).to(self.device)
        with torch.no_grad():
            self.itm_head.bias[0] = math.log(2.0 / 3.0)  # P(no-match)
            self.itm_head.bias[1] = math.log(1.0 / 3.0)  # P(match)

        for param in self.vit.parameters():
            param.requires_grad = False
        self.vit.eval()

        self.optimizer = torch.optim.AdamW([
            {'params': self.qformer.parameters()},
            {'params': self.itm_head.parameters()},
        ], lr=lr, weight_decay=0.05)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=len(train_loader) * epochs
        )

    def fit(self):
        self.qformer.train()
        self.itm_head.train()
        self.vit.eval()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for i, batch in enumerate(self.train_loader):
                pixel_values   = batch['pixel_values'].to(self.device, non_blocking=True)
                input_ids      = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)

                image_emb = self.vit(pixel_values)

                # ----- ITC: two clean uni-modal forward passes -----
                image_feat, text_feat = itc_features(
                    self.qformer, image_emb, input_ids, attention_mask
                )
                l_itc, acc_itc = loss_itc(image_feat, text_feat, self.qformer.logit_scale)

                # ----- ITM: second multimodal forward pass with hard negatives -----
                l_itm, acc_itm = loss_itm(
                    self.qformer, image_emb, input_ids, attention_mask,
                    image_feat.detach(), text_feat.detach(),
                    self.itm_head,
                )

                itm_weight = (
                    1.0 if self.itm_warmup_steps <= 0
                    else min(1.0, self._global_step / float(self.itm_warmup_steps))
                )
                loss = l_itc + itm_weight * l_itm
                self._global_step += 1

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.qformer.parameters()) + list(self.itm_head.parameters()),
                    self.grad_clip,
                )
                self.optimizer.step()
                self.scheduler.step()

                # Keep the CLIP temperature in a sane range, matching CLIP/BLIP practice.
                with torch.no_grad():
                    self.qformer.logit_scale.clamp_(0.0, math.log(100.0))

                total_loss += loss.item()
                temp_val = (1.0 / self.qformer.logit_scale.exp().clamp(max=100.0)).item()

                print(
                    f"Epoch {epoch+1}/{self.epochs}, "
                    f"Batch {i+1}/{len(self.train_loader)}: "
                    f"Loss={loss.item():.4f}, "
                    f"ITC={l_itc.item():.4f} (Acc={acc_itc.item():.4f}), "
                    f"ITM={l_itm.item():.4f} (Acc={acc_itm.item():.4f}, w={itm_weight:.2f}), "
                    f"T={temp_val:.4f}, "
                    f"LR={self.scheduler.get_last_lr()[0]:.6f}"
                )

            avg = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.epochs}  Average Loss={avg:.4f}")


if __name__ == "__main__":
    trainer = Trainer(vit, qformer, train_loader, epochs=10, lr=1e-4)
    trainer.fit()
