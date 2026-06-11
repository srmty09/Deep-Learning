import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import BlipProcessor
from PIL import Image
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from tqdm import tqdm
from vlm_model import VisionLanguageModel


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class VisionLanguageDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: BlipProcessor,
        tokenizer,
        max_text_length: int = 256,
        image_size: int = 224,
        split: str = "train",
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.image_size = image_size

        with open(data_path, "r") as f:
            all_data = json.load(f)

        split_idx = int(len(all_data) * 0.9)
        self.data = all_data[:split_idx] if split == "train" else all_data[split_idx:]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        caption = item["caption"]

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        prompt = f"<image> {item.get('question', 'Describe this image:')} "
        full_text = prompt + caption + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        prompt_encoding = self.tokenizer(prompt, return_tensors="pt")
        prompt_len = prompt_encoding.input_ids.shape[1]

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class SyntheticVLDataset(Dataset):
    def __init__(self, processor, tokenizer, num_samples: int = 100):
        from synthetic_image_gen import generate_sample
        self.generate_sample = generate_sample
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        image, caption, question = self.generate_sample(idx)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.squeeze(0)

        prompt = f"<image> {question} "
        full_text = prompt + caption + self.tokenizer.eos_token

        encoding = self.tokenizer(
            full_text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]

        input_ids = encoding.input_ids.squeeze(0)
        attention_mask = encoding.attention_mask.squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class VLMTrainer:
    def __init__(
        self,
        model: VisionLanguageModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        output_dir: str = "./vlm_checkpoints",
        batch_size: int = 4,
        num_epochs: int = 3,
        learning_rate: float = 2e-4,
        vision_lr_multiplier: float = 0.1,
        warmup_steps: int = 100,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        save_every_n_steps: int = 500,
        eval_every_n_steps: int = 200,
        device: str = "auto",
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = float(max_grad_norm)
        self.save_every_n_steps = save_every_n_steps
        self.eval_every_n_steps = eval_every_n_steps
        learning_rate = float(learning_rate)
        vision_lr_multiplier = float(vision_lr_multiplier)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        self.val_loader = None
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
            )

        vision_params = (
            list(model.vision_encoder.parameters())
            + list(model.cross_attention.parameters())
            + list(model.vision_token_compressor.parameters())
            + [model.vision_query]
        )
        vision_param_ids = {id(p) for p in vision_params}
        lm_params = [p for p in model.parameters() if id(p) not in vision_param_ids]

        self.optimizer = AdamW([
            {"params": vision_params, "lr": learning_rate * vision_lr_multiplier},
            {"params": lm_params,     "lr": learning_rate},
        ], weight_decay=0.01)

        total_steps = max(1, len(self.train_loader) * num_epochs // gradient_accumulation_steps)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=1e-6)
        self.global_step = 0
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(progress_bar):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()

            total_loss += outputs.loss.item()
            num_batches += 1

            if (step + 1) % self.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.eval_every_n_steps == 0 and self.val_loader:
                    val_loss = self.evaluate()
                    logger.info(f"Step {self.global_step} - Val Loss: {val_loss:.4f}")
                    self.model.train()

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best_model")

                if self.global_step % self.save_every_n_steps == 0:
                    self.save_checkpoint(f"step_{self.global_step}")

            avg_loss = total_loss / num_batches
            progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})

        return total_loss / num_batches

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

        return total_loss / num_batches

    def save_checkpoint(self, name: str):
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.vision_encoder.state_dict(), checkpoint_dir / "vision_encoder.pt")
        torch.save(self.model.vision_query, checkpoint_dir / "vision_query.pt")
        torch.save(self.model.cross_attention.state_dict(), checkpoint_dir / "cross_attention.pt")
        torch.save(self.model.vision_token_compressor.state_dict(), checkpoint_dir / "compressor.pt")
        self.model.lm.save_pretrained(checkpoint_dir / "lm")
        self.model.tokenizer.save_pretrained(checkpoint_dir / "tokenizer")

        config = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }
        with open(checkpoint_dir / "training_state.json", "w") as f:
            json.dump(config, f)

        logger.info(f"Checkpoint saved: {checkpoint_dir}")

    def train(self):
        logger.info(f"Training on {self.device} | Epochs: {self.num_epochs} | Steps/epoch: {len(self.train_loader)}")

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Train Loss: {train_loss:.4f}")

            if self.val_loader:
                val_loss = self.evaluate()
                logger.info(f"Epoch {epoch + 1}/{self.num_epochs} - Val Loss: {val_loss:.4f}")

            self.save_checkpoint(f"epoch_{epoch + 1}")

        logger.info("Training complete.")
