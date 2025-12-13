import random

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn import functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# --- ALBERT SOP Data Generation ---


class AlbertDataGenerator:
    def __init__(self, raw_ds):
        self.raw_ds = raw_ds

    def _read_wiki(self):
        texts = []
        for i in range(len(self.raw_ds)):
            raw = self.raw_ds["text"][i].strip().lower()
            paras = raw.split("\n\n")
            for p in paras:
                p = p.strip()
                if not p:
                    continue
                sents = self.split_into_sentences(p)
                if len(sents) == 0:
                    continue
                texts.append(sents)
        return texts

    def split_into_sentences(self, paragraph):
        parts = paragraph.split(" . ")
        sentences = []
        for s in parts:
            s = s.strip()
            if not s:
                continue
            if not s.endswith("."):
                s += "."
            if (
                s.endswith(" = =.")
                or s.endswith(" = = =.")
                or s.endswith(" =.")
                or len(s) <= 2
            ):
                continue
            sentences.append(s)
        return sentences

    # Sentence Order Prediction (SOP) Logic
    def _get_sentence_order_pair(self, s1, s2):
        # s1 and s2 are guaranteed to be consecutive sentences (s1 is followed by s2)
        if random.random() < 0.5:
            # Positive Sample (50%): Correct order
            return s1, s2, True
        else:
            # Negative Sample (50%): Reversed order (SOP)
            return s2, s1, False

    def build_dataset(self, paragraphs):
        raw_sop_data = []
        for paragraph in paragraphs:
            for i in range(len(paragraph) - 1):
                s1 = paragraph[i]
                s2 = paragraph[i + 1]

                # Generate SOP pair and label
                sentence_a, sentence_b, is_ordered = self._get_sentence_order_pair(
                    s1, s2
                )

                raw_sop_data.append(
                    {
                        "sentence_a": sentence_a,
                        "sentence_b": sentence_b,
                        "is_ordered": is_ordered,
                    }
                )
        return raw_sop_data

    def get(self):
        paragraphs = self._read_wiki()
        raw_data = self.build_dataset(paragraphs)
        return raw_data


# --- AlbertPretrainingDataset (Tokenization and Masking) ---


class AlbertPretrainingDataset(Dataset):
    def __init__(self, raw_ds, tokenizer):
        self.raw_ds = raw_ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.raw_ds)

    def __getitem__(self, idx):
        s1 = self.raw_ds[idx]["sentence_a"]
        s2 = self.raw_ds[idx]["sentence_b"]
        sop_label = int(self.raw_ds[idx]["is_ordered"])

        enc = self.tokenizer(
            s1,
            s2,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        )

        token_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = enc["token_type_ids"].squeeze(0)

        mlm_labels = token_ids.clone()

        for i in range(len(token_ids)):
            # Skip special tokens from masking
            if token_ids[i] in [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]:
                mlm_labels[i] = -100
                continue

            # Apply MLM (15% chance to mask)
            if random.random() < 0.15:
                prob = random.random()
                if prob < 0.8:
                    # 80% of 15% -> Mask token
                    token_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    # 10% of 15% -> Random token
                    token_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
                # 10% of 15% -> Keep token (mlm_labels[i] remains the original token ID)
            else:
                # 85% of the time, don't mask (set label to ignore)
                mlm_labels[i] = -100

        return token_ids, attention_mask, token_type_ids, sop_label, mlm_labels
