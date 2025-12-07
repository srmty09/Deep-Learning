import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
import random
from torch.utils.data import Dataset


class Creating_BertPretraining_Dataset:
    def __init__(self, ds):
        self.ds = ds

    def _read_wiki(self):
        texts = []
        for i in range(len(self.ds)):
            raw = self.ds['text'][i].strip().lower()
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
        parts = paragraph.split(' . ')
        sentences = []
        for s in parts:
            s = s.strip()
            if not s:
                continue
            if not s.endswith('.'):
                s += '.'
            if s.endswith(' = =.') or s.endswith(' = = =.') or s.endswith(' =.') or len(s) <= 2:
                continue
            sentences.append(s)
        return sentences

    def _get_next_sentence(self, s1, s2, paragraphs):
        if random.random() < 0.5:
            return s1, s2, True
        else:
            s2 = random.choice(random.choice(paragraphs))
            return s1, s2, False

    def build_dataset(self, paragraphs):
        nsp_dataset = []
        for paragraph in paragraphs:
            for i in range(len(paragraph) - 1):
                s1 = paragraph[i]
                s2 = paragraph[i + 1]
                s1, s2, is_next = self._get_next_sentence(s1, s2, paragraphs)
                nsp_dataset.append({
                    "sentence_a": s1,
                    "sentence_b": s2,
                    "is_next": is_next
                })
        return nsp_dataset

    def get(self):
        paragraphs = self._read_wiki()
        dataset = self.build_dataset(paragraphs)
        return dataset

class BertDataset(Dataset):
    def __init__(self, ds, tokenizer):
        self.ds = ds
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s1 = self.ds[idx]['sentence_a']
        s2 = self.ds[idx]['sentence_b']
        to_next = int(self.ds[idx]['is_next'])

        enc = self.tokenizer(
            s1,
            s2,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt"
        )

        token_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        token_type_ids = enc["token_type_ids"].squeeze(0)

        mlm_labels = token_ids.clone()

        for i in range(len(token_ids)):
            if token_ids[i] in [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]:
                mlm_labels[i] = -100
                continue

            if random.random() < 0.15:
                prob = random.random()
                if prob < 0.8:
                    token_ids[i] = self.tokenizer.mask_token_id
                elif prob < 0.9:
                    token_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
            else:
                mlm_labels[i] = -100

        return token_ids, attention_mask, token_type_ids, to_next, mlm_labels