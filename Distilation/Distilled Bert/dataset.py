import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import BertTokenizer
import random


class TinyShakespeare(Dataset):
    def __init__(self, filepath, tokenizer, seq_len=512):
        super().__init__()
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        
        self.tokens = self.tokenizer.encode(text, add_special_tokens=False)
        self.targets = self.tokens.copy()
        self.index = []
        
        self.ProcessForMLM()
        
        for idx in range(len(self.targets)):
            if idx not in self.index:
                self.targets[idx] = -100
        
        self.inp_seq = []
        self.tar_seq = []
        
        for i in range(0, len(self.tokens), seq_len):
            inp_chunk = self.tokens[i:i+seq_len]
            tar_chunk = self.targets[i:i+seq_len]
            
            if len(inp_chunk) < seq_len:
                inp_chunk = inp_chunk + [0] * (seq_len - len(inp_chunk))  
                tar_chunk = tar_chunk + [-100] * (seq_len - len(tar_chunk)) 
            
            self.inp_seq.append(inp_chunk)
            self.tar_seq.append(tar_chunk)
    
    def __len__(self):
        return len(self.inp_seq)
    
    def __getitem__(self, index):
        return torch.tensor(self.inp_seq[index]), torch.tensor(self.tar_seq[index])
    
    def ProcessForMLM(self):
        for idx, t in enumerate(self.tokens):
            if random.random() <= 0.15:
                self.index.append(idx)
                rand = random.random()
                
                if rand < 0.8:
                    self.tokens[idx] = 103
                elif rand < 0.9:
                    self.tokens[idx] = random.randint(0, 30521)

filepath = ""
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
dataset = TinyShakespeare(filepath=filepath, tokenizer=tokenizer,seq_len = 8)