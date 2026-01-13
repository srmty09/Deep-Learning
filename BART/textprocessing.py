import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import re
import random
from transformers import BartTokenizer
import numpy as np

from datasets import load_dataset

# dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir="/home/smruti/Desktop/git repos/Deep-Learning/BART/")

# with open("/home/smruti/Desktop/git repos/Deep-Learning/BART/wikitext.txt", "w") as f:
#     for item in dataset['train']:
#         f.write(item['text'] + '\n')  # type: ignore

# print("WikiText downloaded successfully!")

with open("/home/smruti/Desktop/git repos/Deep-Learning/BART/wikitext.txt", "r") as f:
    document = f.read()

class TextPreprocessing:
    def __init__(self, document: str):
        self.document = document
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    
    def get_paragraphs(self):
        self.paragraphs = []
        for para in self.document.splitlines():
            para = para.strip()
            if para and not para.startswith('='):
                self.paragraphs.append(para)
    
    def get_sentence(self):
        self.sentences = []
        for para in self.paragraphs:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            para_sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
            self.sentences.append(para_sentences)

    def sentence_permutation(self, para_sentence, iterations=5):
        """Randomly permute sentences in a paragraph"""
        if len(para_sentence) < 2:
            return
            
        for _ in range(iterations):
            n = len(para_sentence)
            
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            while i == j:
                j = random.randint(0, n - 1)
            
            para_sentence[i], para_sentence[j] = para_sentence[j], para_sentence[i]
    
    def tokenize_sentences(self, para_sentences):
        """Tokenize each sentence separately with space preservation"""
        tokenized = []
        for i, sentence in enumerate(para_sentences):
            if i > 0:
                sentence = " " + sentence
            tokens = self.tokenizer.encode(sentence, add_special_tokens=False)
            tokenized.append(tokens)
        return tokenized
    
    def text_infilling(self, token_ids, mask_ratio=0.3):
        """Replace random spans of tokens with a single MASK token"""
        token_ids_copy = token_ids.copy()
        n = len(token_ids_copy)
        
        if n == 0:
            return token_ids_copy
        
        num_tokens_to_mask = max(1, int(n * mask_ratio))
        
        num_spans = max(1, num_tokens_to_mask // 3)  
        span_lengths = np.random.poisson(3, num_spans)
        span_lengths = np.clip(span_lengths, 1, 10)
        
        total = sum(span_lengths)
        if total > num_tokens_to_mask:
            span_lengths = [max(1, int(length * num_tokens_to_mask / total)) for length in span_lengths]
        
        mask_token_id = self.tokenizer.mask_token_id
        masked_indices = set()
        spans = []
        
        for span_len in span_lengths:
            attempts = 0
            while attempts < 50:  
                if len(masked_indices) >= n:
                    break
                start_pos = random.randint(0, max(0, n - span_len))
                overlap = any(i in masked_indices for i in range(start_pos, min(start_pos + span_len, n)))
                if not overlap:
                    spans.append((start_pos, start_pos + span_len))
                    for i in range(start_pos, min(start_pos + span_len, n)):
                        masked_indices.add(i)
                    break
                attempts += 1
        
        spans.sort()
        
        result = []
        last_end = 0
        
        for start, end in spans:
            result.extend(token_ids_copy[last_end:start])
            result.append(mask_token_id)
            last_end = end
        
        result.extend(token_ids_copy[last_end:])
        
        return result
    
    def prepare_data(self, para_sentences, apply_corruption=True, mask_ratio=0.3, 
                     apply_sentence_permutation=True):
        """
        Prepare data for BART training with optional corruption
        
        Args:
            para_sentences: List of sentences in a paragraph
            apply_corruption: Whether to apply text infilling
            mask_ratio: Ratio of tokens to mask (for text infilling)
            apply_sentence_permutation: Whether to shuffle sentences
        """
        import copy
        
        tokenized_original = self.tokenize_sentences(para_sentences)
        original_tokens = [tok for sent in tokenized_original for tok in sent]
        
        if apply_sentence_permutation and len(para_sentences) > 1:
            para_sentences_shuffled = copy.deepcopy(para_sentences)
            self.sentence_permutation(para_sentences_shuffled, iterations=5)
            
            tokenized_shuffled = self.tokenize_sentences(para_sentences_shuffled)
            encoder_tokens = [tok for sent in tokenized_shuffled for tok in sent]
        else:
            encoder_tokens = original_tokens.copy()
        
        if apply_corruption:
            corrupted_tokens = self.text_infilling(encoder_tokens, mask_ratio)
        else:
            corrupted_tokens = encoder_tokens
        
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        
        encoder_input = [bos_token_id] + corrupted_tokens + [eos_token_id]
        decoder_input = [bos_token_id] + original_tokens
        decoder_target = original_tokens + [eos_token_id]
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'decoder_target': decoder_target
        }


class BartDataset(Dataset):
    def __init__(self, document: str, apply_corruption=True, mask_ratio=0.3, 
                 apply_sentence_permutation=True):
        """
        BART Dataset with text infilling and sentence permutation
        
        Args:
            document: Input text document
            apply_corruption: Whether to apply text infilling
            mask_ratio: Ratio of tokens to mask
            apply_sentence_permutation: Whether to shuffle sentences
        """
        self.preprocessor = TextPreprocessing(document)
        self.preprocessor.get_paragraphs()
        self.preprocessor.get_sentence()
        self.apply_corruption = apply_corruption
        self.mask_ratio = mask_ratio
        self.apply_sentence_permutation = apply_sentence_permutation
        
    def __len__(self):
        return len(self.preprocessor.sentences)
    
    def __getitem__(self, idx):
        para_sentences = self.preprocessor.sentences[idx]
        data = self.preprocessor.prepare_data(
            para_sentences, 
            self.apply_corruption, 
            self.mask_ratio,
            self.apply_sentence_permutation
        )
        
        return {
            'encoder_input': torch.tensor(data['encoder_input'], dtype=torch.long),
            'decoder_input': torch.tensor(data['decoder_input'], dtype=torch.long),
            'decoder_target': torch.tensor(data['decoder_target'], dtype=torch.long)
        }


def collate_fn(batch):
    """Collate function to pad sequences in a batch"""
    pad_token_id = 1
    
    encoder_inputs = [item['encoder_input'] for item in batch]
    decoder_inputs = [item['decoder_input'] for item in batch]
    decoder_targets = [item['decoder_target'] for item in batch]
    
    encoder_inputs_padded = torch.nn.utils.rnn.pad_sequence(
        encoder_inputs, batch_first=True, padding_value=pad_token_id
    )
    decoder_inputs_padded = torch.nn.utils.rnn.pad_sequence(
        decoder_inputs, batch_first=True, padding_value=pad_token_id
    )
    decoder_targets_padded = torch.nn.utils.rnn.pad_sequence(
        decoder_targets, batch_first=True, padding_value=pad_token_id
    )
    
    return {
        'encoder_input': encoder_inputs_padded,
        'decoder_input': decoder_inputs_padded,
        'decoder_target': decoder_targets_padded
    }
    

dataset = BartDataset(
    document, 
    apply_corruption=True,         
    mask_ratio=0.3,                 
    apply_sentence_permutation=True 
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

for batch in dataloader:
    print("Encoder Input Shape:", batch['encoder_input'].shape)
    print("Decoder Input Shape:", batch['decoder_input'].shape)
    print("Decoder Target Shape:", batch['decoder_target'].shape)
    print("\nEncoder Input (first sample):")
    print(tokenizer.decode(batch['encoder_input'][3]))
    print("\nDecoder Input (first sample):")
    print(tokenizer.decode(batch['decoder_input'][3]))
    print("\nDecoder Target (first sample):")
    print(tokenizer.decode(batch['decoder_target'][3]))
    break