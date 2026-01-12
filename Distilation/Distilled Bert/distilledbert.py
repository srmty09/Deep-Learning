import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from bert import BertForMaskedLM, BertConfig
from transformers import BertTokenizer,BertForMaskedLM,BertConfig
import os
from dataset import TinyShakespeare


device = "cuda" if torch.cuda.is_available() else "cpu"

#tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# load dataset
dataset = TinyShakespeare(filepath="/home/smruti/Desktop/git repos/Deep-Learning/Distilation/Distilled Bert/input.txt",tokenizer=tokenizer,seq_len=8)

print(f"total size of dataset{len(dataset)}")



TeacherConfig = BertConfig()
TeacherConfig.attention_probs_dropout_prob = 0.4

TeacherModel = BertForMaskedLM.from_pretrained(
    "bert-base-uncased",
    config=TeacherConfig
)

for param in TeacherModel.parameters():
    param.requires_grad = False

for param in TeacherModel.cls.predictions.parameters():
    param.requires_grad = True

# fine tune the teacher model



def load_distilbert_weights(student_model, teacher_model, student_num_layer=6):
    teacher_state = teacher_model.state_dict()
    student_state = student_model.state_dict()
    
    for key in student_state.keys():
        if 'embeddings' in key and key in teacher_state:
            student_state[key] = teacher_state[key]
    
    for student_layer_idx in range(student_num_layer):
        teacher_layer_idx = student_layer_idx * 2

        for key in student_state.keys():
            if f'encoder.layer.{student_layer_idx}.' in key:
                teacher_key = key.replace(
                    f'encoder.layer.{student_layer_idx}.',
                    f'encoder.layer.{teacher_layer_idx}.'
                )
                if teacher_key in teacher_state:
                    student_state[key] = teacher_state[teacher_key]
    
    for key in student_state.keys():
        if key.startswith('cls.predictions') and key in teacher_state:
            student_state[key] = teacher_state[key]
    
    student_model.load_state_dict(student_state)
    return student_model

student_config = BertConfig.from_pretrained("bert-base-uncased")
student_config.num_hidden_layers = 6
student_model = BertForMaskedLM(student_config)