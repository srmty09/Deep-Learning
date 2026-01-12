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

# loss: KLD loss, hard loss for the student(mlm loss),cosine similarilty loss for each layer of the student and teacher

def KLD_loss(student_logits, teacher_logits, temperature=4.0):
    """
    Docstring for KLD_loss
    
    :param student_logits: Student's predicted logits
    :param teacher_logits: Teacher's predicted logits
    :param temperature: Scaling factor to make the teacher logits soft target for the student
    """

    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)  

    kld = F.kl_div(student_probs, teacher_probs, reduction="batchmean", log_target=False)  

    kld_loss = kld * (temperature ** 2)

    return kld_loss


def computeCosineLoss(student_hidden_states, teacher_hidden_states):
    """
    Docstring for computeCosineLoss
    
    :param student_hidden_states: All hidden states of the student model
    :param teacher_hidden_states: All hidden states of the teacher model
    """
    cosine_losses = []
    student_state_list = list(student_hidden_states)
    teacher_state_list = list(teacher_hidden_states)

    num_student_state = len(student_state_list)
    num_teacher_state = len(teacher_state_list)
    
    student_emb = student_state_list[0]
    teacher_emb = teacher_state_list[0]

    if student_emb.shape == teacher_emb.shape:
        emb_cosine_loss = 1 - F.cosine_similarity(student_emb.flatten(1), teacher_emb.flatten(1), dim=1).mean()
        cosine_losses.append(emb_cosine_loss)

    for student_idx in range(1, num_student_state):
        teacher_idx = student_idx * 2 - 1  
        
        if teacher_idx < num_teacher_state:
            student_layer = student_state_list[student_idx]
            teacher_layer = teacher_state_list[teacher_idx]
            
            if student_layer.shape == teacher_layer.shape:
                layer_cosine_loss = 1 - F.cosine_similarity(
                    student_layer.flatten(1), 
                    teacher_layer.flatten(1), 
                    dim=1
                ).mean()
                cosine_losses.append(layer_cosine_loss)
    
    total_cosine_loss = sum(cosine_losses) / len(cosine_losses) if cosine_losses else torch.tensor(0.0)  
    return total_cosine_loss


def DistillationLosses(student_output, teacher_output,
                       alpha=0.5, beta=0.3, gamma=0.2, temperature=4.0):
    """
    Docstring for DistillationLosses
    
    :param student_output: Output of the student(which includes logits,loss and hidden_states)
    :param teacher_output: Output of the teacher(which includes logits,hidden_states)
    :param alpha: Weight for cosine loss
    :param beta: Weight for KLD loss
    :param gamma: Weight for student MLM loss
    :param temperature: Temperature for distillation
    """

    student_logits = student_output['logits']
    student_hidden_layers = student_output['hidden_states']
    student_mlm_loss = student_output['loss'] 

    teacher_logits = teacher_output['logits']
    teacher_hidden_layers = teacher_output['hidden_states']

    kld_loss = KLD_loss(student_logits, teacher_logits, temperature)
    cosine_loss = computeCosineLoss(student_hidden_layers, teacher_hidden_layers)
    
    total_loss = alpha * cosine_loss + beta * kld_loss + gamma * student_mlm_loss

    return {
        'total_loss': total_loss,
        'cosine_loss': cosine_loss.item() if isinstance(cosine_loss, torch.Tensor) else cosine_loss,  
        'kld_loss': kld_loss.item(),
        'student_mlm_loss': student_mlm_loss.item()
    }