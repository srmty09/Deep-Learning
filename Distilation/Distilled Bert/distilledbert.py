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
StudentModel = BertForMaskedLM(student_config)

StudentModel = load_distilbert_weights(StudentModel,TeacherModel)

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

from tqdm import tqdm


class DistillationTrainer:
    def __init__(self,
                 TeacherModel,
                 StudentModel,
                 TrainLoader,
                 ValLoader,
                 Optimizer,
                 LossFn,
                 LearningRateScheduler,
                 device 
                 ):
        self.tm = TeacherModel
        self.sm = StudentModel
        self.train_loader = TrainLoader
        self.val_loader = ValLoader
        self.optim = Optimizer
        self.lossfn = LossFn
        self.LrScheduler = LearningRateScheduler
        self.device = device

    def train(self, epochs):
        self.tm = self.tm.to(self.device)
        self.sm = self.sm.to(self.device)
        self.tm.eval()
        
        for ep in range(epochs):
            self.sm.train()
            train_bar = tqdm(
                self.train_loader,
                desc=f"Epoch[{ep+1}/{epochs}] Training",
                leave=False
            )
            
            epoch_total_loss = 0
            epoch_cosine_loss = 0
            epoch_kld_loss = 0
            epoch_mlm_loss = 0
            
            for inp, tar in train_bar:
                inp = inp.to(self.device)
                tar = tar.to(self.device)

                self.optim.zero_grad()

                with torch.no_grad():
                    teacher_outputs = self.tm(inp, output_hidden_states=True)
                
                student_outputs = self.sm(inp, labels=tar, output_hidden_states=True)
                
                loss = self.lossfn(student_outputs, teacher_outputs)
                total_loss = loss['total_loss']
                cosine_loss = loss['cosine_loss']
                kld_loss = loss['kld_loss']
                student_mlm_loss = loss['student_mlm_loss']

                total_loss.backward()
                self.optim.step()
                
                if self.LrScheduler is not None:
                    self.LrScheduler.step()
                
                epoch_total_loss += total_loss.item()
                epoch_cosine_loss += cosine_loss
                epoch_kld_loss += kld_loss
                epoch_mlm_loss += student_mlm_loss

                train_bar.set_postfix({
                    'total': f"{total_loss.item():.4f}",
                    'cos': f"{cosine_loss:.4f}",
                    'kld': f"{kld_loss:.4f}",
                    'mlm': f"{student_mlm_loss:.4f}"
                })
            
            avg_total = epoch_total_loss / len(self.train_loader)
            avg_cos = epoch_cosine_loss / len(self.train_loader)
            avg_kld = epoch_kld_loss / len(self.train_loader)
            avg_mlm = epoch_mlm_loss / len(self.train_loader)
            
            val_loss = self.validate()
            
            print(f"\nEpoch {ep+1}/{epochs}")
            print(f"Train - Total: {avg_total:.4f} | Cosine: {avg_cos:.4f} | KLD: {avg_kld:.4f} | MLM: {avg_mlm:.4f}")
            print(f"Val Loss: {val_loss:.4f}\n")
    
    def validate(self):
        self.sm.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            val_bar = tqdm(self.val_loader, desc="Validation", leave=False)
            for inp, tar in val_bar:
                inp = inp.to(self.device)
                tar = tar.to(self.device)
                
                teacher_outputs = self.tm(inp, output_hidden_states=True)
                student_outputs = self.sm(inp, labels=tar, output_hidden_states=True)
                
                loss = self.lossfn(student_outputs, teacher_outputs)
                total_val_loss += loss['total_loss'].item()
                
                val_bar.set_postfix({'loss': f"{loss['total_loss'].item():.4f}"})
        
        return total_val_loss / len(self.val_loader)
    

Optimizer = torch.optim.AdamW(
    StudentModel.parameters(), 
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
num_epochs = 10
total_steps = len(train_loader) * num_epochs

from torch.optim.lr_scheduler import LinearLR

warmup_steps = int(0.1 * total_steps)
LrScheduler = LinearLR(
    Optimizer,
    start_factor=0.1,
    end_factor=1.0,
    total_iters=warmup_steps
)


trainer = DistillationTrainer(
    TeacherModel=TeacherModel,
    StudentModel=StudentModel,
    TrainLoader=train_loader,
    ValLoader=val_loader,
    Optimizer=Optimizer,
    LossFn=DistillationLosses,
    LearningRateScheduler=LrScheduler,
    device=device
)
trainer.train(epochs=num_epochs)