"""
Teacher Model: Resnet18
Student Model: Custom model with very less dense
DataSet: CIFAR10

"""


import torch
import torch.nn as nn
import torchvision
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor()
])


dataset = datasets.CIFAR10("./data",train=True,download=True,transform=transform)


model = resnet18()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)


train_loader = DataLoader(dataset,batch_size=64)
eps = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)
device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    def __init__(self, loader, model, optimizer, loss_fn, eps, device):
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.eps = eps
        self.device = device
        print(f"training on device: {self.device}")

    def __call__(self):
        self.model.train()
        self.model.to(self.device)

        for ep in range(self.eps):
            ep_loss = 0.0
            correct = 0
            total = 0

            for img, label in self.loader:
                img = img.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()
                y_pred = self.model(img)

                loss = self.loss_fn(y_pred, label)
                loss.backward()
                self.optimizer.step()

                ep_loss += loss.item()

                preds = torch.argmax(y_pred, dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

            avg_loss = ep_loss / len(self.loader)
            acc = correct / total

            print(f"ep: {ep+1}/{self.eps}  loss: {avg_loss:.4f}  acc: {acc:.4f}")




train_model = Trainer(train_loader,model,optimizer,loss_fn,10,device)


train_model()


class SmallStudent224(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def kldLoss(TeacherLogits, StudentLogits, Temp):
    return F.kl_div(
        F.log_softmax(StudentLogits / Temp, dim=1),
        F.softmax(TeacherLogits / Temp, dim=1),
        reduction="batchmean"
    ) * (Temp * Temp)

def ceLoss(StudentLogits, labels):
    return F.cross_entropy(StudentLogits, labels)

def loss_fn(TeacherLogits, StudentLogits, Temp, labels, alpha):
    kd_loss = kldLoss(TeacherLogits, StudentLogits, Temp)
    ce_loss = ceLoss(StudentLogits, labels)
    return alpha * kd_loss + (1 - alpha) * ce_loss


class Distillation:
    def __init__(
        self,
        TeacherModel,
        StudentModel,
        loader,
        optimizer,
        loss_fn,
        eps,
        device,
        Temp,
        alpha
    ):
        self.TeacherModel = TeacherModel
        self.StudentModel = StudentModel
        self.loader = loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.eps = eps
        self.device = device
        self.Temp = Temp
        self.alpha = alpha
        print(f"training on device: {self.device}")

    def __call__(self):
        self.TeacherModel.eval()
        self.StudentModel.train()

        self.TeacherModel.to(self.device)
        self.StudentModel.to(self.device)

        for ep in range(self.eps):
            ep_loss = 0.0
            correct = 0
            total = 0

            for img, label in self.loader:
                img = img.to(self.device)
                label = label.to(self.device)

                self.optimizer.zero_grad()

                with torch.no_grad():
                    TeacherLogits = self.TeacherModel(img)

                StudentLogits = self.StudentModel(img)

                loss = self.loss_fn(
                    TeacherLogits,
                    StudentLogits,
                    self.Temp,
                    label,
                    self.alpha
                )

                loss.backward()
                self.optimizer.step()

                ep_loss += loss.item()

                preds = torch.argmax(StudentLogits, dim=1)
                correct += (preds == label).sum().item()
                total += label.size(0)

            avg_loss = ep_loss / len(self.loader)
            acc = correct / total

            print(
                f"ep: {ep+1}/{self.eps}  "
                f"loss: {avg_loss:.4f}  "
                f"acc: {acc:.4f}"
            )


TeacherModel = model
StudentModel = SmallStudent224().to(device)

for p in TeacherModel.parameters():
    p.requires_grad = False

optimizerfordis = optim.Adam(StudentModel.parameters(), lr=1e-4)

Temp = 4
alpha = 0.4

distillation = Distillation(
    TeacherModel=TeacherModel,
    StudentModel=StudentModel,
    loader=train_loader,
    optimizer=optimizerfordis,  
    loss_fn=loss_fn,
    eps=eps,
    device=device,
    Temp=Temp,
    alpha=alpha
)

distillation()
