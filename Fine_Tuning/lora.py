"""
Benefits of Lora:
    1. Less parameters to train, if the weight vector is dxk then 
       the number of parameters to train or optimize is dxr but if 
       with lora we have only 2 low rank weight vectors which are 
       dxr and rxd hence the total numbers of parameters are (dxr)+(rxk) where r is very small.

    2. We can easily switch between fine tune models because the base model is completely same 
       as the pretrain model, we just need to load these lora adaptor.
"""
import torch
from torchvision import datasets,transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn.utils.parametrize as parametrize

device = "cuda" if torch.cuda.is_available() else "cpu"

_ = torch.manual_seed(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])

mnist_train = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
train_loader = DataLoader(mnist_train,batch_size=32,shuffle=True)

mnist_test = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
test_loader = DataLoader(mnist_test,batch_size=32)

class BigNet(nn.Module):
    def __init__(self, hidden_size=1000, hidden_size2=2000):
        super().__init__()
        self.linear1 = nn.Linear(28 * 28, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size2)
        self.linear3 = nn.Linear(hidden_size2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(img.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x

model = BigNet().to(device)

def train(train_loader, model, epochs=5):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=epoch_loss / (total / labels.size(0)), acc=100.0 * correct / total)

        print(f"Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Accuracy={100.0*correct/total:.2f}%")

train(train_loader, model, epochs=5)

original_weights = {}
for name,param in model.named_parameters():
    original_weights[name] = param.clone().detach()

def test():
    model.eval()
    correct = 0
    total = 0
    wrong_counts = [0 for _ in range(10)]

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            preds = output.argmax(dim=1)
            total += y.size(0)
            correct += (preds == y).sum().item()

            for idx in range(y.size(0)):
                if preds[idx] != y[idx]:
                    wrong_counts[y[idx].item()] += 1

    accuracy = 100.0 * correct / total
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("Wrong predictions per class:")
    for i in range(10):
        print(f"Class {i}: {wrong_counts[i]}")

test()

class LoRAParameterization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device="cpu"):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(rank, features_in, device=device))
        self.lora_B = nn.Parameter(torch.zeros(features_out, rank, device=device))
        nn.init.normal_(self.lora_A, mean=0.0, std=1.0)
        nn.init.zeros_(self.lora_B)
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if not self.enabled:
            return original_weights
        delta_W = torch.matmul(self.lora_B, self.lora_A)
        return original_weights + self.scale * delta_W

def linear_layer_parameterization(layer, device, rank=1, lora_alpha=1):
    out_features, in_features = layer.weight.shape
    return LoRAParameterization(features_in=in_features, features_out=out_features, rank=rank, alpha=lora_alpha, device=device)

parametrize.register_parametrization(model.linear1, "weight", linear_layer_parameterization(model.linear1, device, rank=4, lora_alpha=8))
parametrize.register_parametrization(model.linear2, "weight", linear_layer_parameterization(model.linear2, device, rank=4, lora_alpha=8))
parametrize.register_parametrization(model.linear3, "weight", linear_layer_parameterization(model.linear3, device, rank=4, lora_alpha=8))

def enable_disable_lora(enabled=True):
    for layer in [model.linear1, model.linear2, model.linear3]:
        layer.parametrizations.weight[0].enabled = enabled # type: ignore

for name, param in model.named_parameters():
    if "lora_" not in name:
        print(f"Freezing non-LoRA parameter {name}")
        param.requires_grad = False

def train_lora(train_loader, model, epochs=5, lr=1e-3):
    loss_fn = nn.CrossEntropyLoss()
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(lora_params, lr=lr)
    print(f"\nTraining {len(lora_params)} LoRA parameters")
    print(f"Total trainable params: {sum(p.numel() for p in lora_params):,}")
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        loop = tqdm(train_loader, desc=f"LoRA Epoch [{epoch+1}/{epochs}]")

        for images, labels in loop:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=epoch_loss / (total / labels.size(0)), acc=100.0 * correct / total)

        print(f"LoRA Epoch {epoch+1}: Loss={epoch_loss/len(train_loader):.4f}, Accuracy={100.0*correct/total:.2f}%")

train_lora(train_loader, model, epochs=5, lr=1e-3)

print("\n=== Testing with LoRA enabled ===")
test()

print("\n=== Testing with LoRA disabled (original weights) ===")
enable_disable_lora(enabled=False)
test()
enable_disable_lora(enabled=True)