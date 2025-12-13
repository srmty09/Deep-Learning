import datasets
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from safetensors.torch import save_file
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)


ds = load_dataset("stanfordnlp/imdb")


class sentiment_dataset(Dataset):
    def __init__(self, ds, tokenizer):
        self.ds = ds
        self.tokenizer = tokenizer

    def __len__(self):
        return self.ds.num_rows

    def __getitem__(self, idx):
        text = self.ds["text"][idx]
        target = torch.tensor(self.ds["label"][idx])

        encoding = self.tokenizer(
            text,
            max_length=128,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}

        return encoding, target


train_data = sentiment_dataset(ds["train"], tokenizer)
test_data = sentiment_dataset(ds["test"], tokenizer)


train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
epochs = 10
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)

for ep in range(epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    train_progress = tqdm(
        train_loader,
        desc=f"Epoch {
                          ep + 1}/{epochs} [TRAIN]",
    )

    for batch_idx, (texts, targets) in enumerate(train_progress):
        inputs = {key: val.to(device) for key, val in texts.items()}
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(**inputs)
        loss = loss_fn(outputs.logits, targets)

        loss.backward()
        optimizer.step()

        preds = outputs.logits.argmax(-1)
        train_total += targets.size(0)
        train_correct += (preds == targets).sum().item()
        train_accuracy = 100 * train_correct / train_total

        train_loss += loss.item()
        avg_train_loss = train_loss / (batch_idx + 1)

        train_progress.set_postfix(
            {"avg_loss": f"{avg_train_loss:.4f}", "accuracy": f"{train_accuracy:.2f}%"}
        )

    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    test_progress = tqdm(test_loader, desc=f"Epoch {ep + 1}/{epochs} [TEST]")

    with torch.no_grad():
        for batch_idx, (texts, targets) in enumerate(test_progress):
            inputs = {key: val.to(device) for key, val in texts.items()}
            targets = targets.to(device)

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, targets)

            preds = outputs.logits.argmax(-1)
            test_total += targets.size(0)
            test_correct += (preds == targets).sum().item()
            test_accuracy = 100 * test_correct / test_total

            test_loss += loss.item()
            avg_test_loss = test_loss / (batch_idx + 1)

            test_progress.set_postfix(
                {
                    "avg_loss": f"{avg_test_loss:.4f}",
                    "accuracy": f"{test_accuracy:.2f}%",
                }
            )

    print(f"\n{'=' * 60}")
    print(f"Epoch {ep + 1}/{epochs} Summary:")
    print(f"  Train - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")
    print(f"  Test  - Loss: {avg_test_loss:.4f}, Accuracy: {test_accuracy:.2f}%")
    print(f"{'=' * 60}\n")
