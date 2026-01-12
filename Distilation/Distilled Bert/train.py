import torch
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device):
        self.model = model.to(device)
        self.op = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

    def train(self, epochs=10):
        for ep in range(epochs):
            self.model.train()
            epoch_loss = 0.0

            train_bar = tqdm(
                self.train_loader,
                desc=f"Epoch [{ep+1}/{epochs}] Training",
                leave=False
            )

            for inp, tar in train_bar:
                inp = inp.to(self.device)
                tar = tar.to(self.device)

                self.op.zero_grad()

                outputs = self.model(input_ids=inp, labels=tar)
                loss = outputs.loss

                loss.backward()
                self.op.step()

                epoch_loss += loss.item()

                train_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            avg_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch [{ep+1}/{epochs}] Train Loss: {avg_loss:.4f}")

            self.validate(ep, epochs)

    def validate(self, ep, epochs):
        self.model.eval()
        val_loss = 0.0

        val_bar = tqdm(
            self.val_loader,
            desc=f"Epoch [{ep+1}/{epochs}] Validation",
            leave=False
        )

        with torch.no_grad():
            for inp, tar in val_bar:
                inp = inp.to(self.device)
                tar = tar.to(self.device)

                outputs = self.model(input_ids=inp, labels=tar)
                loss = outputs.loss

                val_loss += loss.item()

                val_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
