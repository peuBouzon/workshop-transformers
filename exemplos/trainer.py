import torch
import torch.nn as nn
from sklearn.metrics import classification_report, recall_score

class Trainer:
  def __init__(self, device, has_mask=True):
    self.device = device
    self.has_mask = has_mask
    self.criterion = nn.CrossEntropyLoss()

  def fit(self, model, learning_rate, max_epochs, weights, train_loader, val_loader, num_classes, int_to_label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    self.criterion = nn.CrossEntropyLoss(weight=weights)
    print("\nStarting training...")
    for epoch in range(max_epochs):
        train_loss = self._train_epoch(model, train_loader, optimizer, self.criterion)
        val_loss, val_accuracy, _, _ = self.evaluate(model, val_loader)
        print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Val. Recall: {val_accuracy*100:.2f}%")

    print("\nFinal Evaluation...")
    _, _, val_preds, val_labels = self.evaluate(model, val_loader)
    report = classification_report(
        val_labels,
        val_preds,
        target_names=[int_to_label[i] for i in range(num_classes)],
        zero_division=0
    )
    print(report)

  def _train_epoch(self, model, iterator, optimizer, criterion):
      model.train()
      epoch_loss = 0
      for batch in iterator:
        mask = None
        if self.has_mask:
            x, label, mask = batch['x'].to(self.device), batch['label'].to(self.device), batch['mask'].to(self.device) 
        else:
            x, label = batch['x'].to(self.device), batch['label'].to(self.device)

        optimizer.zero_grad()
        output = model(x, mask)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
      return epoch_loss / len(iterator)

  def evaluate(self, model, iterator):
      model.eval()
      epoch_loss = 0
      all_preds, all_labels = [], []
      with torch.no_grad():
          for batch in iterator:
            mask = None
            if self.has_mask:
                x, label, mask = batch['x'].to(self.device), batch['label'].to(self.device), batch['mask'].to(self.device) 
            else:
                x, label = batch['x'].to(self.device), batch['label'].to(self.device)
            output = model(x, mask)
            loss = self.criterion(output, label)
            epoch_loss += loss.item()
            preds = torch.argmax(output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
      recall = recall_score(all_labels, all_preds, average='macro')
      return epoch_loss / len(iterator), recall, all_preds, all_labels
