import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score
from imblearn.metrics import specificity_score

class Trainer:
	def __init__(self, device, save_name, num_classes, weights, has_mask=True):
		self.device = device
		self.has_mask = has_mask
		self.criterion = nn.CrossEntropyLoss(weight=weights) if num_classes > 2 else nn.BCEWithLogitsLoss(weight=weights)
		self.save_name = save_name
		self.is_multiclass = True if num_classes > 2 else False

	def fit(self, model, learning_rate, max_epochs, weights, train_loader, val_loader, num_classes, int_to_label, threshold=0.5):
		self.is_multiclass = True if num_classes > 2 else False
		optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
		self.criterion = nn.CrossEntropyLoss(weight=weights) if num_classes > 2 else nn.BCEWithLogitsLoss(weight=weights)
		print("\nStarting training...")
		self.min_loss = float('inf')
		for epoch in range(max_epochs):
			train_loss = self._train_epoch(model, train_loader, optimizer, self.criterion)
			val_loss, val_recall, val_precision, val_fpr, _, _ = self.evaluate(model, val_loader, threshold)
			if val_loss < self.min_loss:
				self.min_loss = val_loss
				torch.save(model.state_dict(), f'{self.save_name}')
			print(f"Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {val_loss:.3f} | Val. Recall: {val_recall*100:.2f}% | Val. Precision: {val_precision*100:.2f}% | Val. FPR: {val_fpr*100:.2f}%")

		_, _, _, _, val_preds, val_labels = self.evaluate(model, val_loader, threshold)

		print('Carregando o melhor modelo salvo...')
		model.load_state_dict(torch.load(self.save_name))
		report = classification_report(
			val_labels,
			val_preds > threshold if not self.is_multiclass else val_preds,
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
			output = output.squeeze(-1) if not self.is_multiclass else output
			loss = criterion(output, label.float() if not self.is_multiclass else label)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
		return epoch_loss / len(iterator)

	def evaluate(self, model, iterator, threshold=0.5):
		if threshold != 0.5 and self.is_multiclass:
			print("Aviso: threshold é ignorado em classificação multiclasse.\n")

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
				output = output.squeeze(-1) if not self.is_multiclass else output
				loss = self.criterion(output, label.float() if not self.is_multiclass else label)
				epoch_loss += loss.item()
				preds = torch.argmax(output, dim=1) if self.is_multiclass else torch.sigmoid(output)
				all_preds.extend(preds.cpu().numpy())
				all_labels.extend(label.cpu().numpy())
		all_preds = np.array(all_preds)
		all_labels = np.array(all_labels)

		preds = all_preds > threshold if not self.is_multiclass else all_preds
		recall = recall_score(all_labels, preds, average='macro' if self.is_multiclass else 'binary', zero_division=0)
		precision = precision_score(all_labels, preds, average='macro' if self.is_multiclass else 'binary', zero_division=0)
		fpr = 1 - specificity_score(all_labels, preds, average='macro' if self.is_multiclass else 'binary')

		return epoch_loss / len(iterator), recall, precision, fpr, all_preds, all_labels
