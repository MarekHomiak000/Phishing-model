# utils.py
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, roc_auc_score, classification_report
)

# function to convert data to tensors, packs them into dict, adds labels and make them approachable through index (Trainer can work with this)
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(texts), padding=True, truncation=True,
                                   max_length=128, return_tensors='pt')
        self.labels = torch.tensor(labels.values)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# evaluate function
def evaluate_model(trainer, dataset, model_name):
    predictions = trainer.predict(dataset)

    logits = predictions.predictions
    probs  = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds  = np.argmax(logits, axis=-1)
    labels = predictions.label_ids

    results = {
        'accuracy':  accuracy_score(labels, preds),
        'f1':        f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall':    recall_score(labels, preds),
        'roc_auc':   roc_auc_score(labels, probs[:, 1])
    }

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"F1-score:  {results['f1']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"\nDetailed report:")
    print(classification_report(labels, preds,
      target_names=['Safe', 'Phishing']))

    return results
    