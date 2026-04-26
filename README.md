# Phishing Email Detection with BERT & CySecBERT

A project comparing two transformer-based models — **BERT-base-uncased** and **CySecBERT** — for binary phishing email classification. The project covers full fine-tuning, evaluation, error analysis, LIME explainability, cross-dataset generalization testing, and threshold tuning.


## Project Structure

```
├── phishing-model.ipynb     # Main training notebook — fine-tuning, evaluation, LIME

├── CEAS_08_dataset
    ├── model_test.ipynb     # Cross-dataset generalization & threshold analysis
    └── phishing_email.csv   # Dataset used in model_test.ipynb
├── utils.py                 # Shared EmailDataset class and evaluate_model helper
├── saved_models/
│   ├── bert/                # Fine-tuned BERT-base-uncased weights & tokenizer
│   └── cysec/               # Fine-tuned CySecBERT weights & tokenizer
├── results/                 # Training checkpoints (auto-generated)
├── logs/                    # Training logs (auto-generated)
├── requirements.txt
└── README.md
```

## Models

| Model | Description |
|---|---|
| `bert-base-uncased` | General-purpose bidirectional transformer, used as a baseline |
| `markusbayer/CySecBERT` | BERT variant pre-trained on cybersecurity text (phishing, malware, attack reports) |

Both models are fine-tuned for binary classification: **Safe (0)** vs **Phishing (1)**.


## Datasets

**Training dataset:** [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) from Hugging Face

**Cross-dataset test:** `phishing_email.csv` — an independent dataset of ~80,000 Enron-style corporate emails used in `model_test.ipynb` to evaluate generalization.


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/phishing-email-detection.git
cd phishing-email-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify GPU availability (optional but recommended for training)

```python
import torch
print(torch.cuda.is_available())  # Should print True if a CUDA GPU is available
```

> Training on CPU is possible but very slow. Using Google Colab with a T4 GPU is a good free alternative.


## How to Use

### Training the models (`phishing-model.ipynb`)

1. Open `phishing-model.ipynb` in Jupyter or Google Colab.
2. Run all cells from top to bottom.
3. The notebook will:
   - Download the Hugging Face dataset automatically
   - Preprocess and split the data
   - Fine-tune both BERT and CySecBERT for 3 epochs
   - Evaluate on the test set using Accuracy, F1, Precision, Recall, and ROC-AUC
   - Run error analysis (false positives / false negatives)
   - Generate LIME explanations for misclassified samples
   - Save fine-tuned models to `./saved_models/bert` and `./saved_models/cysec`

### Cross-dataset generalization testing (`model_test.ipynb`)

1. Make sure training has been completed and saved models exist in `./saved_models/`.
2. Place `phishing_email.csv` in the project root directory.
3. Open and run `model_test.ipynb`.
4. The notebook will:
   - Load the saved models and tokenizers
   - Evaluate both models zero-shot on the new dataset
   - Compare metrics and visualize confusion matrices and ROC curves
   - Run error analysis and LIME explanations on FP/FN samples
   - Tune the decision threshold using Youden's J statistic

### Output files generated

| File | Description |
|---|---|
| `confusion_matrices.png` | Confusion matrices for both models |
| `roc_curves.png` | ROC curves with AUC scores |
| `metrics_comparison.png` | Bar chart comparing all metrics |
| `lime_aggregated.png` | Aggregated LIME feature importance for FP/FN |
| `threshold_comparison_cm.png` | Confusion matrices at default vs optimal threshold |


## Results Summary

### In-distribution test set (training notebook)

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|---|
| BERT-base-uncased | 0.9798 | 0.9730 | 0.9725 | 0.9735 | 0.9977 |
| CySecBERT | 0.9874 | 0.9832 | 0.9827 | 0.9837 | 0.9986 |

CySecBERT outperformed BERT across all metrics (+1.02pp F1, +0.09pp ROC-AUC) on the in-distribution test set.

### Cross-dataset generalization (model_test.ipynb)

Both models retained strong discriminative ability (ROC-AUC > 0.93) on the unseen dataset, but showed accuracy drops due to distributional shift. This was largely resolved through threshold tuning using Youden's J statistic, which reduced false positives by ~70%.


## Key Dependencies

- `transformers` — model loading, tokenization, and Trainer API
- `torch` — model training and inference
- `datasets` — Hugging Face dataset loading
- `scikit-learn` — metrics, TF-IDF analysis, threshold tuning
- `lime` — local interpretability explanations
- `evaluate` — metric computation during training
- `matplotlib` / `seaborn` — visualizations

See `requirements.txt` for the full list with pinned versions.


## Reproducibility

All experiments use a fixed random seed of `42` across Python, NumPy, and PyTorch (including CUDA). Deterministic algorithms are enforced via:

```python
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
```


## Notes

- The `utils.py` file must be present in the project root (or parent directory when running `model_test.ipynb`). It contains the shared `EmailDataset` class and `evaluate_model` function used by both notebooks.
- `model_test.ipynb` uses `sys.path.append('..')` to import from `utils.py`, assuming the notebook is located one level below the root. Adjust the path if your folder structure differs.


## Authors

> Hugo Juraj Bulinský, Marek Homiak, Marko Olejník, Adrián Polák

