# 🚢 Titanic Survival Prediction — PyTorch + Streamlit

An end-to-end **binary classification pipeline** that predicts Titanic passenger survival.
The project fetches the dataset from Kaggle, performs EDA, applies leakage-safe
preprocessing, trains a neural network in PyTorch with Stratified K-Fold
cross-validation, and serves evaluation + inference through a Streamlit app.

---

## 📑 Table of Contents

- [Project Structure](#-project-structure)
- [Setup](#-setup)
- [Installation](#-installation)
- [Kaggle Credentials](#-kaggle-credentials)
- [Running the Pipeline](#-running-the-pipeline)
- [Using the Streamlit App](#-using-the-streamlit-app)
- [Architecture & Design Choices](#-architecture--design-choices)
- [Results](#-results)

---

## 📂 Project Structure

```
.
├── data/                       # Dataset folder (Titanic train.csv is downloaded/placed here)
├── download_data.py            # Securely fetches the dataset from Kaggle via the Kaggle API
├── preprocessing.py            # TitanicPreprocessor: feature engineering + scaling (leakage-safe)
├── model.py                    # TitanicMLP: the PyTorch model definition
├── train.py                    # Training script (Stratified K-Fold CV + early stopping)
├── app.py                      # Streamlit app (evaluation dashboard + inference UI)
├── EDA.ipynb                   # Exploratory Data Analysis notebook (data exploration + plots)
├── best_fold.json              # Best fold index + its ROC-AUC (keeps the app in sync)
├── titanic_mlp_model.pth       # Trained weights of the best fold
├── titanic_preprocessor.pkl    # Fitted preprocessor for the best fold (used at inference)
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

**Artifacts produced by `train.py`** (saved next to the script):

- `titanic_mlp_model.pth` — trained weights of the best fold
- `best_fold.json` — which fold was saved + its ROC-AUC (keeps the app in sync)
- `titanic_preprocessor.pkl` — the fitted preprocessor for that fold (used at inference)

---

## 🛠 Setup

**Requirements:**

- Python 3.10 – 3.12
- A Kaggle account (to download the dataset)

It is recommended to use a virtual environment:

```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

---

## 📦 Installation

```
git clone https://github.com/YarinMoshe/titanic.git
cd titanic
pip install -r requirements.txt
```

---

## 🔑 Kaggle Credentials

`download_data.py` fetches the dataset using the Kaggle API and **never hardcodes secrets**.
You need to provide your own credentials in one of the following ways:

1. **`kaggle.json` (recommended)**

  - Log in to Kaggle → *Account* → *Create New API Token*. This downloads `kaggle.json`.
  - Place it at:
    * Windows: `C:\Users\<you>\.kaggle\kaggle.json`
    * macOS / Linux: `~/.kaggle/kaggle.json`
  - On macOS/Linux, restrict permissions: `chmod 600 ~/.kaggle/kaggle.json`

2. **Environment variables**

```
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
```

> **Note:** You must also accept the competition rules once on the [Titanic competition page](https://www.kaggle.com/competitions/titanic/data),
> otherwise the API download will be rejected.

If you prefer not to use the API, simply download `train.csv` manually from the
competition page and place it in the `data/` folder. The pipeline will detect it
and skip the download.

---

## 🚀 Running the Pipeline

### 1. Train the model

```
python train.py
```

This will:

- Download the dataset (if not already in `data/`),
- Run 5-fold Stratified Cross-Validation,
- Print per-fold and mean metrics,
- Save the best fold's weights, fold index, and fitted preprocessor to disk.

### 2. Launch the app

```
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.
Press `Ctrl+C` in the terminal to stop it.

> Run `train.py` **before** the app, since the app loads the artifacts it produces.

---

## 🖥 Using the Streamlit App

### Tab 1 — Validation Dashboard

Automatically evaluates the saved model on the reproducible validation split of the **best fold**, displaying:

- Accuracy and ROC-AUC metrics,
- Confusion matrix,
- ROC curve,
- A full per-class classification report.

### Tab 2 — Custom Inference

1. Enter the absolute path to a dataset CSV.
2. Enter the path to the trained weights (`.pth`).
3. Click **Run Model Inference**.

The app loads the saved preprocessor, transforms the data, runs the forward pass,
and shows predictions. If the CSV contains a `Survived` column, it also displays
evaluation metrics and plots. You can download the full predictions as a CSV.

---

## 🧠 Architecture & Design Choices

### Model (`model.py`)

A compact multi-layer perceptron suited to the small, tabular Titanic dataset:

```
Input → Linear(→64) → LayerNorm → Mish → Dropout(0.3)
      → Linear(→32) → LayerNorm → Mish → Dropout(0.2)
      → Linear(→1)  → logit
```

- **Mish activation** — a smooth, non-monotonic activation that tends to train
more stably than ReLU on small datasets.
- **LayerNorm** — stabilises training and is batch-size independent, which suits
small batches better than BatchNorm.
- **Dropout** — regularisation to combat overfitting given the limited sample size.
- The final layer outputs a single **logit** (no sigmoid), paired with
`BCEWithLogitsLoss` for numerical stability.

### Training (`train.py`)

- **5-Fold Stratified Cross-Validation** preserves the survival ratio in every fold
and gives a robust estimate of generalisation rather than a single lucky split.
- **Class imbalance** is handled with `pos_weight` in `BCEWithLogitsLoss`,
computed from the train fold's class counts.
- **Optimizer:** Adam (`lr=0.002`, `weight_decay=1e-4`).
- **Early stopping** on validation loss (`patience=30`) restores the best epoch's
weights for each fold.
- The **best fold by ROC-AUC** is persisted as the representative model, together
with its fitted preprocessor, so inference uses exactly the statistics the model
was trained with.

### Preprocessing (`preprocessing.py`)

Feature engineering and transforms, all fit on the training fold only:

- **Title** extracted from passenger names, with rare titles grouped.
- **FamilySize** and **IsAlone** derived from `SibSp` + `Parch`.
- **FarePerPerson** and **IsChild** as additional engineered features.
- **Missing values** imputed using train-fold statistics (Fare median, Embarked
mode, Age median per title).
- **Log transform** applied to skewed fare features.
- **StandardScaler** on continuous features; **one-hot encoding** on categoricals,
with a fixed column schema so inference data always matches the training layout.
- `Cabin`, `Ticket`, `Name`, and `PassengerId` are dropped from the feature matrix.

---

## 📊 Results

5-Fold Stratified Cross-Validation (mean ± std):

| Metric   | Score           |
| -------- | --------------- |
| Accuracy | 0.8125 ± 0.0237 |
| ROC-AUC  | 0.8677 ± 0.0085 |
| F1       | 0.7575 ± 0.0357 |

> Numbers may vary slightly across machines/library versions despite fixed seeds.
> The dashboard metrics are evaluated on the best fold's validation split, which was
> also used for early stopping, so they may be marginally optimistic relative to the
> cross-validation mean reported above.
