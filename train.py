import os
import random
import numpy as np
import torch
import pandas as pd
import json
import joblib

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from download_data import download_titanic_data
from preprocessing import TitanicPreprocessor
from model import TitanicMLP

# Fixing Seeds for all relevant libraries to ensure accurate recovery
def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def main():
    set_seed(42)
    
    # Secure token loading 
    token_path = os.path.expanduser('~/.kaggle/access_token')
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            # Read Kaggle token from local hidden file and load it into runtime environment only
            os.environ["KAGGLE_API_TOKEN"] = f.read().strip()
    
    # load data
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(BASE_DIR, 'data')
    download_titanic_data(data_dir)
    df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    
    # K-Folds definitions and results lists
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_accuracies = []
    fold_aucs = []
    fold_f1s = []
    
    # Tracking the best fold by AUC
    best_fold_idx = -1
    best_fold_auc = -1.0
    best_fold_state = None
    best_fold_preprocessor = None
    
    print(f"Starting {n_splits}-Fold Stratified Cross-Validation...")
    
    # Folds loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Survived'])):
        print(f"\n--- Training Fold {fold + 1}/{n_splits} ---")
        
        # Dividing the data according to the indexes of the current fold
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Data preprocessing (fit_transform is done only on the current fold's train to prevent leakage)
        pre = TitanicPreprocessor()
        X_train = pre.fit_transform(train_df)
        y_train = pre.get_target(train_df)
        
        X_val = pre.transform(val_df)
        y_val = pre.get_target(val_df)
        
        # Convert to Tensors
        X_tr_t = torch.tensor(X_train, dtype=torch.float32)
        y_tr_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        
        train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
        
        # Build model and handle class imbalance with pos_weight
        model = TitanicMLP(input_dim=X_tr_t.shape[1])
        
        num_neg = (y_tr_t == 0).sum().item()
        num_pos = (y_tr_t == 1).sum().item()
        # Gives more importance to the small department (survived)
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        # Early Stopping
        best_val_loss = float('inf')
        best_model_state = None
        patience = 30 # Number of epochs allowed without improvement before stopping training
        patience_counter = 0
        
        # Training loop for the current fold
        # Iterates over epochs and mini-batches, performing forward pass, loss computation, backpropagation, and parameter updates
        epochs = 200
        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
            # Evaluate model on validation set 
            model.eval() 
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()
            
            # Check if current model is the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1  # no improvement, increase counter
            
            if patience_counter >= patience: # early stopping
                break
                
        # Load best model and generate validation predictions
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            
        model.eval()
        with torch.no_grad():
            final_outputs = model(X_val_t)
            final_probs = torch.sigmoid(final_outputs).numpy()
            final_preds = (final_probs >= 0.5).astype(np.float32)
            
        # Calculating metrics for the current fold
        acc = accuracy_score(y_val_t.numpy(), final_preds)
        auc = roc_auc_score(y_val_t.numpy(), final_probs)
        f1 = f1_score(
            y_val_t.numpy(),
            final_preds
        )
        
        print(
            f"Fold {fold + 1} Results -> "
            f"Val Acc: {acc:.4f} | "
            f"ROC-AUC: {auc:.4f} | "
            f"F1: {f1:.4f}"
        )
                
        fold_accuracies.append(acc)
        fold_aucs.append(auc)
        fold_f1s.append(f1)
        
        # Keeping the weights and preprocessor if this is the best fold so far
        if auc > best_fold_auc:
            best_fold_auc = auc
            best_fold_idx = fold
            best_fold_state = {
                k: v.clone()
                for k, v in model.state_dict().items()
            }
            best_fold_preprocessor = pre

    # Print cross-validation performance summary
    print("\n========================================")
    print("Final Cross-Validation Summary:")
    print(f"Mean Validation Accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
    print(f"Mean Validation ROC-AUC: {np.mean(fold_aucs):.4f} (+/- {np.std(fold_aucs):.4f})")
    print(f"Mean Validation F1: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")
    print("========================================")
    print("Done! Representative model saved successfully.")

    # Save best model weights
    torch.save(best_fold_state, "titanic_mlp_model.pth")

    # Save best fold metadata
    with open("best_fold.json", "w") as f:
        json.dump(
            {
                "best_fold": best_fold_idx,
                "best_auc": best_fold_auc
            },
            f
        )

    # Save preprocessing 
    joblib.dump(best_fold_preprocessor, "titanic_preprocessor.pkl")

    print(
        f"\nSaved model from best fold: Fold {best_fold_idx + 1} "
        f"(ROC-AUC: {best_fold_auc:.4f})"
    )

if __name__ == "__main__":
    main()