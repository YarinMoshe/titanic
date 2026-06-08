import os
import numpy as np
import pandas as pd
import torch
import json
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve, classification_report

from preprocessing import TitanicPreprocessor
from model import TitanicMLP

# Configuring the page in Streamlit
st.set_page_config(page_title="Titanic DL Dashboard", layout="wide", initial_sidebar_state="expanded")

st.title("🚢 Titanic Deep Learning - Evaluation & Inference")
st.markdown("---")

# Load data and return a stratified validation split with preprocessing
@st.cache_data  # cache results to avoid reloading and recomputing each time
def get_validation_data(data_path, fold_to_load=0):
    if not os.path.exists(data_path):
        return None, None, None, None
    
    df = pd.read_csv(data_path)

    # create identical stratified folds as in training
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # select the requested fold for train/validation split
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['Survived'])):
        if fold == fold_to_load:
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()
            break
            
    # Preprocessor calculation on the Train and start Validation 
    preprocessor = TitanicPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    
    X_val = preprocessor.transform(val_df)
    y_val = preprocessor.get_target(val_df)
    
    return X_val, y_val, preprocessor, X_train.shape[1]

# Creating tabs in the interface
tab1, tab2 = st.tabs(["📊 Validation Dashboard", "🔮 Custom Inference Interface"])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, 'data', 'train.csv')
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, 'titanic_mlp_model.pth')

# Tab 1: Performance evaluation dashboard on the validation set
with tab1:
    # Reading the best fold index stored by train.py
    best_fold = 0
    fold_json = os.path.join(BASE_DIR, "best_fold.json")
    if os.path.exists(fold_json):
        with open(fold_json) as f:
            best_fold = json.load(f).get("best_fold", 0)

    st.header(f"Held-Out Validation Set Evaluation (Best Fold: Fold {best_fold + 1})")
    st.write("This tab evaluates the saved model weights against the reproducible validation split from training.")
    
    if os.path.exists(DEFAULT_DATA_DIR) and os.path.exists(DEFAULT_MODEL_PATH):
        
        # Loading the data and weights
        X_val, y_val, preprocessor, input_dim = get_validation_data(DEFAULT_DATA_DIR, best_fold)
        
        # Loading the model
        model = TitanicMLP(input_dim=input_dim)
        model.load_state_dict(torch.load(DEFAULT_MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        # Inference and predictions
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        with torch.no_grad():
            outputs = model(X_val_t)
            probs = torch.sigmoid(outputs).numpy()
            preds = (probs >= 0.5).astype(np.float32)
            
        
        acc = accuracy_score(y_val, preds)
        
        # Test: ROC-AUC does not work if there is only one class
        has_multiple_classes = len(np.unique(y_val)) > 1
        if has_multiple_classes:
            auc = roc_auc_score(y_val, probs)
            auc_display = f"{auc:.4f}"
        else:
            auc_display = "N/A (Single class in data)"
        
        # Displaying metrics 
        st.markdown("### Detailed Classification Report")
        report_dict = classification_report(y_val, preds, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.4f}"))
        
        col1, col2 = st.columns(2)
        col1.metric(label="Validation Accuracy", value=f"{acc:.4%}")
        col2.metric(label="Validation ROC-AUC Score", value=auc_display)
        
        st.info(f"💡 **Methodological Note:** The performance metrics displayed above are evaluated on the validation split of Fold {best_fold + 1}, which was also utilized for the Early Stopping criteria during training (Consequently, these validation results may be slightly optimistic).")
        st.markdown("---")
        
        st.markdown("### Visualization Plots")
        plot_col1, plot_col2 = st.columns(2)
        
        # Confusion Matrix
        with plot_col1:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_val, preds)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=['Died (0)', 'Survived (1)'],
                        yticklabels=['Died (0)', 'Survived (1)'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            st.pyplot(fig)
            
        # ROC Curve
        with plot_col2:
            st.subheader("Receiver Operating Characteristic (ROC)")
            if has_multiple_classes:
                fpr, tpr, _ = roc_curve(y_val, probs)
                fig, ax = plt.subplots(figsize=(5, 4))
                ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
                ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                ax.set_xlim([0.0, 1.0])
                ax.set_ylim([0.0, 1.05])
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend(loc="lower right")
                st.pyplot(fig)
            else:
                st.warning("⚠️ Cannot generate ROC Curve: The data slice contains only a single class, making ROC calculations mathematically undefined.")
            
        
    else:
        st.error(f"Could not automatically load the default files. Please ensure files exist at:\n"
                 f"Data: {DEFAULT_DATA_DIR}\nModel: {DEFAULT_MODEL_PATH}")



# Tab 2: Custom prediction interface for new files from a path 
with tab2:
    st.header("Run Inference on Custom Dataset")
    st.write("Provide the paths below to load a dataset from disk and run predictions using your trained neural network.")
    
    # Input fields for manually entering paths
    csv_path_input = st.text_input("Enter absolute path to the dataset CSV file:", value=DEFAULT_DATA_DIR)
    model_path_input = st.text_input("Enter absolute path to the trained model weights (.pth):", value=DEFAULT_MODEL_PATH)
    
    if st.button("Run Model Inference"):
        # Verify that the files actually exist in the entered path
        if not os.path.exists(csv_path_input):
            st.error(f"Dataset file not found at: {csv_path_input}")
        elif not os.path.exists(model_path_input):
            st.error(f"Model file not found at: {model_path_input}")
        else:
            with st.spinner("Loading assets and running forward pass..."):
                try:
                    # Loading the inference file from the provided path
                    inference_df = pd.read_csv(csv_path_input)

                    # Loading the trained preprocessor saved during training (titanic_preprocessor.pkl)
                    preprocessor_path = os.path.join(BASE_DIR, "titanic_preprocessor.pkl")
                    if not os.path.exists(preprocessor_path):
                        st.error(
                            "Fitted preprocessor not found (titanic_preprocessor.pkl). "
                            "Please run train.py first to generate it alongside the model weights."
                        )
                        st.stop()

                    trained_preprocessor = joblib.load(preprocessor_path)
                    input_dim = len(trained_preprocessor.feature_columns)
                    X_infer = trained_preprocessor.transform(inference_df)

                    # Loading the net and weights from the provided path
                    model = TitanicMLP(input_dim=input_dim)
                    model.load_state_dict(torch.load(model_path_input, map_location=torch.device('cpu')))
                    model.eval()
                    
                    # Inference and predictions
                    X_infer_t = torch.tensor(X_infer, dtype=torch.float32)
                    with torch.no_grad():
                        outputs = model(X_infer_t)
                        probs = torch.sigmoid(outputs).numpy()
                        preds = (probs >= 0.5).astype(np.float32)
                    
                    st.success("Inference completed successfully!")
                    
                    # Test: ROC-AUC cannot be computed if there are fewer than 2 classes
                    if "Survived" in inference_df.columns:
                        st.info("Ground-truth labels found! Displaying evaluation metrics for this dataset...")
                        y_true = trained_preprocessor.get_target(inference_df)
                        
                        infer_acc = accuracy_score(y_true, preds)
                        
                        has_multiple_classes_infer = len(np.unique(y_true)) > 1
                        if has_multiple_classes_infer:
                            infer_auc = roc_auc_score(y_true, probs)
                            infer_auc_display = f"{infer_auc:.4f}"
                        else:
                            infer_auc_display = "N/A (Single class in uploaded file)"
                        
                        st.markdown("### Detailed Classification Report")
                        report_dict = classification_report(y_true, preds, output_dict=True)
                        report_df = pd.DataFrame(report_dict).transpose()
                        st.dataframe(report_df.style.format("{:.4f}"))
                        
                        inf_col1, inf_col2 = st.columns(2)
                        inf_col1.metric("Dataset Accuracy", f"{infer_acc:.4%}")
                        inf_col2.metric("Dataset ROC-AUC", infer_auc_display)
                        
                        # Creating graphs for the new inference file
                        inf_p1, inf_p2 = st.columns(2)
                        with inf_p1:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            sns.heatmap(confusion_matrix(y_true, preds), annot=True, fmt='d', cmap='Greens', ax=ax,
                                        xticklabels=['Died (0)', 'Survived (1)'], yticklabels=['Died (0)', 'Survived (1)'])
                            ax.set_ylabel('True Label (Actual)')
                            ax.set_xlabel('Predicted Label')
                            ax.set_title("Inference Confusion Matrix")
                            st.pyplot(fig)
                        with inf_p2:
                            if has_multiple_classes_infer:
                                fig, ax = plt.subplots(figsize=(4, 3))
                                fpr_i, tpr_i, _ = roc_curve(y_true, probs)
                                ax.plot(fpr_i, tpr_i, color='green', lw=2, label=f'AUC = {infer_auc:.4f}')
                                ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
                                ax.set_title("Inference ROC Curve")
                                ax.legend()
                                st.pyplot(fig)
                            else:
                                st.warning("⚠️ ROC Curve omitted: Uploaded file contains only one target class.")
                    
                    # The final results table with the probabilities
                    st.markdown("### Prediction Results Table")
                    results_df = inference_df.copy()
                    results_df["Predicted_Probability"] = probs
                    results_df["Prediction_Class"] = preds
                    results_df["Prediction_Label"] = results_df["Prediction_Class"].map({0.0: "Died", 1.0: "Survived"})
                    
                    # Displaying the first 15 lines as a sample in the interface
                    st.write("Previewing first 15 predictions:")
                    st.dataframe(results_df[["PassengerId", "Name", "Pclass", "Sex", "Predicted_Probability", "Prediction_Label"]].head(15))
                    
                    # Option to download the full CSV file with the forecasts
                    csv_buffer = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Predictions CSV",
                        data=csv_buffer,
                        file_name="titanic_predictions_output.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"An error occurred during inference pipeline: {str(e)}")