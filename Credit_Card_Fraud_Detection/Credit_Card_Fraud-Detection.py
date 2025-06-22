# === ULTIMATE FRAUD DETECTION WITH ENHANCED VISUALIZATIONS ===
import pandas as pd
import numpy as np
import time
import psutil
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, average_precision_score,
                           roc_curve, precision_recall_curve)
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

# === Benchmarking Decorator ===
def benchmark(stage_name):
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            start_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            
            print(f"\n⏳ Starting {stage_name}...")
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_mem = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
            
            elapsed = end_time - start_time
            mem_used = end_mem - start_mem
            
            benchmark.results[stage_name] = {
                'time_sec': round(elapsed, 2),
                'memory_mb': round(mem_used, 2)
            }
            
            print(f"✅ Completed {stage_name} in {elapsed:.2f}s (ΔMemory: {mem_used:+.2f}MB)")
            return result
        return wrapper
    return decorator
benchmark.results = {}

# === Enhanced Visualization Functions ===
def plot_enhanced_confusion_matrix(y_true, y_pred, model_name):
    """Create a visually enhanced confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                annot_kws={'size': 16, 'weight': 'bold'},
                cbar=False, linewidths=0.5, linecolor='gray')
    
    plt.title(f'{model_name} Confusion Matrix\n', fontsize=14, pad=20)
    plt.xlabel('\nPredicted Label', fontsize=12)
    plt.ylabel('True Label\n', fontsize=12)
    
    plt.xticks([0.5, 1.5], ['Genuine (0)', 'Fraud (1)'], rotation=0)
    plt.yticks([0.5, 1.5], ['Genuine (0)', 'Fraud (1)'], rotation=0)
    
    accuracy = np.trace(cm) / np.sum(cm)
    precision = cm[1,1] / (cm[1,1] + cm[0,1])
    recall = cm[1,1] / (cm[1,1] + cm[1,0])
    
    plt.figtext(0.5, -0.1, 
                f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}",
                ha='center', fontsize=11, bbox=dict(facecolor='lightgray', alpha=0.5))
    plt.tight_layout()
    plt.show()

def plot_roc_pr_curves(y_true, y_proba, model_name):
    """Plot combined ROC and Precision-Recall curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    ax1.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title(f'{model_name} - ROC Curve', fontsize=14)
    ax1.legend(loc="lower right")
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    ax2.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AUC = {pr_auc:.2f})')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title(f'{model_name} - Precision-Recall Curve', fontsize=14)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()

def plot_enhanced_feature_importance(model, feature_names, top_n=15):
    """Create an interactive-style feature importance plot"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(indices)), importances[indices], color='darkorange', align='center')
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                va='center', ha='left', fontsize=10)
    
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=12)
    plt.title(f'Top {top_n} Feature Importances\n', fontsize=16, pad=20)
    plt.xlabel('\nRelative Importance', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    
    ax = plt.gca()
    ax.set_facecolor('#f5f5f5')
    
    plt.tight_layout()
    plt.show()

# === 1. Optimized Data Loading ===
@benchmark("Data Loading")
def load_data(path):
    df = pd.read_csv(path)
    print(f"\n📊 Dataset Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print("🔍 Class Distribution:")
    print(df['Class'].value_counts(normalize=True).apply(lambda x: f"{x:.4%}"))
    return df

df = load_data("D:\\Coding\\CODSOFT\\Credit_Card_Fraud_Detection\\Dataset_CreditCardFraudDetection\\creditcard.csv")

# === 2. Fast Feature Engineering ===
@benchmark("Feature Engineering")
def engineer_features(df):
    initial_cols = df.shape[1]
    
    df['Amount_scaled'] = RobustScaler().fit_transform(df[['Amount']])
    df['hour_of_day'] = (df['Time'] % (24*3600)) // 3600
    df['V1_V2'] = df['V1'] * df['V2']
    df['V3_V4'] = df['V3'] * df['V4']
    df = df.drop(['Time', 'Amount'], axis=1)
    
    print(f"➕ Added {df.shape[1] - initial_cols} new features")
    print(f"📉 Reduced to {df.shape[1]} total features")
    return df

df = engineer_features(df)

# === 3. Balanced Data Split ===
@benchmark("Data Splitting")
def split_data(df):
    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n✂️ Split Ratio: {len(X_train):,} train / {len(X_test):,} test samples")
    print("📌 Class Balance in Test Set:")
    print(y_test.value_counts(normalize=True).apply(lambda x: f"{x:.4%}"))
    return X, X_train, X_test, y_train, y_test

X, X_train, X_test, y_train, y_test = split_data(df)

# === 4. Smart Sampling ===
@benchmark("Data Resampling")
def resample_data(X_train, y_train):
    before_count = y_train.value_counts()
    smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    print("\n⚖️ Before Resampling:")
    print(before_count.apply(lambda x: f"{x:,}"))
    print("\n⚖️ After Resampling:")
    print(pd.Series(y_res).value_counts().apply(lambda x: f"{x:,}"))
    return X_res, y_res

X_train_res, y_train_res = resample_data(X_train, y_train)

# === 5. Focused Model Training ===
models = {
    "XGBoost (Fast)": XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=100,
        n_jobs=-1,
        random_state=42,
        tree_method='hist'
    ),
    "LightGBM (Fast)": LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        boosting_type='goss'
    )
}

# === 6. Enhanced Model Evaluation ===
results = {}
for name, model in models.items():
    @benchmark(f"{name} Training")
    def train_model(model, X, y):
        model.fit(X, y)
        return model
    
    print(f"\n🏗️ Training {name} Model...")
    models[name] = train_model(model, X_train_res, y_train_res)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n📊 {name} Performance:")
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        'ROC-AUC': roc_auc_score(y_test, y_proba),
        'PR-AUC': average_precision_score(y_test, y_proba)
    }
    
    plot_enhanced_confusion_matrix(y_test, y_pred, name)
    plot_roc_pr_curves(y_test, y_proba, name)

# === 7. Final Optimized Ensemble ===
@benchmark("Final Ensemble Training")
def train_ensemble(X, y):
    ensemble = XGBClassifier(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        n_jobs=-1,
        random_state=42
    )
    ensemble.fit(X, y)
    return ensemble

print("\n🚀 Creating optimized ensemble...")
ensemble = train_ensemble(X_train_res, y_train_res)

# Final evaluation
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print("\n🏆 FINAL ENSEMBLE PERFORMANCE:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"PR-AUC: {average_precision_score(y_test, y_proba):.4f}")

plot_enhanced_confusion_matrix(y_test, y_pred, "Final Ensemble")
plot_roc_pr_curves(y_test, y_proba, "Final Ensemble")
plot_enhanced_feature_importance(ensemble, X.columns)

# === 8. Save Model ===
@benchmark("Model Saving")
def save_model(model, path):
    joblib.dump(model, path)

save_model(ensemble, 'optimized_fraud_model.pkl')
print("\n💾 Production model saved as 'optimized_fraud_model.pkl'")

# === Performance Benchmark Report ===
print("\n📊 PRECISE PERFORMANCE BENCHMARKS:")
print("┌──────────────────────────────┬────────────┬─────────────┐")
print("│ Stage                        │ Time (sec) │ Memory (MB) │")
print("├──────────────────────────────┼────────────┼─────────────┤")

max_stage_len = max(len(stage) for stage in benchmark.results)
for stage, metrics in benchmark.results.items():
    stage_padded = stage.ljust(max_stage_len)
    print(f"│ {stage_padded} │ {metrics['time_sec']:>10.2f} │ {metrics['memory_mb']:>11.2f} │")

print("└──────────────────────────────┴────────────┴─────────────┘")

total_time = sum(m['time_sec'] for m in benchmark.results.values())
peak_mem = max(m['memory_mb'] for m in benchmark.results.values())

print(f"\n⏱️ Total Execution Time: {total_time:.2f} seconds")
print(f"💾 Peak Memory Usage: {peak_mem:.2f} MB")
print(f"🏆 Best Model ROC-AUC: {results[max(results, key=lambda x: results[x]['ROC-AUC'])]['ROC-AUC']:.4f}")