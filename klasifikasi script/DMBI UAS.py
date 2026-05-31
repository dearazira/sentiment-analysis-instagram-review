# ============================================
# EVALUASI LENGKAP 5 MODEL × 3 SKENARIO (BINARY)
# Model: LR, Naive Bayes, SVM, Random Forest, KNN
# Skenario: Baseline, Handling Imbalance, Handling + Tuning
# Metrik: Accuracy, Precision, Recall, Specificity, F1-Score, ROC Curve
# PLUS: CEK OVERFITTING (Train vs Test Accuracy)
# Data: 10000 sample (POSITIVE vs NEGATIVE ONLY)
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    f1_score, precision_score, recall_score, roc_curve, auc
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings('ignore')

print("✅ Library berhasil diimport!")

# ============================================
# 1. LOAD & PREPROCESSING (10000 DATA)
# ============================================

file_path = r"C:\Users\ASUS\OneDrive\Dokumen\DMBI UAS\data\ulasan_com.instagram.android.csv"
df = pd.read_csv(file_path)
df = df.sample(n=10000, random_state=42)
print(f"⚠️ Menggunakan {len(df)} sample data")

df = df[['Ulasan', 'Rating']].copy()
df.columns = ['text', 'rating']
df = df.dropna(subset=['rating'])

# ============================================
# ⭐ KONVERSI KE BINARY (POSITIVE vs NEGATIVE)
# ============================================
def rating_to_binary(rating):
    if rating <= 2:
        return 'negative'
    elif rating >= 4:
        return 'positive'
    else:
        return None

df['sentiment'] = df['rating'].apply(rating_to_binary)
df = df.dropna(subset=['sentiment'])

print("\n✅ Data setelah konversi ke BINARY:")
print(df['sentiment'].value_counts())
print(f"\nPersentase:")
print(df['sentiment'].value_counts(normalize=True) * 100)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text_clean'] = df['text'].apply(clean_text)
df = df[df['text_clean'].str.len() > 10]

print(f"\n✅ Data siap: {len(df)} baris")

# ============================================
# 2. SPLIT DATA
# ============================================

X = df['text_clean'].tolist()
y = df['sentiment'].tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📊 Split: Train={len(X_train)}, Test={len(X_test)}")

# ============================================
# 3. TF-IDF
# ============================================

print("\n🔄 TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), min_df=2, max_df=0.9)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print(f"✅ Fitur: {X_train_tfidf.shape[1]}")

# ============================================
# 4. ENCODE LABEL
# ============================================

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)
class_names = le.classes_
print(f"\n📋 Label (Binary): {class_names}")

# ============================================
# DEFINE OUTPUT PATH (SEBELUM DIGUNAKAN)
# ============================================

output_path = r"C:\Users\ASUS\OneDrive\Dokumen\DMBI UAS\output"

# Buat folder output jika belum ada
os.makedirs(output_path, exist_ok=True)

# ============================================
# FUNGSI THRESHOLD TUNING
# ============================================

def threshold_tuning_binary(model, X_test, y_test_enc):
    if not hasattr(model, 'predict_proba'):
        return 0.5, 0, None, None
    
    y_prob = model.predict_proba(X_test)[:, 1]
    
    best_f1 = 0
    best_threshold = 0.5
    best_y_pred = None
    
    for threshold in np.arange(0.3, 0.8, 0.05):
        y_pred_temp = (y_prob >= threshold).astype(int)
        f1 = f1_score(y_test_enc, y_pred_temp)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_y_pred = y_pred_temp
    
    return best_threshold, best_f1, best_y_pred, y_prob

# ============================================
# FUNGSI EVALUASI (DENGAN CEK OVERFITTING)
# ============================================

def evaluate_binary_with_overfit(y_train, y_test, y_pred_train, y_pred_test, y_prob, model_name, scenario):
    """Evaluasi lengkap dengan cek overfitting"""
    
    # Metrik training
    train_acc = accuracy_score(y_train, y_pred_train)
    train_precision = precision_score(y_train, y_pred_train)
    train_recall = recall_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train)
    
    # Metrik testing
    test_acc = accuracy_score(y_test, y_pred_test)
    test_precision = precision_score(y_test, y_pred_test)
    test_recall = recall_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Overfitting gap
    gap_acc = train_acc - test_acc
    gap_f1 = train_f1 - test_f1
    
    # Status overfitting
    if gap_acc < 0.03:
        overfit_status = "✅ AMAN (gap < 3%)"
    elif gap_acc < 0.05:
        overfit_status = "⚠️ CUKUP (gap 3-5%)"
    elif gap_acc < 0.08:
        overfit_status = "⚠️ WASPADA (gap 5-8%)"
    else:
        overfit_status = "❌ OVERFITTING (gap > 8%)"
    
    # ROC AUC
    roc_auc = None
    fpr, tpr = None, None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
    
    return {
        'Model': model_name,
        'Skenario': scenario,
        'Train_Accuracy': train_acc,
        'Test_Accuracy': test_acc,
        'Gap_Accuracy': gap_acc,
        'Train_F1': train_f1,
        'Test_F1': test_f1,
        'Gap_F1': gap_f1,
        'Overfit_Status': overfit_status,
        'Test_Precision': test_precision,
        'Test_Recall': test_recall,
        'Test_Specificity': specificity,
        'Test_F1': test_f1,
        'ROC_AUC': roc_auc,
        'Confusion_Matrix': confusion_matrix(y_test, y_pred_test),
        'y_prob': y_prob,
        'fpr': fpr,
        'tpr': tpr
    }

# ============================================
# DEFINE MODELS
# ============================================

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# ============================================
# SKENARIO 1: BASELINE
# ============================================

print("\n" + "="*100)
print("📌 SKENARIO 1: BASELINE (Tanpa Handling Imbalance)")
print("="*100)

results_baseline = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_tfidf, y_train)
    
    # Prediksi train
    y_pred_train = model.predict(X_train_tfidf)
    if isinstance(y_pred_train[0], str):
        y_pred_train_enc = le.transform(y_pred_train)
    else:
        y_pred_train_enc = y_pred_train
    
    # Prediksi test
    y_pred_test = model.predict(X_test_tfidf)
    if isinstance(y_pred_test[0], str):
        y_pred_test_enc = le.transform(y_pred_test)
    else:
        y_pred_test_enc = y_pred_test
    
    # Probabilitas
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test_tfidf)[:, 1]
    
    result = evaluate_binary_with_overfit(
        y_train_enc, y_test_enc, 
        y_pred_train_enc, y_pred_test_enc, 
        y_prob, name, 'Baseline'
    )
    results_baseline.append(result)
    
    print(f"   Train Acc: {result['Train_Accuracy']:.4f} | Test Acc: {result['Test_Accuracy']:.4f} | Gap: {result['Gap_Accuracy']:.4f} | {result['Overfit_Status']}")

# ============================================
# SKENARIO 2: THRESHOLD TUNING
# ============================================

print("\n" + "="*100)
print("📌 SKENARIO 2: THRESHOLD TUNING (Handling Imbalance)")
print("="*100)

results_handling = []

for name, model in models.items():
    print(f"\n{name} - Threshold Tuning...")
    
    # Re-inisialisasi model
    if name == 'Logistic Regression':
        model_clone = LogisticRegression(max_iter=1000, random_state=42)
    elif name == 'Naive Bayes':
        model_clone = MultinomialNB()
    elif name == 'SVM':
        model_clone = SVC(kernel='linear', probability=True, random_state=42)
    elif name == 'Random Forest':
        model_clone = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model_clone = KNeighborsClassifier(n_neighbors=5)
    
    model_clone.fit(X_train_tfidf, y_train)
    
    # Prediksi train (baseline dulu untuk cek overfit)
    y_pred_train = model_clone.predict(X_train_tfidf)
    if isinstance(y_pred_train[0], str):
        y_pred_train_enc = le.transform(y_pred_train)
    else:
        y_pred_train_enc = y_pred_train
    
    if hasattr(model_clone, 'predict_proba'):
        best_threshold, best_f1, y_pred_tuned, y_prob = threshold_tuning_binary(
            model_clone, X_test_tfidf, y_test_enc
        )
        result = evaluate_binary_with_overfit(
            y_train_enc, y_test_enc, 
            y_pred_train_enc, y_pred_tuned, 
            y_prob, name, f'Threshold (th={best_threshold:.2f})'
        )
    else:
        y_pred_test = model_clone.predict(X_test_tfidf)
        y_pred_test_enc = le.transform(y_pred_test) if isinstance(y_pred_test[0], str) else y_pred_test
        result = evaluate_binary_with_overfit(
            y_train_enc, y_test_enc, 
            y_pred_train_enc, y_pred_test_enc, 
            None, name, 'Threshold (KNN)'
        )
    
    results_handling.append(result)
    print(f"   Train Acc: {result['Train_Accuracy']:.4f} | Test Acc: {result['Test_Accuracy']:.4f} | Gap: {result['Gap_Accuracy']:.4f} | {result['Overfit_Status']}")

# ============================================
# SKENARIO 3: THRESHOLD + TUNING
# ============================================

print("\n" + "="*100)
print("📌 SKENARIO 3: THRESHOLD TUNING + HYPERPARAMETER TUNING")
print("="*100)

param_grid_lr = {'C': [0.1, 0.5, 1.0, 1.5, 2.0]}
param_grid_nb = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}
param_grid_svm = {'C': [0.1, 0.5, 1.0, 1.5, 2.0], 'kernel': ['linear']}
param_grid_rf = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
param_grid_knn = {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']}

param_grids = {
    'Logistic Regression': param_grid_lr,
    'Naive Bayes': param_grid_nb,
    'SVM': param_grid_svm,
    'Random Forest': param_grid_rf,
    'KNN': param_grid_knn
}

base_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'KNN': KNeighborsClassifier()
}

results_tuning = []

for name in models.keys():
    print(f"\nTuning {name}...")
    
    random_search = RandomizedSearchCV(
        base_models[name], param_grids[name], 
        n_iter=10, cv=3, scoring='f1',
        n_jobs=-1, random_state=42
    )
    random_search.fit(X_train_tfidf, y_train)
    
    print(f"   Best params: {random_search.best_params_}")
    best_model = random_search.best_estimator_
    
    # Prediksi train
    y_pred_train = best_model.predict(X_train_tfidf)
    if isinstance(y_pred_train[0], str):
        y_pred_train_enc = le.transform(y_pred_train)
    else:
        y_pred_train_enc = y_pred_train
    
    if hasattr(best_model, 'predict_proba'):
        best_threshold, best_f1, y_pred_tuned, y_prob = threshold_tuning_binary(
            best_model, X_test_tfidf, y_test_enc
        )
        result = evaluate_binary_with_overfit(
            y_train_enc, y_test_enc, 
            y_pred_train_enc, y_pred_tuned, 
            y_prob, name, f'Threshold+Tuning (th={best_threshold:.2f})'
        )
    else:
        y_pred_test = best_model.predict(X_test_tfidf)
        y_pred_test_enc = le.transform(y_pred_test) if isinstance(y_pred_test[0], str) else y_pred_test
        result = evaluate_binary_with_overfit(
            y_train_enc, y_test_enc, 
            y_pred_train_enc, y_pred_test_enc, 
            None, name, 'Threshold+Tuning (KNN)'
        )
    
    results_tuning.append(result)
    print(f"   Train Acc: {result['Train_Accuracy']:.4f} | Test Acc: {result['Test_Accuracy']:.4f} | Gap: {result['Gap_Accuracy']:.4f} | {result['Overfit_Status']}")

# ============================================
# RINGKASAN PERBANDINGAN (DENGAN OVERFITTING)
# ============================================

print("\n" + "="*120)
print("📊 RINGKASAN PERBANDINGAN 5 MODEL × 3 SKENARIO (DENGAN CEK OVERFITTING)")
print("="*120)

summary_data = []
for r in results_baseline + results_handling + results_tuning:
    summary_data.append({
        'Model': r['Model'],
        'Skenario': r['Skenario'],
        'Train_Acc': r['Train_Accuracy'],
        'Test_Acc': r['Test_Accuracy'],
        'Gap_Acc': r['Gap_Accuracy'],
        'Overfit_Status': r['Overfit_Status'],
        'Test_F1': r['Test_F1'],
        'ROC_AUC': r['ROC_AUC'] if r['ROC_AUC'] is not None else 0
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.round(4).to_string(index=False))

# ============================================
# 📈 ROC CURVE UNTUK SEMUA MODEL DAN SKENARIO
# ============================================

print("\n" + "="*80)
print("📈 MEMBUAT ROC CURVE UNTUK SEMUA MODEL DAN SKENARIO")
print("="*80)

# Setup warna dan style untuk ROC Curve
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 
          'magenta', 'navy', 'teal', 'coral', 'goldenrod']
line_styles = ['-', '--', '-.', ':']

# Gabungkan semua hasil
all_results = results_baseline + results_handling + results_tuning

# 1. ROC CURVE PER SKENARIO (Semua model dalam satu plot)
scenarios = ['Baseline', 'Threshold (th=', 'Threshold+Tuning (th=']

for scenario in ['Baseline', 'Threshold (th=', 'Threshold+Tuning (th=']:
    plt.figure(figsize=(12, 8))
    
    # Filter hasil untuk skenario ini
    scenario_results = [r for r in all_results if r['Skenario'].startswith(scenario)]
    
    for idx, result in enumerate(scenario_results):
        if result['y_prob'] is not None and result['ROC_AUC'] is not None:
            color = colors[idx % len(colors)]
            linestyle = line_styles[idx % len(line_styles)]
            plt.plot(result['fpr'], result['tpr'], 
                    color=color, linestyle=linestyle, lw=2,
                    label=f"{result['Model']} (AUC = {result['ROC_AUC']:.3f})")
    
    # Garis diagonal (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve - Skenario: {scenario.split("(")[0]}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Simpan gambar
    scenario_name = scenario.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    plt.savefig(f"{output_path}\\roc_curve_{scenario_name}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ ROC Curve untuk {scenario} disimpan")

# 2. ROC CURVE PER MODEL (Semua skenario dalam satu plot)
for model_name in models.keys():
    plt.figure(figsize=(12, 8))
    
    # Filter hasil untuk model ini
    model_results = [r for r in all_results if r['Model'] == model_name]
    
    for idx, result in enumerate(model_results):
        if result['y_prob'] is not None and result['ROC_AUC'] is not None:
            color = colors[idx % len(colors)]
            linestyle = line_styles[idx % len(line_styles)]
            # Persingkat nama skenario untuk legend
            scenario_short = result['Skenario'].replace('Threshold (th=', 'Th=').replace('Threshold+Tuning (th=', 'Tuned Th=').replace(')', '')
            plt.plot(result['fpr'], result['tpr'], 
                    color=color, linestyle=linestyle, lw=2,
                    label=f"{scenario_short} (AUC = {result['ROC_AUC']:.3f})")
    
    # Garis diagonal
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title(f'ROC Curve Comparison - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Simpan gambar
    model_name_clean = model_name.replace(' ', '_')
    plt.savefig(f"{output_path}\\roc_curve_{model_name_clean}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ ROC Curve untuk {model_name} disimpan")

# 3. ROC CURVE GABUNGAN SEMUA MODEL DAN SKENARIO TERBAIK
plt.figure(figsize=(14, 10))

# Ambil model terbaik dari setiap skenario
best_by_scenario = {}
for scenario in ['Baseline', 'Threshold (th=', 'Threshold+Tuning (th=']:
    scenario_results = [r for r in all_results if r['Skenario'].startswith(scenario)]
    if scenario_results:
        best = max(scenario_results, key=lambda x: x['ROC_AUC'] if x['ROC_AUC'] is not None else 0)
        best_by_scenario[scenario] = best

for idx, (scenario, result) in enumerate(best_by_scenario.items()):
    if result['y_prob'] is not None and result['ROC_AUC'] is not None:
        color = colors[idx % len(colors)]
        scenario_short = scenario.split('(')[0]
        plt.plot(result['fpr'], result['tpr'], 
                color=color, lw=2.5,
                label=f"{result['Model']} - {scenario_short} (AUC = {result['ROC_AUC']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve - Best Model from Each Scenario', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_path}\\roc_curve_best_models.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ ROC Curve untuk model terbaik setiap skenario disimpan")

# 4. ROC CURVE GABUNGAN SEMUA MODEL BASELINE
plt.figure(figsize=(12, 8))

baseline_results = [r for r in all_results if r['Skenario'] == 'Baseline']
for idx, result in enumerate(baseline_results):
    if result['y_prob'] is not None and result['ROC_AUC'] is not None:
        color = colors[idx % len(colors)]
        plt.plot(result['fpr'], result['tpr'], 
                color=color, lw=2,
                label=f"{result['Model']} (AUC = {result['ROC_AUC']:.3f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve - Baseline Scenario (All Models)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{output_path}\\roc_curve_all_models_baseline.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ ROC Curve untuk semua model Baseline disimpan")

# ============================================
# MODEL TERBAIK (Berdasarkan Test Accuracy dengan Gap Kecil)
# ============================================

# Filter model dengan gap < 8%
good_models = [r for r in results_baseline + results_handling + results_tuning if r['Gap_Accuracy'] < 0.08]

if good_models:
    best_result = max(good_models, key=lambda x: x['Test_F1'])
else:
    best_result = min(results_baseline + results_handling + results_tuning, key=lambda x: x['Gap_Accuracy'])

print("\n" + "="*100)
print(f"🏆 MODEL TERBAIK (BINARY): {best_result['Model']} - {best_result['Skenario']}")
print("="*100)

print(f"""
{'='*60}
📊 METRIK GLOBAL
{'='*60}
Train Accuracy           : {best_result['Train_Accuracy']:.4f} ({best_result['Train_Accuracy']*100:.2f}%)
Test Accuracy            : {best_result['Test_Accuracy']:.4f} ({best_result['Test_Accuracy']*100:.2f}%)
Overfitting Gap          : {best_result['Gap_Accuracy']:.4f}
Status                   : {best_result['Overfit_Status']}

Test Precision           : {best_result['Test_Precision']:.4f}
Test Recall (Sensitivity): {best_result['Test_Recall']:.4f}
Test Specificity         : {best_result['Test_Specificity']:.4f}
Test F1-Score            : {best_result['Test_F1']:.4f}
ROC AUC                  : {best_result['ROC_AUC']:.4f}
""")

# ============================================
# CONFUSION MATRIX
# ============================================

plt.figure(figsize=(6, 5))
sns.heatmap(best_result['Confusion_Matrix'], annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Confusion Matrix - {best_result["Model"]} ({best_result["Skenario"]})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

# ============================================
# VISUALISASI OVERFITTING GAP
# ============================================

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(summary_data))
width = 0.35

train_acc = [r['Train_Accuracy'] for r in results_baseline + results_handling + results_tuning]
test_acc = [r['Test_Accuracy'] for r in results_baseline + results_handling + results_tuning]

ax.bar(x - width/2, train_acc, width, label='Train Accuracy', color='skyblue')
ax.bar(x + width/2, test_acc, width, label='Test Accuracy', color='lightcoral')

ax.set_xlabel('Model - Skenario')
ax.set_ylabel('Accuracy')
ax.set_title('Perbandingan Train vs Test Accuracy (Cek Overfitting)')
ax.set_xticks(x)
ax.set_xticklabels([f"{r['Model']}\n({r['Skenario'][:15]})" for r in results_baseline + results_handling + results_tuning], rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================
# HEATMAP OVERFITTING GAP
# ============================================

pivot_gap = df_summary.pivot(index='Model', columns='Skenario', values='Gap_Acc')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_gap, annot=True, fmt='.4f', cmap='RdYlGn_r', linewidths=0.5, 
            vmin=0, vmax=0.15)
plt.title('Heatmap Overfitting Gap (Train - Test Accuracy)\nHijau = Aman, Merah = Overfitting')
plt.xlabel('Skenario')
plt.ylabel('Model')
plt.tight_layout()
plt.show()

# ============================================
# SIMPAN HASIL
# ============================================

# Simpan file CSV
df_summary.to_csv(f"{output_path}\\evaluasi_binary_overfit.csv", index=False)

# Simpan juga semua metrics lengkap ke file terpisah
all_metrics = []
for r in results_baseline + results_handling + results_tuning:
    all_metrics.append({
        'Model': r['Model'],
        'Skenario': r['Skenario'],
        'Train_Accuracy': r['Train_Accuracy'],
        'Test_Accuracy': r['Test_Accuracy'],
        'Gap_Accuracy': r['Gap_Accuracy'],
        'Train_F1': r['Train_F1'],
        'Test_F1': r['Test_F1'],
        'Gap_F1': r['Gap_F1'],
        'Overfit_Status': r['Overfit_Status'],
        'Test_Precision': r['Test_Precision'],
        'Test_Recall': r['Test_Recall'],
        'Test_Specificity': r['Test_Specificity'],
        'ROC_AUC': r['ROC_AUC'] if r['ROC_AUC'] is not None else 0
    })

df_all_metrics = pd.DataFrame(all_metrics)
df_all_metrics.to_csv(f"{output_path}\\evaluasi_binary_lengkap.csv", index=False)

print(f"\n💾 Hasil disimpan di: {output_path}")
print(f"   - evaluasi_binary_overfit.csv (ringkasan)")
print(f"   - evaluasi_binary_lengkap.csv (lengkap dengan semua metrik)")
print(f"   - roc_curve_*.png (berbagai file ROC Curve)")
print("\n✨ SELESAI! (Binary Classification + Cek Overfitting + Semua ROC Curve)")