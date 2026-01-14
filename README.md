# 🧬 Breast Cancer Diagnosis Mini Project

Predict malignant vs benign tumors on the **Breast Cancer Wisconsin (Diagnostic)** dataset using supervised machine learning (baseline Logistic Regression vs. tuned SVM), with clear steps for data prep, modeling, and evaluation.

---

## 1. Introduction
Breast cancer remains a leading cause of death among women. Early, accurate diagnosis can save lives. This mini project explores bioinformatics techniques with supervised ML on the Wisconsin Breast Cancer (Diagnostic) dataset, using geometric properties of cell nuclei (radius, texture, concavity, etc.) to classify tumors as **malignant** or **benign**.

## 1.1 Problem Background
Manual reading of Fine Needle Aspirate (FNA) images is labor-intensive and prone to human fatigue. Automated systems can process large amounts of data and uncover subtle patterns, offering clinicians a decision-support tool that complements traditional diagnostics.

## 1.2 Problem Statement
Build an ML model that predicts malignancy/benignity from 10 geometric features of digitized cell nuclei. Start with a Logistic Regression baseline, then close the accuracy gap using a tuned Support Vector Machine (SVM) with feature scaling and hyperparameter search (GridSearch).

## 1.3 Objectives
- Use **Logistic Regression** as a baseline for binary breast cancer classification.
- Train an **SVM** with feature scaling and **GridSearch** hyperparameter tuning.
- Compare models via **Accuracy, Precision, Recall, F1-score**, and discuss results.

## 1.4 Scope
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) – 569 samples, 30 numeric features (Kaggle / scikit-learn built-in).
- **Focus:** Classification and identifying informative geometric features as biomarkers.
- **Tools:** Python, pandas, scikit-learn, matplotlib, seaborn; runnable in Jupyter/Colab or locally.
- **Versioning:** Code/notebooks intended for GitHub with regular commits/branches.

## 1.5 Conclusion (Goal)
Demonstrate the impact of model optimization in bioinformatics by contrasting a simple baseline with a tuned model, aiming for reliable malignant/benign predictions that assist in clinical decision-making.

---

## 2. Project Structure
```
Breast Cancer Classification/
├─ breast_cancer_model_comparison.py    # Model training and comparison script
├─ breast_cancer_data.csv               # Dataset (place in root directory)
├─ README.md                            # This document
└─ requirements.txt                     # Python dependencies
```

---

## 3. Setup & Installation

### 3.1 Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### 3.2 Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3.3 Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt

# Or install manually:
pip install streamlit pandas scikit-learn matplotlib seaborn numpy
```

Install with Jupyter for notebook development (optional):
```bash
pip install jupyter notebook
```

### 3.4 Prepare Dataset
Download the **Breast Cancer Wisconsin (Diagnostic)** dataset:

**Option 1: Using scikit-learn (automatic)**
```python
from sklearn.datasets import load_breast_cancer
import pandas as pd

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target
df.to_csv('breast_cancer_data.csv', index=False)
```

**Option 2: Download from Kaggle**
- Visit [Kaggle Breast Cancer Dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- Download and place `data.csv` or rename it to `breast_cancer_data.csv` in the project root

**Option 3: UCI ML Repository**
- Download from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## 4. Data Overview
- **Source:** Breast Cancer Wisconsin (Diagnostic). Available via scikit-learn (`load_breast_cancer()`) or Kaggle.
- **Samples:** 569 instances
- **Features:** 30 numeric attributes (10 geometric means × 3 calculations)
- **Target:** `diagnosis` (M = Malignant, B = Benign)
- **Class Distribution:** ~357 Benign (63%), ~212 Malignant (37%)
- **Features Include:** 
  - Radius, Texture, Perimeter, Area
  - Smoothness, Compactness, Concavity
  - Concave Points, Symmetry, Fractal Dimension
  - (Each calculated as mean, standard error, and worst value)

---

## 5. Training Features

### 5.1 Dataset Overview
- **Dataset Statistics:**
  - Total samples: 569
  - Number of features: 30
  - Class distribution (Malignant vs Benign)
  - First few rows display
  - Visual class distribution chart

### 5.2 Data Preprocessing
- **Automated Processing:**
  - Feature scaling using StandardScaler
  - Train-test split with stratification
  - Label encoding for diagnosis
  - Train/test sample counts display
  - Feature names preservation

### 5.3 Model Training
- **Logistic Regression (Baseline):**
  - Simple, interpretable model
  - Fast training
  - Serves as performance baseline
  
- **Support Vector Machine (SVM):**
  - GridSearchCV hyperparameter optimization
  - 40 parameter combinations tested
  - 5-fold cross-validation
  - Best parameters identified and displayed

### 5.4 Comprehensive Visualizations
- **Performance Metrics:**
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC tables
  - Bar chart comparison of both models
  
- **Confusion Matrices:**
  - Logistic Regression confusion matrix
  - SVM confusion matrix
  - Visual heatmap representation
  
- **ROC Curves:**
  - Both models plotted together
  - AUC scores displayed
  - Random classifier reference line
  
- **Classification Reports:**
  - Per-class metrics (Precision, Recall, F1)
  - Support values
  - Macro/weighted averages
  
- **Feature Importance (Linear Kernel):**
  - Top 10 features by coefficient magnitude
  - Logistic Regression coefficients
  - SVM coefficients (if linear kernel selected)
  - Visual bar charts with color coding

### 5.5 Results Summary
- **Statistical Comparison:**
  - Accuracy improvement percentage
  - Best SVM hyperparameters
  - Model performance analysis
  - Key findings section
  
- **Downloadable Results:**
  - CSV export of metrics comparison
  - All numerical results included

---

## 6. Modeling Approach

### 6.1 Data Preprocessing
- **Feature Scaling:** StandardScaler applied to normalize features (crucial for SVM)
- **Train-Test Split:** Stratified split to maintain class distribution
- **Encoding:** Label encoding for diagnosis (M=1, B=0)

### 6.2 Model 1: Logistic Regression (Baseline)
- **Purpose:** Simple, interpretable baseline model
- **Configuration:** max_iter=10000 for convergence
- **Advantages:** Fast training, interpretable coefficients, probability outputs
- **Use Case:** Establishes performance floor for comparison

### 6.3 Model 2: Support Vector Machine (SVM)
- **Kernel:** RBF and Linear kernels tested
- **Hyperparameter Tuning:** GridSearchCV with 5-fold cross-validation
- **Parameter Grid:**
  - C: [0.1, 1, 10, 100] (regularization parameter)
  - kernel: ['rbf', 'linear'] (kernel type)
  - gamma: ['scale', 'auto', 0.001, 0.01, 0.1] (kernel coefficient)
- **Total Combinations:** 40 parameter sets evaluated
- **Advantages:** Effective with high-dimensional data, better separation

### 6.4 Evaluation Metrics
- **Accuracy:** Overall correctness of predictions
- **Precision:** Ratio of true positives to predicted positives (minimize false alarms)
- **Recall:** Ratio of true positives to actual positives (catch all cases)
- **F1-Score:** Harmonic mean of precision and recall (balanced metric)
- **ROC-AUC:** Area under ROC curve (probability threshold analysis)
- **Confusion Matrix:** Visual breakdown of TP, TN, FP, FN

---

## 7. How to Run

### 7.1 Running the Training Script
```bash
# Make sure you're in the virtual environment
streamlit run breast_cancer_model_comparison.py
```

The application will open in your default web browser at `http://localhost:8501`

### 7.2 Using Jupyter Notebook (Alternative)
```bash
jupyter notebook
# Create a new notebook and import the training code
```

### 7.3 Workflow
1. **Ensure dataset is present:** Place `breast_cancer_data.csv` in the same directory as the script
2. **Configure parameters (optional):**
   - Adjust test set size (default: 20%)
   - Set random state for reproducibility (default: 42)
3. **Click "🚀 Train and Compare Models" button**
4. **Wait for training completion:**
   - Logistic Regression trains in seconds
   - SVM with GridSearch takes 1-2 minutes
5. **Review comprehensive results:**
   - Performance metrics comparison
   - Visualizations (confusion matrices, ROC curves)
   - Feature importance analysis
   - Key findings and conclusions
6. **Download results:** Export metrics as CSV if needed

### 7.4 Custom Configuration
Edit these parameters in the sidebar:
```python
test_size = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random State", 0, 100, 42)
```

Or modify hyperparameter grid in the code:
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}
```

---

## 8. Expected Results

### 8.1 Performance Benchmarks
Based on the Wisconsin Breast Cancer dataset:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Logistic Regression | ~97% | ~96% | ~99% | ~97% | ~0.99 |
| SVM (Tuned) | ~97-98% | ~97-98% | ~98-99% | ~98% | ~0.99+ |

*Note: Exact values depend on random seed and train-test split*

### 8.2 SVM Best Parameters (Typical)
- **C:** 10-100 (moderate to strong regularization)
- **kernel:** 'rbf' (non-linear separation often preferred)
- **gamma:** 'scale' or 0.001 (smooth decision boundaries)

### 8.3 Key Findings
- Both models demonstrate strong performance (>95% accuracy)
- SVM with GridSearch typically achieves marginal improvements over baseline
- High recall is critical in medical applications (minimize missed cases)
- Model ensemble or agreement checking recommended for clinical use

---

## 9. Future Improvements

### 9.1 Model Enhancements
- [ ] Implement ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- [ ] Add calibration methods for better probability estimates
- [ ] Implement k-fold cross-validation with stratification
- [ ] Try neural networks (MLPs, CNNs) for feature extraction
- [ ] Add SHAP values for model explainability
- [ ] Implement class weight balancing for imbalanced data

### 9.2 Training Features
- [ ] Save trained models to disk (pickle/joblib)
- [ ] Model versioning and comparison
- [ ] Hyperparameter sensitivity analysis
- [ ] Learning curves visualization
- [ ] Validation curve analysis
- [ ] Cross-validation metrics display

### 9.3 Data & Features
- [ ] Feature engineering and selection optimization
- [ ] Dimensionality reduction (PCA, t-SNE)
- [ ] Handling class imbalance (SMOTE, weighted loss)
- [ ] External validation on independent datasets
- [ ] Data quality checks and preprocessing options

### 9.4 Visualization & Reporting
- [ ] Export comprehensive training reports (PDF)
- [ ] Permutation feature importance
- [ ] Partial dependence plots
- [ ] Calibration curves
- [ ] Learning rate analysis
- [ ] Hyperparameter importance visualization

---

## 10. Project Structure (Expanded)

```
Breast Cancer Classification/
├── breast_cancer_model_comparison.py    # Model training and comparison script
├── breast_cancer_data.csv               # Dataset
├── requirements.txt                     # Dependencies
├── README.md                            # Documentation
│
├── models/                              # (Optional) Saved model artifacts
│   ├── lr_model.pkl
│   └── svm_model.pkl
│
├── notebooks/                           # (Optional) Analysis notebooks
│   ├── eda.ipynb                       # Exploratory data analysis
│   ├── model_training.ipynb            # Detailed training walkthrough
│   └── evaluation.ipynb                # Evaluation and visualization
│
├── src/                                 # (Optional) Helper functions
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
└── data/                                # (Optional) Data management
    ├── raw/                            # Original dataset
    └── processed/                      # Preprocessed data
```

---

## 11. Requirements.txt

```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 12. Troubleshooting

### Issue: "FileNotFoundError: breast_cancer_data.csv not found"
**Solution:** Ensure the CSV file is in the same directory as the Python script, or generate it using the scikit-learn dataset loader (see Section 3.4).

### Issue: "SVM training is taking too long"
**Solution:** The GridSearch is testing 40 hyperparameter combinations with 5-fold CV. This is normal. You can:
- Reduce the parameter grid in the code
- Increase n_jobs to use more CPU cores (n_jobs=-1 uses all cores)
- Reduce number of folds in GridSearchCV (cv parameter)
- Wait 2-3 minutes for completion (this is expected)

### Issue: "Module not found errors"
**Solution:** Ensure you've activated the virtual environment and installed all requirements:
```bash
# Activate environment first
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows

# Then install dependencies
pip install -r requirements.txt
```

### Issue: "MemoryError during training"
**Solution:** 
1. Close other applications to free up RAM
2. Reduce training data size temporarily
3. Set n_jobs=1 instead of n_jobs=-1 in GridSearchCV

### Issue: "Import errors (pandas, sklearn, etc.)"
**Solution:**
```bash
# Make sure you're using the correct Python environment
which python  # Check active Python path
python --version  # Verify Python 3.8+

# Reinstall packages
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

---

## 13. References

- **Dataset:** [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **scikit-learn Documentation:**
  - [load_breast_cancer](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
  - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
  - [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)
  - [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
- **Streamlit Documentation:** [Streamlit Docs](https://docs.streamlit.io/)
- **Original Dataset Paper:** Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). Machine learning techniques to diagnose breast cancer from fine-needle aspirates.

---

## 14. License & Attribution

This project is developed for educational purposes as part of **SECB3203-02 Programming for Bioinformatics**.

Dataset citation:
```
Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995). 
Machine learning techniques to diagnose breast cancer from 
fine-needle aspirates. Journal of the American Medical Association, 
273(5), 408-415.
```

---

## 15. Contact & Support

For questions or issues:
1. Check the **Troubleshooting** section (Section 12)
2. Review code comments in `breast_cancer_model_comparison.py`
3. Consult official documentation for libraries used
4. Review the original dataset documentation
5. Open an issue on GitHub (if applicable)

---

**Last Updated:** January 2026  
**Status:** ✅ Ready for Use  
**Version:** 2.0 (Model Training Only)
