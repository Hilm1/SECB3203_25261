# 🧬 Breast Cancer Classification: SVM vs Logistic Regression

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **SECB3203-02 Programming for Bioinformatics Project** > An interactive web application to compare Support Vector Machine (SVM) and Logistic Regression models for diagnosing breast cancer using the Wisconsin Diagnostic Dataset.

---

## 📑 Table of Contents
- [✨ Features](#-features)
- [🛠️ System Architecture](#-system-architecture)
- [📦 Installation & Setup](#-installation--setup)
- [🚀 Usage Guide](#-usage-guide)
- [🧠 Methodology](#-methodology)
- [📊 Sample Results](#-sample-results)
- [📂 Project Structure](#-project-structure)

---

## ✨ Features

This application is divided into two main functional areas:

### 1. 🎯 Model Training & Comparison
- **Interactive Tuning:** Adjust Test Set size and Random State on the fly.
- **Automated Preprocessing:** Handles Label Encoding and Standard Scaling automatically.
- **GridSearch Optimization:** Automatically finds the best hyperparameters (`C`, `kernel`, `gamma`) for the SVM model.
- **Visual Analytics:**
  - Confusion Matrices (Seaborn Heatmaps).
  - ROC Curves (Receiver Operating Characteristic).
  - Feature Importance charts.
  - Comparative Bar Charts for Accuracy, Precision, Recall, and F1-Score.

### 2. 🔮 Prediction Engine
- **Manual Diagnosis:** Input specific features (Radius, Texture, Perimeter, etc.) manually via the UI.
- **Batch Processing:** Upload a CSV file to process multiple patients at once.
- **Probability Scores:** Displays the confidence level (probability %) for Benign vs. Malignant predictions.
- **Consensus Check:** Automatically flags if the two models disagree on a diagnosis.

---

## 🛠️ System Architecture

```mermaid
graph TD
    A[Raw Data (CSV)] --> B(Preprocessing);
    B --> C{Split Data};
    C -->|Train Set| D[Model Training];
    C -->|Test Set| E[Evaluation];
    
    subgraph Training
    D --> F[Logistic Regression (Baseline)];
    D --> G[SVM + GridSearchCV];
    end
    
    subgraph Evaluation
    E --> H[Confusion Matrix];
    E --> I[ROC Curves];
    E --> J[Metrics Comparison];
    end
    
    K[User Input / CSV Upload] --> L[Scaler Transform];
    L --> M[Prediction];
    F --> M;
    G --> M;
    M --> N[Final Diagnosis Output];
