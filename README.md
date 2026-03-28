# 🏥 Chronic Kidney Disease (CKD) Classification Dashboard

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![KDIGO](https://img.shields.io/badge/KDIGO-2026%20Guidelines-green.svg)](https://kdigo.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interactive machine learning dashboard for early detection and staging of Chronic Kidney Disease (CKD) using the **KDIGO 2026 CGA (Cause-GFR-Albuminuria)** classification system.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [KDIGO 2026 Staging System](#-kdigo-2026-staging-system)
4. [Data Dictionary](#-data-dictionary)
5. [Installation](#-installation)

---

## 🎯 Project Overview

This project implements a comprehensive machine learning pipeline for **early detection and staging of Chronic Kidney Disease (CKD)** using clinical and laboratory measurements. The system follows the **KDIGO 2026 CGA (Cause-GFR-Albuminuria) guidelines** for accurate staging.
LICENSE: CC0: Public Domain Kaggle Dataset: Chronic Kidney Disease (CKD) Clinical Dataset https://www.kaggle.com/datasets/priyankabarik/chronic-kidney-disease-ckd-clinical-dataset

### Key Highlights:
- ✅ **Real-time CKD prediction** with interactive Streamlit dashboard
- ✅ **KDIGO 2026 compliant staging** (Cause-GFR-Albuminuria system)
- ✅ **Complete kidney damage assessment** based on clinical guidelines
- ✅ **Comprehensive risk factor analysis** with color-coded feedback
- ✅ **3D and 2D interactive visualizations** for data exploration
- ✅ **Multiple ML models** with performance comparison
- ✅ **Clinical recommendations** based on risk levels

---

## ✨ Key Features

### 🔮 Real-Time Prediction
- Interactive patient data input form
- Instant CKD stage prediction using KDIGO 2026 guidelines
- **Key Clinical Parameters Dashboard** (eGFR, Creatinine, Blood Pressure)
- **Kidney Damage Assessment** with detailed factor analysis
- **Risk Factors Identification** with severity warnings
- Color-coded feedback based on risk levels

### 📊 Clinical Parameters Dashboard
- **eGFR**: Normal/Mildly reduced/Moderately reduced/Severely reduced/Kidney failure
- **Serum Creatinine**: Normal/Mildly elevated/Moderately elevated/Severely elevated
- **Blood Pressure**: Normal/Elevated/Stage 1/Stage 2 Hypertension

### 🔍 Kidney Damage Assessment (Based on KDIGO 2026)
- Albuminuria detection (ACR ≥ 30 OR urine albumin > 30)
- Proteinuria detection (urine protein > 30)
- Diabetes with reduced kidney function (eGFR < 90)
- Hypertension with reduced kidney function (eGFR < 90)

### ⚠️ Risk Factor Analysis
- **15+ risk factors** evaluated including:
  - Diabetes, Hypertension, Smoking
  - Family history, Advanced age
  - Elevated creatinine, Low eGFR
  - Obesity, Overweight
  - Albuminuria, Proteinuria
  - Low hemoglobin, Elevated blood pressure

### 📈 Interactive Visualizations
- **2D Scatter Plots**: Explore relationships between any two features
- **3D Visualizations**: Interactive 3D plots for multi-feature analysis
- **Correlation Heatmaps**: Feature correlation analysis
- **Feature Importance**: Top 15 most important features
- **Confusion Matrix**: Model performance visualization

### 🤖 Machine Learning Models
- **HistGradientBoosting** - Gradient boosted decision trees
- **Random Forest** - Ensemble of decision trees
- **Logistic Regression** - Baseline linear classifier
- **K-Nearest Neighbors** - Instance-based learning

### 📊 Model Evaluation
- Accuracy & Balanced Accuracy
- F1-Score (Weighted)
- Confusion Matrix
- Classification Report
- Model Comparison Dashboard

---

## 📊 KDIGO 2026 Staging System

### GFR Categories (G1-G5)
| Category | eGFR (mL/min/1.73m²) | Description |
|----------|---------------------|-------------|
| G1 | ≥ 90 | Normal/high |
| G2 | 60-89 | Mildly decreased |
| G3a | 45-59 | Mild-moderate decrease |
| G3b | 30-44 | Moderate-severe decrease |
| G4 | 15-29 | Severe decrease |
| G5 | < 15 | Kidney failure |

### Albuminuria Categories (A1-A3)
| Category | ACR (mg/g) | Description |
|----------|------------|-------------|
| A1 | < 30 | Normal to mildly increased |
| A2 | 30-300 | Moderately increased |
| A3 | > 300 | Severely increased |

### KDIGO Risk Classification
| GFR Category | A1 | A2 | A3 |
|--------------|----|----|-----|
| G1/G2 | Low | Moderate | High |
| G3a | Moderate | High | Very High |
| G3b | High | Very High | Very High |
| G4/G5 | Very High | Very High | Very High |

---

## 📚 Data Dictionary

### Demographic Features
| Column | Description |
|--------|-------------|
| Age | Patient age (18-100 years) |
| Gender | Male (1) / Female (0) |
| BMI | Body Mass Index (15-50 kg/m²) |

### Vital Signs
| Column | Description | Normal Range |
|--------|-------------|--------------|
| Systolic_BP | Systolic blood pressure | 90-120 mmHg |
| Diastolic_BP | Diastolic blood pressure | 60-80 mmHg |
| Heart_Rate | Pulse rate | 60-100 bpm |

### Blood Tests
| Column | Description | Normal Range |
|--------|-------------|--------------|
| Hemoglobin | Oxygen carrying protein | 12-16 g/dL |
| RBC_Count | Red blood cells | 4.5-5.9 million/µL |
| WBC_Count | White blood cells | 4.5-11 thousand/µL |
| Platelet_Count | Blood clotting cells | 150-450 thousand/µL |

### Kidney Function Markers
| Column | Description | Normal Range |
|--------|-------------|--------------|
| Serum_Creatinine | Waste product in blood | 0.6-1.2 mg/dL |
| Blood_Urea_Nitrogen | Blood urea nitrogen | 7-20 mg/dL |
| eGFR | Estimated GFR | >90 mL/min |
| Albumin_Creatinine_Ratio | Urine ACR | <30 mg/g |
| Urine_Albumin | Direct urine albumin | <30 mg/dL |
| Urine_Protein | Direct urine protein | <30 mg/dL |

### Medical History
| Column | Values |
|--------|--------|
| Diabetes | Yes / No |
| Hypertension | Yes / No |
| Smoking_Status | Yes / No |
| Family_History_Kidney | Yes / No |

### Electrolytes
| Column | Description | Normal Range |
|--------|-------------|--------------|
| Sodium | Serum sodium | 135-145 mEq/L |
| Potassium | Serum potassium | 3.5-5.5 mEq/L |
| Calcium | Serum calcium | 8.5-10.5 mg/dL |
| Bicarbonate | Serum bicarbonate | 22-28 mEq/L |

### Derived Features (CKD Staging)
| Column | Description |
|--------|-------------|
| GFR_Category | G1-G5 categories based on eGFR |
| Albuminuria_Category | A1-A3 categories based on ACR |
| Has_Kidney_Damage | Evidence of kidney damage |
| CKD_Stage_CGA | Full CGA-based CKD stage |
| KDIGO_Risk_Level | Risk stratification (Low/Moderate/High/Very High) |

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git (optional)

### Clone Repository
```bash
git clone https://github.com/yourusername/ckd-classification.git
cd ckd-classification

ckd-classification/
│
├── ckd_dashboard.py                 # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── LICENSE.md
│
├── data/
│   ├── raw_CKD_Dataset.csv     # original dataset
│   ├── CKD_Dataset_Enhanced.csv     # Enhanced dataset with CGA staging
│   └── CKD_Enchanced_Data_Dictionary.csv  # Data dictionary
│  
├── notebooks/
│   ├── CKD_Staging_R_Script.R    # R script for CGA staging
