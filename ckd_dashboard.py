# ============================================================================
# ENHANCED CKD CLASSIFICATION DASHBOARD 
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             balanced_accuracy_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="CKD Classification Dashboard - 2026 KDIGO Guidelines",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CKD STAGE MAPPING (Based on CKD_Stage_CGA column from R processing)
# ============================================================================

# Map from CKD_Stage_CGA values to display information
CKD_STAGES_CGA = {
    'No CKD': {
        'display_name': 'Healthy Kidney',
        'icon': '✅',
        'color': '#2ecc71',
        'card_class': 'healthy-card',
        'message': 'Normal kidney function. Continue healthy lifestyle.',
        'recommendation': '✅ Maintain healthy habits\n✅ Regular exercise (30 min/day)\n✅ Balanced diet\n✅ Annual check-ups',
        'clinical_note': 'eGFR ≥ 60 without proteinuria or other markers of kidney damage'
    },
    'No CKD (Normal Aging)': {
        'display_name': 'Healthy Kidney (Normal Aging)',
        'icon': '✅',
        'color': '#2ecc71',
        'card_class': 'healthy-card',
        'message': 'Normal age-related decline in kidney function. No evidence of kidney disease.',
        'recommendation': '✅ Continue healthy lifestyle\n✅ Annual kidney function monitoring\n✅ Maintain BP control\n✅ Regular exercise',
        'clinical_note': 'For adults ≥70, eGFR 60-89 may represent normal aging without kidney damage'
    },
    'Stage 1 CKD': {
        'display_name': 'Stage 1 CKD',
        'icon': '⚠️',
        'color': '#f39c12',
        'card_class': 'stage1-card',
        'message': 'Kidney damage with normal eGFR. Early intervention recommended.',
        'recommendation': '⚠️ Monitor blood pressure\n⚠️ Control blood sugar\n⚠️ Reduce salt intake\n⚠️ Avoid NSAIDs\n⚠️ Follow-up in 6 months',
        'clinical_note': 'eGFR ≥ 90 with proteinuria, albuminuria, or structural abnormalities'
    },
    'Stage 2 CKD': {
        'display_name': 'Stage 2 CKD',
        'icon': '⚠️⚠️',
        'color': '#e67e22',
        'card_class': 'stage2-card',
        'message': 'Mildly decreased kidney function with kidney damage.',
        'recommendation': '⚠️ Consult nephrologist\n⚠️ Strict BP control (<130/80)\n⚠️ Low protein diet\n⚠️ Monitor eGFR every 6 months',
        'clinical_note': 'eGFR 60-89 with proteinuria, albuminuria, or structural abnormalities'
    },
    'Stage 3a CKD': {
        'display_name': 'Stage 3a CKD',
        'icon': '⚠️⚠️⚠️',
        'color': '#e74c3c',
        'card_class': 'stage3-card',
        'message': 'Mildly to moderately decreased kidney function.',
        'recommendation': '🔴 Urgent nephrology consult\n🔴 Strict medication adherence\n🔴 Monitor for complications\n🔴 Lifestyle modifications',
        'clinical_note': 'eGFR 45-59 indicates moderate kidney damage'
    },
    'Stage 3b CKD': {
        'display_name': 'Stage 3b CKD',
        'icon': '⚠️⚠️⚠️',
        'color': '#e74c3c',
        'card_class': 'stage3-card',
        'message': 'Moderately to severely decreased kidney function.',
        'recommendation': '🔴 Urgent nephrology consult\n🔴 Strict medication adherence\n🔴 Monitor for anemia and bone disease\n🔴 Fluid management',
        'clinical_note': 'eGFR 30-44 indicates significant kidney damage'
    },
    'Stage 4 CKD': {
        'display_name': 'Stage 4 CKD',
        'icon': '🔴',
        'color': '#c0392b',
        'card_class': 'stage4-card',
        'message': 'Severely decreased kidney function. Specialist consultation urgently needed.',
        'recommendation': '🔴 Immediate specialist consult\n🔴 Prepare for dialysis education\n🔴 Strict dietary restrictions\n🔴 Frequent monitoring',
        'clinical_note': 'eGFR 15-29 indicates severe kidney damage'
    },
    'Stage 5 CKD (Kidney Failure)': {
        'display_name': 'Stage 5 CKD (Kidney Failure)',
        'icon': '🔴🔴',
        'color': '#8e44ad',
        'card_class': 'stage5-card',
        'message': 'Kidney failure. Immediate medical attention required.',
        'recommendation': '🔴 Emergency nephrology consult\n🔴 Discuss dialysis/transplant\n🔴 Strict fluid restriction\n🔴 Immediate intervention needed',
        'clinical_note': 'eGFR < 15 indicates kidney failure, dialysis or transplant needed'
    },
    'Unclassified': {
        'display_name': 'Unclassified',
        'icon': '❓',
        'color': '#95a5a6',
        'card_class': 'healthy-card',
        'message': 'Unable to classify. Please consult a healthcare provider.',
        'recommendation': 'Please consult a nephrologist for proper evaluation.',
        'clinical_note': 'Additional testing may be required'
    }
}

# Simplified mapping for 6-class classification
SIMPLIFIED_STAGES = {
    'No CKD': 'Healthy Kidney',
    'No CKD (Normal Aging)': 'Healthy Kidney',
    'Stage 1 CKD': 'Mild CKD (Stage 1–2)',
    'Stage 2 CKD': 'Mild CKD (Stage 1–2)',
    'Stage 3a CKD': 'Moderate CKD (Stage 3)',
    'Stage 3b CKD': 'Moderate CKD (Stage 3)',
    'Stage 4 CKD': 'Severe CKD (Stage 4)',
    'Stage 5 CKD (Kidney Failure)': 'Kidney Failure (Stage 5)',
    'Unclassified': 'Unclassified'
}

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateY(-20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .healthy-card { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
    .stage1-card { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }
    .stage2-card { background: linear-gradient(135deg, #e67e22 0%, #d35400 100%); }
    .stage3-card { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
    .stage4-card { background: linear-gradient(135deg, #c0392b 0%, #8e44ad 100%); }
    .stage5-card { background: linear-gradient(135deg, #8e44ad 0%, #2c3e50 100%); }
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .clinical-note {
        background-color: #e8f4fd;
        padding: 0.8rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .clinical-param-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .clinical-param-card:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.15); }
    .param-value { font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0; }
    .param-label { font-size: 0.9rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
    .param-status { font-size: 0.85rem; padding: 0.25rem 0.5rem; border-radius: 20px; display: inline-block; margin-top: 0.5rem; }
    .status-normal { background-color: #d4edda; color: #155724; }
    .status-mild { background-color: #fff3cd; color: #856404; }
    .status-moderate { background-color: #ffe5b4; color: #cc7b00; }
    .status-severe { background-color: #f8d7da; color: #721c24; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING - FIXED PATH
# ============================================================================

@st.cache_data
def load_enhanced_data():
    """Load the enhanced dataset from your specific directory"""
    # Your exact file path
    file_path = 'data/CKD_Dataset_Enhanced.csv'
    
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"❌ Dataset not found at: {file_path}")
        st.info("Please ensure the file exists at the specified location")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# ============================================================================
# DATA PREPARATION - ALWAYS 80/20 SPLIT
# ============================================================================

@st.cache_data
def prepare_data():
    """Load and prepare data with 80/20 train-test split"""
    
    df = load_enhanced_data()
    
    if df is None:
        st.stop()  # Stop execution if no data
    
    # Define feature columns
    feature_cols = ['Age', 'BMI', 'Systolic_BP', 'Diastolic_BP', 'Hemoglobin', 
                    'RBC_Count', 'WBC_Count', 'Platelet_Count', 'Serum_Creatinine', 
                    'Blood_Urea_Nitrogen', 'eGFR', 'Albumin_Creatinine_Ratio']
    
    cat_cols = ['Gender', 'Diabetes', 'Hypertension', 'Smoking_Status', 'Family_History_Kidney']
    
    # Create feature matrix
    X = df[feature_cols + cat_cols].copy()
    y = df['Target'].copy()
    
    # Encode categorical variables
    for col in cat_cols:
        if col == 'Gender':
            X[col] = X[col].map({'Male': 0, 'Female': 1})
        else:
            X[col] = X[col].map({'No': 0, 'Yes': 1})
    
    # Handle missing values
    if X.isnull().any().any():
        for col in X.columns:
            if X[col].isnull().any():
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
    
    if X.isnull().any().any():
        X = X.fillna(0)
    
    # ALWAYS 80/20 TRAIN-TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    X_train_df = pd.DataFrame(X_train, columns=X.columns)
    X_test_df = pd.DataFrame(X_test, columns=X.columns)
    
    return {
        'df': df,
        'X_train': X_train_df,
        'X_test': X_test_df,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X.columns.tolist(),
        'target_classes': sorted(y.unique().tolist()),
        'scaler': scaler
    }

@st.cache_resource
def train_models(_data):
    """Train multiple models for comparison"""
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        'HistGradientBoosting': HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        ),
        'KNN': KNeighborsClassifier(
            n_neighbors=11,
            weights='distance'
        )
    }
    
    results = {}
    
    for name, model in models.items():
        try:
            if name in ['Random Forest', 'HistGradientBoosting']:
                model.fit(_data['X_train'], _data['y_train'])
                y_pred = model.predict(_data['X_test'])
            else:
                model.fit(_data['X_train_scaled'], _data['y_train'])
                y_pred = model.predict(_data['X_test_scaled'])
            
            results[name] = {
                'model': model,
                'y_pred': y_pred,
                'accuracy': accuracy_score(_data['y_test'], y_pred),
                'balanced_accuracy': balanced_accuracy_score(_data['y_test'], y_pred),
                'f1_weighted': f1_score(_data['y_test'], y_pred, average='weighted')
            }
        except Exception as e:
            results[name] = {
                'model': None,
                'y_pred': None,
                'accuracy': 0.0,
                'balanced_accuracy': 0.0,
                'f1_weighted': 0.0
            }
    
    return results

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_ckd_stage(input_data, data, models):
    """Predict CKD stage using multiple models and clinical guidelines"""
    
    input_df = pd.DataFrame([input_data])
    
    input_df['Gender'] = input_df['Gender'].map({'Male': 0, 'Female': 1})
    input_df['Diabetes'] = input_df['Diabetes'].map({'No': 0, 'Yes': 1})
    input_df['Hypertension'] = input_df['Hypertension'].map({'No': 0, 'Yes': 1})
    input_df['Smoking_Status'] = input_df['Smoking_Status'].map({'No': 0, 'Yes': 1})
    input_df['Family_History_Kidney'] = input_df['Family_History_Kidney'].map({'No': 0, 'Yes': 1})
    
    for col in data['feature_names']:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[data['feature_names']]
    input_df = input_df.fillna(0)
    
    egfr = input_data['eGFR']
    age = input_data['Age']
    acr = input_data.get('Albumin_Creatinine_Ratio', 0)
    creatinine = input_data.get('Serum_Creatinine', 0)
    systolic_bp = input_data.get('Systolic_BP', 0)
    diastolic_bp = input_data.get('Diastolic_BP', 0)
    urine_albumin = input_data.get('Urine_Albumin', 0)
    urine_protein = input_data.get('Urine_Protein', 0)
    diabetes = input_data.get('Diabetes', 'No')
    hypertension = input_data.get('Hypertension', 'No')
    
    if egfr >= 90:
        gfr_cat = 'G1'
    elif egfr >= 60:
        gfr_cat = 'G2'
    elif egfr >= 45:
        gfr_cat = 'G3a'
    elif egfr >= 30:
        gfr_cat = 'G3b'
    elif egfr >= 15:
        gfr_cat = 'G4'
    else:
        gfr_cat = 'G5'
    
    if acr < 30:
        alb_cat = 'A1'
    elif acr <= 300:
        alb_cat = 'A2'
    else:
        alb_cat = 'A3'
    
    risk_matrix = {
        ('G1', 'A1'): 'Low', ('G1', 'A2'): 'Moderate', ('G1', 'A3'): 'High',
        ('G2', 'A1'): 'Low', ('G2', 'A2'): 'Moderate', ('G2', 'A3'): 'High',
        ('G3a', 'A1'): 'Moderate', ('G3a', 'A2'): 'High', ('G3a', 'A3'): 'Very High',
        ('G3b', 'A1'): 'High', ('G3b', 'A2'): 'Very High', ('G3b', 'A3'): 'Very High',
        ('G4', 'A1'): 'Very High', ('G4', 'A2'): 'Very High', ('G4', 'A3'): 'Very High',
        ('G5', 'A1'): 'Very High', ('G5', 'A2'): 'Very High', ('G5', 'A3'): 'Very High'
    }
    kdigo_risk = risk_matrix.get((gfr_cat, alb_cat), 'Unknown')
    
    diabetes_bool = diabetes == 'Yes'
    hypertension_bool = hypertension == 'Yes'
    albuminuria_present = (acr >= 30) or (urine_albumin > 30)
    proteinuria_present = (urine_protein > 30)
    
    has_damage = (
        albuminuria_present or proteinuria_present or
        (diabetes_bool and egfr < 90) or (hypertension_bool and egfr < 90)
    )
    
    if age >= 70 and gfr_cat in ['G1', 'G2'] and not has_damage:
        clinical_stage = 'No CKD (Normal Aging)'
    elif gfr_cat == 'G1':
        clinical_stage = 'Stage 1 CKD' if has_damage else 'No CKD'
    elif gfr_cat == 'G2':
        clinical_stage = 'Stage 2 CKD' if has_damage else 'No CKD'
    elif gfr_cat == 'G3a':
        clinical_stage = 'Stage 3a CKD'
    elif gfr_cat == 'G3b':
        clinical_stage = 'Stage 3b CKD'
    elif gfr_cat == 'G4':
        clinical_stage = 'Stage 4 CKD'
    else:
        clinical_stage = 'Stage 5 CKD (Kidney Failure)'
    
    model_predictions = {}
    for name, result in models.items():
        if result['model'] is not None:
            try:
                if name in ['Random Forest', 'HistGradientBoosting']:
                    pred = result['model'].predict(input_df)[0]
                else:
                    X_scaled = data['scaler'].transform(input_df)
                    pred = result['model'].predict(X_scaled)[0]
                model_predictions[name] = pred
            except:
                model_predictions[name] = "Error"
        else:
            model_predictions[name] = "Not available"
    
    clinical_params = {
        'egfr': egfr,
        'egfr_status': get_egfr_status(egfr),
        'creatinine': creatinine,
        'creatinine_status': get_creatinine_status(creatinine),
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'bp_status': get_bp_status(systolic_bp, diastolic_bp)
    }
    
    return {
        'clinical_stage': clinical_stage,
        'gfr_category': gfr_cat,
        'albuminuria_category': alb_cat,
        'kdigo_risk': kdigo_risk,
        'has_kidney_damage': has_damage,
        'model_predictions': model_predictions,
        'final_stage': clinical_stage,
        'simplified_stage': SIMPLIFIED_STAGES.get(clinical_stage, 'Unclassified'),
        'clinical_params': clinical_params,
        'kidney_damage_details': {
            'albuminuria_present': albuminuria_present,
            'proteinuria_present': proteinuria_present,
            'diabetes_with_reduced_egfr': diabetes_bool and egfr < 90,
            'hypertension_with_reduced_egfr': hypertension_bool and egfr < 90
        }
    }

def get_egfr_status(egfr):
    if egfr >= 90:
        return {'text': 'Normal', 'class': 'status-normal', 'description': 'Normal kidney function'}
    elif egfr >= 60:
        return {'text': 'Mildly reduced', 'class': 'status-mild', 'description': 'Mildly reduced kidney function'}
    elif egfr >= 45:
        return {'text': 'Moderately reduced', 'class': 'status-moderate', 'description': 'Moderately reduced kidney function'}
    elif egfr >= 30:
        return {'text': 'Severely reduced', 'class': 'status-severe', 'description': 'Severely reduced kidney function'}
    else:
        return {'text': 'Kidney failure', 'class': 'status-severe', 'description': 'Kidney failure'}

def get_creatinine_status(creatinine):
    if creatinine <= 1.2:
        return {'text': 'Normal', 'class': 'status-normal', 'description': 'Normal creatinine level'}
    elif creatinine <= 1.5:
        return {'text': 'Mildly elevated', 'class': 'status-mild', 'description': 'Mildly elevated creatinine'}
    elif creatinine <= 2.0:
        return {'text': 'Moderately elevated', 'class': 'status-moderate', 'description': 'Moderately elevated creatinine'}
    else:
        return {'text': 'Severely elevated', 'class': 'status-severe', 'description': 'Severely elevated creatinine'}

def get_bp_status(systolic, diastolic):
    if systolic < 120 and diastolic < 80:
        return {'text': 'Normal', 'class': 'status-normal', 'description': 'Optimal blood pressure'}
    elif systolic < 130 and diastolic < 85:
        return {'text': 'Elevated', 'class': 'status-mild', 'description': 'Elevated blood pressure'}
    elif systolic < 140 and diastolic < 90:
        return {'text': 'Stage 1 Hypertension', 'class': 'status-moderate', 'description': 'Stage 1 hypertension'}
    else:
        return {'text': 'Stage 2 Hypertension', 'class': 'status-severe', 'description': 'Stage 2 hypertension'}

# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Load data and train models
data = prepare_data()
models = train_models(data)

# Get best model for feature importance
best_model = models.get('Random Forest', {}).get('model', None)

if best_model is not None:
    try:
        feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
    except:
        feature_importance = pd.DataFrame({
            'feature': data['feature_names'],
            'importance': np.ones(len(data['feature_names'])) / len(data['feature_names'])
        }).sort_values('importance', ascending=False)
else:
    feature_importance = pd.DataFrame({
        'feature': data['feature_names'],
        'importance': np.ones(len(data['feature_names'])) / len(data['feature_names'])
    }).sort_values('importance', ascending=False)

top_features = feature_importance['feature'].tolist()

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("🏥 CKD Dashboard")
    st.markdown("---")
    st.markdown("### KDIGO 2026 CGA Staging")
    st.markdown("""
    **CGA = Cause - GFR - Albuminuria**
    
    **GFR Categories:**
    - G1: ≥90 (Normal/high)
    - G2: 60-89 (Mildly decreased)
    - G3a: 45-59 (Mild-moderate)
    - G3b: 30-44 (Moderate-severe)
    - G4: 15-29 (Severe)
    - G5: <15 (Kidney failure)
    
    **Albuminuria Categories:**
    - A1: <30 mg/g (Normal)
    - A2: 30-300 mg/g (Moderate)
    - A3: >300 mg/g (Severe)
    """)
    st.markdown("---")
    st.markdown("### Important Notes")
    st.markdown("""
    - eGFR 60-89 is only CKD if there's evidence of kidney damage
    - For adults ≥70, eGFR 60-89 may be normal aging
    - Diagnosis requires persistent abnormalities for ≥3 months
    """)
    st.markdown("---")
    st.markdown(f"### Dataset Info")
    st.markdown(f"- **Total samples:** {len(data['df'])}")
    st.markdown(f"- **Training:** {len(data['X_train'])} (80%)")
    st.markdown(f"- **Test:** {len(data['X_test'])} (20%)")
    st.markdown(f"- **Key Features:** {len(data['feature_names'])}")

# ============================================================================
# MAIN HEADER
# ============================================================================

st.markdown('<div class="main-header">🏥 Chronic Kidney Disease (CKD) Classification Dashboard<br><small style="font-size: 0.8rem;">Based on KDIGO 2026 CGA Staging Guidelines</small></div>', unsafe_allow_html=True)

# ============================================================================
# TABS
# ============================================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔮 Predict CKD", 
    "📊 EDA & CGA Analysis", 
    "🤖 Model Training", 
    "📈 2D Visualizations", 
    "🧊 3D Visualizations",
    "⚖️ Model Comparison", 
    "ℹ️ About"
])

# ============================================================================
# TAB 1: PREDICT CKD
# ============================================================================

with tab1:
    st.header("Real-time CKD Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics & Medical History")
        age = st.number_input("Age", min_value=18, max_value=100, value=50, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=25.0, step=0.5)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=200, value=125, step=5)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=120, value=85, step=5)
        diabetes = st.selectbox("Diabetes", ["No", "Yes"])
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        smoking = st.selectbox("Smoking Status", ["No", "Yes"])
        family_history = st.selectbox("Family History of Kidney Disease", ["No", "Yes"])
    
    with col2:
        st.subheader("Laboratory Tests")
        hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=18.0, value=13.5, step=0.1)
        rbc = st.number_input("RBC Count (million/µL)", min_value=2.0, max_value=6.0, value=4.5, step=0.1)
        wbc = st.number_input("WBC Count (thousand/µL)", min_value=2.0, max_value=15.0, value=7.0, step=0.5)
        platelets = st.number_input("Platelet Count (thousand/µL)", min_value=100, max_value=500, value=250, step=10)
        
        st.subheader("Kidney Function Markers")
        creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=10.0, value=0.90, step=0.05)
        bun = st.number_input("BUN (mg/dL)", min_value=5.0, max_value=100.0, value=15.0, step=1.0)
        egfr = st.number_input("eGFR (mL/min/1.73m²)", min_value=5, max_value=130, value=75, step=5)
        acr = st.number_input("Urine Albumin-to-Creatinine Ratio (mg/g)", min_value=0, max_value=1000, value=10, step=5)
        
        st.subheader("Urine Analysis")
        urine_albumin = st.number_input("Urine Albumin (mg/dL)", min_value=0, max_value=500, value=20, step=10)
        urine_protein = st.number_input("Urine Protein (mg/dL)", min_value=0, max_value=600, value=30, step=10)
    
    if st.button("🔍 Predict CKD Stage (KDIGO 2026)", type="primary"):
        
        input_dict = {
            'Age': age, 'Gender': gender, 'BMI': bmi,
            'Systolic_BP': systolic_bp, 'Diastolic_BP': diastolic_bp,
            'Diabetes': diabetes, 'Hypertension': hypertension,
            'Smoking_Status': smoking, 'Family_History_Kidney': family_history,
            'Hemoglobin': hemoglobin, 'RBC_Count': rbc, 'WBC_Count': wbc,
            'Platelet_Count': platelets, 'Serum_Creatinine': creatinine,
            'Blood_Urea_Nitrogen': bun, 'eGFR': egfr,
            'Albumin_Creatinine_Ratio': acr,
            'Urine_Albumin': urine_albumin,
            'Urine_Protein': urine_protein
        }
        
        result = predict_ckd_stage(input_dict, data, models)
        stage_info = CKD_STAGES_CGA.get(result['clinical_stage'], CKD_STAGES_CGA['Unclassified'])
        
        st.markdown(f"""
        <div class="prediction-card {stage_info['card_class']}">
            <h1 style="font-size: 2.5rem;">{stage_info['icon']} {stage_info['display_name']}</h1>
            <p style="font-size: 1.2rem; margin-top: 1rem;">{stage_info['message']}</p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
                <div><p style="font-size: 0.9rem;">eGFR</p><p style="font-size: 1.5rem; font-weight: bold;">{egfr:.0f}</p><p style="font-size: 0.8rem;">({result['gfr_category']})</p></div>
                <div><p style="font-size: 0.9rem;">ACR</p><p style="font-size: 1.5rem; font-weight: bold;">{acr:.0f}</p><p style="font-size: 0.8rem;">({result['albuminuria_category']})</p></div>
                <div><p style="font-size: 0.9rem;">KDIGO Risk</p><p style="font-size: 1.5rem; font-weight: bold;">{result['kdigo_risk']}</p></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Key Clinical Parameters")
        cp = result['clinical_params']
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.markdown(f"""
            <div class="clinical-param-card">
                <div class="param-label">eGFR</div>
                <div class="param-value">{cp['egfr']:.0f} <span style="font-size: 0.9rem;">mL/min/1.73m²</span></div>
                <div class="param-status {cp['egfr_status']['class']}">{cp['egfr_status']['text']}</div>
                <div style="font-size: 0.8rem; color: #7f8c8d;">{cp['egfr_status']['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            st.markdown(f"""
            <div class="clinical-param-card">
                <div class="param-label">Serum Creatinine</div>
                <div class="param-value">{cp['creatinine']:.2f} <span style="font-size: 0.9rem;">mg/dL</span></div>
                <div class="param-status {cp['creatinine_status']['class']}">{cp['creatinine_status']['text']}</div>
                <div style="font-size: 0.8rem; color: #7f8c8d;">{cp['creatinine_status']['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            st.markdown(f"""
            <div class="clinical-param-card">
                <div class="param-label">Blood Pressure</div>
                <div class="param-value">{cp['systolic_bp']:.0f}/{cp['diastolic_bp']:.0f} <span style="font-size: 0.9rem;">mmHg</span></div>
                <div class="param-status {cp['bp_status']['class']}">{cp['bp_status']['text']}</div>
                <div style="font-size: 0.8rem; color: #7f8c8d;">{cp['bp_status']['description']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.subheader("Kidney Damage Assessment")
        kd_details = result.get('kidney_damage_details', {})
        kidney_damage_factors = []
        kidney_damage_present = result['has_kidney_damage']
        
        if kd_details.get('albuminuria_present', False):
            if acr >= 30:
                alb_category = "A2 (Moderate)" if acr <= 300 else "A3 (Severe)"
                kidney_damage_factors.append(f"Albuminuria detected (ACR: {acr:.0f} mg/g, Category: {alb_category})")
            if urine_albumin > 30:
                kidney_damage_factors.append(f"Elevated urine albumin (Albumin: {urine_albumin:.0f} mg/dL)")
        
        if kd_details.get('proteinuria_present', False):
            if urine_protein <= 150:
                prot_category = "P1 (Mild)"
            elif urine_protein <= 500:
                prot_category = "P2 (Moderate)"
            else:
                prot_category = "P3 (Severe)"
            kidney_damage_factors.append(f"Proteinuria detected (Protein: {urine_protein:.0f} mg/dL, Category: {prot_category})")
        
        if kd_details.get('diabetes_with_reduced_egfr', False):
            kidney_damage_factors.append(f"Diabetes with reduced kidney function (eGFR: {egfr:.0f} mL/min)")
        
        if kd_details.get('hypertension_with_reduced_egfr', False):
            kidney_damage_factors.append(f"Hypertension with reduced kidney function (eGFR: {egfr:.0f} mL/min)")
        
        if kidney_damage_present:
            st.markdown("**⚠️ Evidence of Kidney Damage Detected:**")
            for factor in kidney_damage_factors:
                st.markdown(f"- 🔴 {factor}")
            st.warning("Kidney damage markers present. Further evaluation recommended.")
        else:
            st.success("✅ No evidence of kidney damage detected.")
        
        st.markdown("---")
        
        st.subheader("⚠️ Key Risk Factors")
        risk_factors = []
        
        if diabetes == "Yes": risk_factors.append("Diabetes")
        if hypertension == "Yes": risk_factors.append("Hypertension")
        if family_history == "Yes": risk_factors.append("Family History of Kidney Disease")
        if smoking == "Yes": risk_factors.append("Smoking")
        if creatinine > 1.2: risk_factors.append(f"Elevated Creatinine ({creatinine:.2f} mg/dL)")
        if egfr < 60: risk_factors.append(f"Low eGFR ({egfr:.0f} mL/min/1.73m²)")
        if age > 60: risk_factors.append(f"Advanced Age ({age} years)")
        if hemoglobin < 12: risk_factors.append(f"Low Hemoglobin ({hemoglobin:.1f} g/dL)")
        if bmi >= 30: risk_factors.append(f"Obesity (BMI: {bmi:.1f})")
        if bmi >= 25 and bmi < 30: risk_factors.append(f"Overweight (BMI: {bmi:.1f})")
        if systolic_bp >= 130 or diastolic_bp >= 85: risk_factors.append(f"Elevated BP ({systolic_bp}/{diastolic_bp} mmHg)")
        if acr >= 30: risk_factors.append(f"Albuminuria (ACR: {acr:.0f} mg/g)")
        if urine_protein > 30: risk_factors.append(f"Proteinuria (Urine Protein: {urine_protein:.0f} mg/dL)")
        
        if risk_factors:
            st.markdown("**Present risk factors:**")
            for factor in risk_factors:
                st.markdown(f"- ⚠️ {factor}")
        else:
            st.success("✅ No major risk factors identified")
        
        st.markdown("---")
        
        st.markdown(f"""
        <div class="clinical-note">
            <strong>📋 CGA Classification:</strong><br>
            • <strong>Cause:</strong> {'Kidney damage present' if result['has_kidney_damage'] else 'No evidence of kidney damage'}<br>
            • <strong>GFR Category:</strong> {result['gfr_category']} (eGFR: {egfr:.0f} mL/min/1.73m²)<br>
            • <strong>Albuminuria Category:</strong> {result['albuminuria_category']} (ACR: {acr:.0f} mg/g)<br>
            • <strong>KDIGO Risk:</strong> {result['kdigo_risk']}
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📋 Clinical Recommendations")
        st.info(stage_info['recommendation'])
        
        if kidney_damage_present:
            st.warning("🔴 **Kidney Damage Present:** Immediate nephrology consultation recommended.")

# ============================================================================
# TAB 2: EDA & CGA Analysis
# ============================================================================

with tab2:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stage_counts = data['df']['CKD_Stage_CGA'].value_counts()
        colors = [CKD_STAGES_CGA.get(s, {}).get('color', '#95a5a6') for s in stage_counts.index]
        fig, ax = plt.subplots(figsize=(10, 6))
        stage_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black')
        ax.set_title('CKD Stage Distribution (KDIGO 2026 CGA)', fontsize=14, fontweight='bold')
        ax.set_xlabel('CKD Stage')
        ax.set_ylabel('Count')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        stage_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=colors)
        ax.set_title('CKD Stage Distribution (%)', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    st.subheader("GFR vs Albuminuria by CKD Stage")
    risk_colors = {'Low': '#2ecc71', 'Moderate': '#f39c12', 'High': '#e74c3c', 'Very High': '#c0392b'}
    
    fig_scatter = px.scatter(
        data['df'],
        x='Albumin_Creatinine_Ratio',
        y='eGFR',
        color='KDIGO_Risk_Level',
        symbol='CKD_Stage_CGA',
        title='eGFR vs Albuminuria with KDIGO Risk Levels',
        labels={'eGFR': 'eGFR (mL/min/1.73m²)', 'Albumin_Creatinine_Ratio': 'ACR (mg/g)'},
        color_discrete_map=risk_colors,
        log_y=True,
        height=500
    )
    fig_scatter.add_hline(x=30, line_dash="dash", line_color="orange", annotation_text="A2 Threshold")
    fig_scatter.add_hline(x=300, line_dash="dash", line_color="red", annotation_text="A3 Threshold")
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    st.markdown("Select features for correlation analysis")
    
    numeric_cols = data['df'].select_dtypes(include=[np.number]).columns.tolist()
    available_features = [f for f in top_features if f in numeric_cols]
    
    corr_features = st.multiselect(
        "Choose features for correlation heatmap",
        available_features,
        default=available_features[:min(10, len(available_features))]
    )
    
    if len(corr_features) >= 2:
        corr_matrix = data['df'][corr_features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', 
                    center=0, square=True, linewidths=0.5, ax=ax,
                    cbar_kws={"shrink": 0.8}, annot_kws={"size": 9})
        ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
        st.pyplot(fig)
        plt.close()

# ============================================================================
# TAB 3: MODEL TRAINING
# ============================================================================

with tab3:
    st.header("🤖 Model Training")
    
    model_choice = st.selectbox(
        "Select Model",
        ["Random Forest", "HistGradientBoosting", "Logistic Regression", "K-Nearest Neighbors"]
    )
    
    st.subheader("Hyperparameters")
    
    if model_choice == "Random Forest":
        n_estimators = st.slider("Number of trees", 50, 300, 200, step=50)
        max_depth = st.slider("Max depth", 5, 25, 15)
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                               random_state=42, n_jobs=-1, class_weight='balanced')
                model.fit(data['X_train'], data['y_train'])
                y_pred = model.predict(data['X_test'])
                acc = accuracy_score(data['y_test'], y_pred)
                bal_acc = balanced_accuracy_score(data['y_test'], y_pred)
                st.success(f"✅ Model trained successfully!")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Balanced Accuracy", f"{bal_acc:.4f}")
    
    elif model_choice == "HistGradientBoosting":
        n_estimators = st.slider("Number of estimators", 50, 300, 100, step=50)
        max_depth = st.slider("Max depth", 3, 15, 5)
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                model = HistGradientBoostingClassifier(max_iter=n_estimators, max_depth=max_depth, random_state=42)
                model.fit(data['X_train'], data['y_train'])
                y_pred = model.predict(data['X_test'])
                acc = accuracy_score(data['y_test'], y_pred)
                bal_acc = balanced_accuracy_score(data['y_test'], y_pred)
                st.success(f"✅ Model trained successfully!")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Balanced Accuracy", f"{bal_acc:.4f}")
    
    elif model_choice == "Logistic Regression":
    C_value = st.slider("Regularization (C)", 0.01, 10.0, 1.0, step=0.1)
    
    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            # REMOVED multi_class='ovr' - it's the default behavior
            model = LogisticRegression(
                C=C_value, 
                max_iter=1000, 
                random_state=42,
                class_weight='balanced'
            )
            model.fit(data['X_train_scaled'], data['y_train'])
            y_pred = model.predict(data['X_test_scaled'])
            acc = accuracy_score(data['y_test'], y_pred)
            bal_acc = balanced_accuracy_score(data['y_test'], y_pred)
            st.success(f"✅ Model trained successfully!")
            col1, col2 = st.columns(2)
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("Balanced Accuracy", f"{bal_acc:.4f}")
    
    else:  # KNN
        k_value = st.slider("k value", 3, 31, 11, step=2)
        
        if st.button("Train Model", type="primary"):
            with st.spinner("Training model..."):
                model = KNeighborsClassifier(n_neighbors=k_value, weights='distance')
                model.fit(data['X_train_scaled'], data['y_train'])
                y_pred = model.predict(data['X_test_scaled'])
                acc = accuracy_score(data['y_test'], y_pred)
                bal_acc = balanced_accuracy_score(data['y_test'], y_pred)
                st.success(f"✅ Model trained successfully!")
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("Balanced Accuracy", f"{bal_acc:.4f}")

# ============================================================================
# TAB 4: 2D VISUALIZATIONS
# ============================================================================

with tab4:
    st.header("2D Interactive Visualizations")
    
    fig_importance = px.bar(
        feature_importance.head(15),
        x='importance', y='feature',
        orientation='h',
        title='Top 15 Most Important Features',
        color='importance',
        color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.subheader("Confusion Matrix - Random Forest")
    y_pred_rf = models['Random Forest']['y_pred']
    if y_pred_rf is not None:
        cm = confusion_matrix(data['y_test'], y_pred_rf, labels=data['target_classes'])
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                           x=data['target_classes'], y=data['target_classes'],
                           labels=dict(x="Predicted", y="Actual", color="Count"),
                           title="Confusion Matrix",
                           color_continuous_scale='Blues')
        st.plotly_chart(fig_cm, use_container_width=True)

    # 2D Scatter Plot
    st.subheader("2D Feature Relationships")
    col1, col2 = st.columns(2)
    with col1:
        x_feat = st.selectbox("X-axis", top_features[:15], key="x_2d")
    with col2:
        y_feat = st.selectbox("Y-axis", top_features[:15], key="y_2d")
    
    plot_data = data['df'].copy()
    plot_data['CKD_Status'] = plot_data['Target'].apply(lambda x: 'CKD' if x != 'Healthy Kidney' else 'Healthy')
    
    fig = px.scatter(plot_data, x=x_feat, y=y_feat, color='CKD_Status',
                     title=f'{x_feat} vs {y_feat} by CKD Status',
                     opacity=0.6, size_max=10,
                     hover_data=['Target', 'eGFR', 'Serum_Creatinine', 'Age'])
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 5: 3D VISUALIZATIONS
# ============================================================================

with tab5:
    st.header("3D Feature Visualization")
    st.markdown("Explore relationships between three important features in 3D space")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_3d = st.selectbox("X-axis (3D)", top_features[:10], index=0, key="x_3d")
    with col2:
        y_3d = st.selectbox("Y-axis (3D)", top_features[:10], index=1, key="y_3d")
    with col3:
        z_3d = st.selectbox("Z-axis (3D)", top_features[:10], index=2, key="z_3d")
    
    plot_data = data['df'].copy()
    plot_data['CKD_Status'] = plot_data['Target'].apply(lambda x: 'CKD' if x != 'Healthy Kidney' else 'Healthy')
    
    fig_3d = px.scatter_3d(
        plot_data, x=x_3d, y=y_3d, z=z_3d, color='CKD_Status',
        title=f'3D Visualization: {x_3d} vs {y_3d} vs {z_3d}',
        opacity=0.7, size_max=8,
        hover_data=['Target', 'eGFR', 'Age', 'Serum_Creatinine']
    )
    fig_3d.update_layout(
        scene=dict(xaxis_title=x_3d, yaxis_title=y_3d, zaxis_title=z_3d,
                   camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))),
        height=600
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # 3D by stage
    st.subheader("3D Visualization by CKD Stage")
    fig_3d_stage = px.scatter_3d(
        plot_data, x=x_3d, y=y_3d, z=z_3d, color='Target',
        title=f'3D Visualization by CKD Stage',
        opacity=0.7, size_max=8,
        hover_data=['eGFR', 'Age']
    )
    fig_3d_stage.update_layout(height=600)
    st.plotly_chart(fig_3d_stage, use_container_width=True)

# ============================================================================
# TAB 6: MODEL COMPARISON
# ============================================================================

with tab6:
    st.header("Model Comparison")
    
    results_df = pd.DataFrame([
        {'Model': name, 'Accuracy': result['accuracy'], 
         'Balanced Accuracy': result['balanced_accuracy'], 
         'F1-Score (Weighted)': result['f1_weighted']}
        for name, result in models.items()
    ]).sort_values('Accuracy', ascending=False)
    
    st.dataframe(results_df.style.format({
        'Accuracy': '{:.4f}', 'Balanced Accuracy': '{:.4f}', 'F1-Score (Weighted)': '{:.4f}'
    }).highlight_max(color='lightgreen'))
    
    fig = go.Figure()
    metrics = ['Accuracy', 'Balanced Accuracy', 'F1-Score (Weighted)']
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(x=results_df['Model'], y=results_df[metric], name=metric, marker_color=color))
    fig.update_layout(title='Model Performance Comparison', barmode='group', height=500)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 7: ABOUT - WITH PROFILE SECTION
# ============================================================================

with tab7:
    st.header("About Project")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-box">
        <h3> Project Objective</h3>
        <p>To develop an accurate machine learning model for early detection and staging of Chronic Kidney Disease (CKD) using the KDIGO 2026 CGA (Cause-GFR-Albuminuria) classification system.</p>
        </div>
        
        <div class="info-box">
        <h3> KDIGO 2026 CGA Staging System</h3>
        <p><strong>GFR Categories (G1-G5):</strong></p>
        <ul>
            <li><strong>G1:</strong> eGFR ≥ 90 (Normal/high)</li>
            <li><strong>G2:</strong> eGFR 60-89 (Mildly decreased)</li>
            <li><strong>G3a:</strong> eGFR 45-59 (Mild-moderate)</li>
            <li><strong>G3b:</strong> eGFR 30-44 (Moderate-severe)</li>
            <li><strong>G4:</strong> eGFR 15-29 (Severe)</li>
            <li><strong>G5:</strong> eGFR &lt; 15 (Kidney failure)</li>
        </ul>
        <p><strong>Albuminuria Categories (A1-A3):</strong></p>
        <ul>
            <li><strong>A1:</strong> ACR &lt; 30 mg/g (Normal-mild)</li>
            <li><strong>A2:</strong> ACR 30-300 mg/g (Moderate)</li>
            <li><strong>A3:</strong> ACR &gt; 300 mg/g (Severe)</li>
        </ul>
        </div>
        
        <div class="info-box">
        <h3> Models Implemented</h3>
        <ul>
            <li><strong>Random Forest</strong> - Ensemble of decision trees</li>
            <li><strong>HistGradientBoosting</strong> - Gradient boosting ensemble</li>
            <li><strong>Logistic Regression</strong> - Baseline linear model</li>
            <li><strong>K-Nearest Neighbors</strong> - Instance-based learning</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
        <h3>⚠️ Important Clinical Notes</h3>
        <ul>
            <li>eGFR 60-89 is only CKD if kidney damage markers present</li>
            <li>For adults ≥70, eGFR 60-89 may be normal aging</li>
            <li>A2/A3 albuminuria requires confirmation (2 of 3 samples)</li>
            <li>Always consult a healthcare provider for diagnosis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Profile Section
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        width: 150px; height: 150px; border-radius: 50%; 
                        margin: 0 auto 1rem auto; display: flex; align-items: center; 
                        justify-content: center;">
                <span style="font-size: 4rem;">👩🏽‍💻</span>
            </div>
            <h2>Data Analyst</h2>
            <p style="color: #666; font-size: 1.1rem;">Public Health | Nutritionist | GenAI | Researcher</p>
            <p> <a href="https://www.linkedin.com/in/a-adnan-bns" target="_blank">https://www.linkedin.com/in/a-adnan-bns</a></p>
            <p>🐙 <a href="https://github.com/Amira-YAA" target="_blank">https://github.com/Amira-YAA</a></p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <p><strong>Developed using Streamlit | Based on KDIGO 2026 Clinical Guidelines</strong></p>
        <p><em>Last Updated: March 2026</em></p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# END OF DASHBOARD
# ============================================================================
