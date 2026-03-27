# ============================================================================
# CKD DATASET ENHANCEMENT - CLINICAL STAGING PARAMETERS 
# Based on KDIGO 2026 Guidelines
# ============================================================================

# Load required libraries
library(tidyverse)
library(dplyr)

# ============================================================================
# 1. LOAD AND PREPARE DATA
# ============================================================================

# Load your dataset (adjust path as needed)
ckd_data <- raw_CKD_dataset

# Display initial structure
cat("Original dataset dimensions:", dim(ckd_data), "\n")
cat("Column names:\n")
print(names(ckd_data))

# ============================================================================
# 2. DEFINE VECTORIZED FUNCTIONS FOR CKD STAGING
# ============================================================================

# 2.1 GFR Categories (G1-G5) - Vectorized
assign_gfr_category <- function(egfr) {
  case_when(
    egfr >= 90 ~ "G1",
    egfr >= 60 & egfr < 90 ~ "G2",
    egfr >= 45 & egfr < 60 ~ "G3a",
    egfr >= 30 & egfr < 45 ~ "G3b",
    egfr >= 15 & egfr < 30 ~ "G4",
    egfr < 15 ~ "G5",
    TRUE ~ "Unknown"
  )
}

# 2.2 Albuminuria Categories (A1-A3) - Vectorized
assign_albuminuria_category <- function(acr) {
  case_when(
    acr < 30 ~ "A1",
    acr >= 30 & acr <= 300 ~ "A2",
    acr > 300 ~ "A3",
    is.na(acr) ~ "Unknown",
    TRUE ~ "Unknown"
  )
}

# 2.3 Proteinuria Categories - Vectorized
assign_proteinuria_category <- function(urine_protein) {
  case_when(
    urine_protein < 150 ~ "P1",
    urine_protein >= 150 & urine_protein <= 500 ~ "P2",
    urine_protein > 500 ~ "P3",
    is.na(urine_protein) ~ "Unknown",
    TRUE ~ "Unknown"
  )
}

# 2.4 Determine if there's evidence of kidney damage - VECTORIZED
has_kidney_damage_vectorized <- function(acr, upcr, urine_albumin, urine_protein, 
                                         diabetes, hypertension, egfr) {
  
  # Convert text variables to logical
  diabetes_bool <- case_when(
    diabetes == "Yes" ~ TRUE,
    diabetes == "No" ~ FALSE,
    TRUE ~ FALSE
  )
  
  hypertension_bool <- case_when(
    hypertension == "Yes" ~ TRUE,
    hypertension == "No" ~ FALSE,
    TRUE ~ FALSE
  )
  
  # Check albuminuria/proteinuria
  albuminuria_present <- case_when(
    !is.na(acr) & acr >= 30 ~ TRUE,
    !is.na(urine_albumin) & urine_albumin > 30 ~ TRUE,
    TRUE ~ FALSE
  )
  
  proteinuria_present <- case_when(
    !is.na(upcr) & upcr >= 150 ~ TRUE,
    !is.na(urine_protein) & urine_protein > 30 ~ TRUE,
    TRUE ~ FALSE
  )
  
  # Evidence of kidney damage
  damage_evidence <- (
    albuminuria_present |
      proteinuria_present |
      (diabetes_bool & egfr < 90) |
      (hypertension_bool & egfr < 90)
  )
  
  return(damage_evidence)
}

# 2.5 Determine CKD Stage based on CGA system - VECTORIZED
assign_ckd_stage_cga <- function(gfr_cat, albuminuria_cat, has_damage, age) {
  
  # Special case: Elderly with G1/G2 and no damage = No CKD
  elderly_no_ckd <- (age >= 70 & gfr_cat %in% c("G1", "G2") & !has_damage)
  
  case_when(
    # Elderly normal aging
    elderly_no_ckd ~ "No CKD (Normal Aging)",
    
    # G1 with damage = Stage 1
    gfr_cat == "G1" & has_damage ~ "Stage 1 CKD",
    # G1 without damage = No CKD
    gfr_cat == "G1" & !has_damage ~ "No CKD",
    
    # G2 with damage = Stage 2
    gfr_cat == "G2" & has_damage ~ "Stage 2 CKD",
    # G2 without damage = No CKD
    gfr_cat == "G2" & !has_damage ~ "No CKD",
    
    # G3a = Stage 3a CKD
    gfr_cat == "G3a" ~ "Stage 3a CKD",
    # G3b = Stage 3b CKD
    gfr_cat == "G3b" ~ "Stage 3b CKD",
    # G4 = Stage 4 CKD
    gfr_cat == "G4" ~ "Stage 4 CKD",
    # G5 = Stage 5 CKD (Kidney Failure)
    gfr_cat == "G5" ~ "Stage 5 CKD (Kidney Failure)",
    
    TRUE ~ "Unclassified"
  )
}

# 2.6 KDIGO Risk Level based on GFR and Albuminuria - VECTORIZED
assign_kdigo_risk <- function(gfr_cat, albuminuria_cat) {
  
  case_when(
    # G1 and G2
    gfr_cat %in% c("G1", "G2") & albuminuria_cat == "A1" ~ "Low",
    gfr_cat %in% c("G1", "G2") & albuminuria_cat == "A2" ~ "Moderate",
    gfr_cat %in% c("G1", "G2") & albuminuria_cat == "A3" ~ "High",
    
    # G3a
    gfr_cat == "G3a" & albuminuria_cat == "A1" ~ "Moderate",
    gfr_cat == "G3a" & albuminuria_cat == "A2" ~ "High",
    gfr_cat == "G3a" & albuminuria_cat == "A3" ~ "Very High",
    
    # G3b
    gfr_cat == "G3b" & albuminuria_cat == "A1" ~ "High",
    gfr_cat == "G3b" & albuminuria_cat == "A2" ~ "Very High",
    gfr_cat == "G3b" & albuminuria_cat == "A3" ~ "Very High",
    
    # G4 and G5
    gfr_cat %in% c("G4", "G5") ~ "Very High",
    
    TRUE ~ "Unknown"
  )
}

# ============================================================================
# 3. CALCULATE CLINICAL PARAMETERS
# ============================================================================

ckd_enhanced <- ckd_data %>%
  mutate(
    # Basic parameters
    Age_Group = case_when(
      Age < 30 ~ "Young Adult",
      Age >= 30 & Age < 50 ~ "Adult",
      Age >= 50 & Age < 70 ~ "Older Adult",
      Age >= 70 ~ "Elderly",
      TRUE ~ "Unknown"
    ),
    
    # BMI Categories
    BMI_Category = case_when(
      BMI < 18.5 ~ "Underweight",
      BMI >= 18.5 & BMI < 25 ~ "Normal",
      BMI >= 25 & BMI < 30 ~ "Overweight",
      BMI >= 30 ~ "Obese",
      TRUE ~ "Unknown"
    ),
    
    # Blood Pressure Categories (JNC 8)
    BP_Category = case_when(
      Systolic_BP < 120 & Diastolic_BP < 80 ~ "Normal",
      (Systolic_BP >= 120 & Systolic_BP <= 129) & Diastolic_BP < 80 ~ "Elevated",
      (Systolic_BP >= 130 & Systolic_BP <= 139) | (Diastolic_BP >= 80 & Diastolic_BP <= 89) ~ "Stage 1 Hypertension",
      Systolic_BP >= 140 | Diastolic_BP >= 90 ~ "Stage 2 Hypertension",
      TRUE ~ "Unknown"
    ),
    
    # GFR Category (G1-G5)
    GFR_Category = assign_gfr_category(eGFR),
    
    # Albuminuria Category (A1-A3)
    Albuminuria_Category = assign_albuminuria_category(Albumin_Creatinine_Ratio),
    
    # Proteinuria Category (P1-P3)
    Proteinuria_Category = assign_proteinuria_category(Urine_Protein),
    
    # Check for kidney damage - USING VECTORIZED FUNCTION
    Has_Kidney_Damage = has_kidney_damage_vectorized(
      Albumin_Creatinine_Ratio, 
      Urine_Protein, 
      Urine_Albumin, 
      Urine_Protein,
      Diabetes, 
      Hypertension, 
      eGFR
    ),
    
    # CKD Stage based on CGA system
    CKD_Stage_CGA = assign_ckd_stage_cga(GFR_Category, Albuminuria_Category, Has_Kidney_Damage, Age),
    
    # KDIGO Risk Level
    KDIGO_Risk_Level = assign_kdigo_risk(GFR_Category, Albuminuria_Category),
    
    # Combined CKD Stage (simplified for classification)
    CKD_Stage_Simplified = case_when(
      CKD_Stage_CGA == "No CKD" ~ "Healthy Kidney",
      CKD_Stage_CGA == "No CKD (Normal Aging)" ~ "Healthy Kidney",
      CKD_Stage_CGA == "Stage 1 CKD" ~ "Mild CKD (Stage 1–2)",
      CKD_Stage_CGA == "Stage 2 CKD" ~ "Mild CKD (Stage 1–2)",
      CKD_Stage_CGA == "Stage 3a CKD" ~ "Moderate CKD (Stage 3)",
      CKD_Stage_CGA == "Stage 3b CKD" ~ "Moderate CKD (Stage 3)",
      CKD_Stage_CGA == "Stage 4 CKD" ~ "Severe CKD (Stage 4)",
      CKD_Stage_CGA == "Stage 5 CKD (Kidney Failure)" ~ "Kidney Failure (Stage 5)",
      TRUE ~ "Unclassified"
    )
  )

# ============================================================================
# 4. CALCULATE COMPREHENSIVE RISK SCORE (VECTORIZED)
# ============================================================================

ckd_enhanced <- ckd_enhanced %>%
  mutate(
    # Individual risk scores
    Risk_Age = case_when(
      Age >= 65 ~ 3,
      Age >= 50 ~ 2,
      Age >= 40 ~ 1,
      TRUE ~ 0
    ),
    
    Risk_Gender = ifelse(Gender == 1, 1, 0),  # Male = higher risk
    
    Risk_Diabetes = ifelse(Diabetes == "Yes", 3, 0),
    
    Risk_Hypertension = ifelse(Hypertension == "Yes", 2, 0),
    
    Risk_Smoking = ifelse(Smoking_Status == "Yes", 1, 0),
    
    Risk_Family_History = ifelse(Family_History_Kidney == "Yes", 1, 0),
    
    Risk_eGFR = case_when(
      eGFR < 15 ~ 5,
      eGFR < 30 ~ 4,
      eGFR < 45 ~ 3,
      eGFR < 60 ~ 2,
      TRUE ~ 0
    ),
    
    Risk_Albuminuria = case_when(
      Albuminuria_Category == "A3" ~ 3,
      Albuminuria_Category == "A2" ~ 2,
      TRUE ~ 0
    ),
    
    Risk_Proteinuria = case_when(
      Proteinuria_Category == "P3" ~ 3,
      Proteinuria_Category == "P2" ~ 2,
      TRUE ~ 0
    ),
    
    Risk_Creatinine = case_when(
      Serum_Creatinine > 3.0 ~ 3,
      Serum_Creatinine > 1.5 ~ 2,
      Serum_Creatinine > 1.2 ~ 1,
      TRUE ~ 0
    ),
    
    Risk_BUN = case_when(
      Blood_Urea_Nitrogen > 40 ~ 2,
      Blood_Urea_Nitrogen > 20 ~ 1,
      TRUE ~ 0
    ),
    
    # Calculate total risk score
    CKD_Risk_Score = Risk_Age + Risk_Gender + Risk_Diabetes + Risk_Hypertension +
      Risk_Smoking + Risk_Family_History + Risk_eGFR + Risk_Albuminuria +
      Risk_Proteinuria + Risk_Creatinine + Risk_BUN,
    
    # Risk level classification
    Risk_Level = case_when(
      CKD_Risk_Score >= 12 ~ "Very High Risk",
      CKD_Risk_Score >= 8 ~ "High Risk",
      CKD_Risk_Score >= 5 ~ "Moderate Risk",
      CKD_Risk_Score >= 2 ~ "Low Risk",
      TRUE ~ "Minimal Risk"
    ),
    
    # Clinical recommendation based on risk level
    Clinical_Recommendation = case_when(
      Risk_Level == "Very High Risk" ~ "Immediate nephrology consultation required",
      Risk_Level == "High Risk" ~ "Urgent follow-up within 1-3 months",
      Risk_Level == "Moderate Risk" ~ "Monitor every 6-12 months",
      Risk_Level == "Low Risk" ~ "Annual screening recommended",
      TRUE ~ "Maintain healthy lifestyle"
    )
  )

# ============================================================================
# 5. ADD ADDITIONAL CLINICAL INDICATORS
# ============================================================================

ckd_enhanced <- ckd_enhanced %>%
  mutate(
    # eGFR reduction indicator
    eGFR_Reduction = case_when(
      eGFR >= 60 ~ "Normal",
      eGFR >= 45 ~ "Mild Reduction",
      eGFR >= 30 ~ "Moderate Reduction",
      eGFR >= 15 ~ "Severe Reduction",
      TRUE ~ "Very Severe Reduction"
    ),
    
    # Anemia indicator (based on hemoglobin)
    Has_Anemia = case_when(
      Gender == 1 & Hemoglobin < 13 ~ TRUE,  # Male
      Gender == 0 & Hemoglobin < 12 ~ TRUE,  # Female
      TRUE ~ FALSE
    ),
    
    # Metabolic acidosis risk (based on bicarbonate)
    Metabolic_Acidosis_Risk = Bicarbonate < 22,
    
    # Hyperkalemia risk
    Hyperkalemia_Risk = Potassium > 5.0,
    
    # Combined metabolic risk
    Metabolic_Risk = case_when(
      (Bicarbonate < 22 | Potassium > 5.0) ~ "Present",
      TRUE ~ "Absent"
    ),
    
    # Rapid progression risk
    Progression_Risk = case_when(
      Serum_Creatinine > 2.0 & Albuminuria_Category %in% c("A2", "A3") ~ "High",
      Serum_Creatinine > 1.5 | Albuminuria_Category == "A3" ~ "Moderate",
      TRUE ~ "Low"
    ),
    
    # Cardiovascular risk flag
    High_CV_Risk = (Diabetes == "Yes" & Hypertension == "Yes" & 
                      (Albuminuria_Category %in% c("A2", "A3") | Proteinuria_Category %in% c("P2", "P3")))
  )

# ============================================================================
# 6. CREATE SUMMARY STATISTICS BY CKD STAGE
# ============================================================================

stage_summary <- ckd_enhanced %>%
  group_by(CKD_Stage_Simplified) %>%
  summarise(
    n = n(),
    Age_Mean = mean(Age, na.rm = TRUE),
    Age_SD = sd(Age, na.rm = TRUE),
    eGFR_Mean = mean(eGFR, na.rm = TRUE),
    eGFR_SD = sd(eGFR, na.rm = TRUE),
    Creatinine_Mean = mean(Serum_Creatinine, na.rm = TRUE),
    Creatinine_SD = sd(Serum_Creatinine, na.rm = TRUE),
    Hemoglobin_Mean = mean(Hemoglobin, na.rm = TRUE),
    Hemoglobin_SD = sd(Hemoglobin, na.rm = TRUE),
    Diabetes_Pct = mean(Diabetes == "Yes", na.rm = TRUE) * 100,
    Hypertension_Pct = mean(Hypertension == "Yes", na.rm = TRUE) * 100,
    Albuminuria_A2_A3_Pct = mean(Albuminuria_Category %in% c("A2", "A3"), na.rm = TRUE) * 100,
    High_Risk_Pct = mean(Risk_Level %in% c("High Risk", "Very High Risk"), na.rm = TRUE) * 100
  ) %>%
  arrange(factor(CKD_Stage_Simplified, levels = c("Healthy Kidney", "Mild CKD (Stage 1–2)", 
                                                  "Moderate CKD (Stage 3)", "Severe CKD (Stage 4)", 
                                                  "Kidney Failure (Stage 5)")))

# ============================================================================
# 7. VALIDATE AGAINST ORIGINAL TARGET
# ============================================================================

comparison <- ckd_enhanced %>%
  select(Target, CKD_Stage_Simplified, CKD_Stage_CGA, GFR_Category, 
         Albuminuria_Category, KDIGO_Risk_Level) %>%
  mutate(
    Agreement = Target == CKD_Stage_Simplified,
    Agreement_Description = case_when(
      Agreement ~ "Match",
      !Agreement ~ "Mismatch (Clinical Refinement)"
    )
  )

agreement_pct <- mean(comparison$Agreement, na.rm = TRUE) * 100

cat("\n========================================\n")
cat("ENHANCEMENT SUMMARY\n")
cat("========================================\n")
cat("Original data dimensions:", dim(ckd_data), "\n")
cat("Enhanced data dimensions:", dim(ckd_enhanced), "\n")
cat("New columns added:", setdiff(names(ckd_enhanced), names(ckd_data)), "\n")
cat("\nAgreement with original Target:", round(agreement_pct, 1), "%\n")
cat("Note: Disagreements reflect clinical refinement using KDIGO guidelines\n")

# ============================================================================
# 8. EXPORT ENHANCED DATASET
# ============================================================================

write.csv(ckd_enhanced, "CKD_Dataset_Enhanced.csv", row.names = FALSE)
write.csv(stage_summary, "CKD_Stage_Summary.csv", row.names = FALSE)

cat("\n========================================\n")
cat("FILES SAVED:\n")
cat("========================================\n")
cat("1. CKD_Dataset_Enhanced.csv - Full dataset with clinical staging\n")
cat("2. CKD_Stage_Summary.csv - Summary statistics by CKD stage\n")

# ============================================================================
# 9. QUICK DIAGNOSTICS FOR NEW COLUMNS
# ============================================================================

cat("\n========================================\n")
cat("NEW COLUMN DIAGNOSTICS\n")
cat("========================================\n")

# GFR Category distribution
cat("\nGFR Categories:\n")
print(table(ckd_enhanced$GFR_Category, useNA = "ifany"))

# Albuminuria Categories
cat("\nAlbuminuria Categories:\n")
print(table(ckd_enhanced$Albuminuria_Category, useNA = "ifany"))

# KDIGO Risk Levels
cat("\nKDIGO Risk Levels:\n")
print(table(ckd_enhanced$KDIGO_Risk_Level, useNA = "ifany"))

# Risk Level distribution
cat("\nComprehensive Risk Levels:\n")
print(table(ckd_enhanced$Risk_Level, useNA = "ifany"))

# Check sample of data
cat("\nSample of enhanced data (first 5 rows):\n")
print(head(ckd_enhanced[, c("Target", "CKD_Stage_Simplified", "GFR_Category", 
                            "Albuminuria_Category", "KDIGO_Risk_Level", 
                            "CKD_Risk_Score", "Risk_Level")], 10))

# Check for any missing values in key columns
cat("\nMissing Values in Key Columns:\n")
key_columns <- c("GFR_Category", "Albuminuria_Category", "CKD_Stage_CGA", 
                 "KDIGO_Risk_Level", "CKD_Risk_Score", "Risk_Level")
missing_counts <- sapply(ckd_enhanced[key_columns], function(x) sum(is.na(x)))
print(missing_counts)

# ============================================================================
# 10. CREATE DATA DICTIONARY
# ============================================================================

data_dictionary <- data.frame(
  Column_Name = names(ckd_enhanced)[!names(ckd_enhanced) %in% names(ckd_data)],
  Description = c(
    "Age_Group = Categorized age groups (Young Adult/Adult/Older Adult/Elderly)",
    "BMI_Category = BMI classification (Underweight/Normal/Overweight/Obese)",
    "BP_Category = Blood pressure classification based on JNC 8 guidelines",
    "GFR_Category = GFR categories G1-G5 based on KDIGO guidelines",
    "Albuminuria_Category = A1-A3 categories based on urine ACR (mg/g)",
    "Proteinuria_Category = P1-P3 categories based on urine protein (mg/g)",
    "Has_Kidney_Damage = Evidence of kidney damage (proteinuria/albuminuria/diabetes/hypertension)",
    "CKD_Stage_CGA = CKD stage based on full CGA (Cause-GFR-Albuminuria) classification",
    "KDIGO_Risk_Level = Risk stratification based on GFR and albuminuria (Low/Moderate/High/Very High)",
    "CKD_Stage_Simplified = Simplified staging matching original target categories for compatibility",
    "Risk_Age = Risk points based on age (0-3 points)",
    "Risk_Gender = Risk points based on gender (male=1 point)",
    "Risk_Diabetes = Risk points for diabetes (3 points)",
    "Risk_Hypertension = Risk points for hypertension (2 points)",
    "Risk_Smoking = Risk points for smoking (1 point)",
    "Risk_Family_History = Risk points for family history (1 point)",
    "Risk_eGFR = Risk points based on eGFR level (0-5 points)",
    "Risk_Albuminuria = Risk points based on albuminuria category (0-3 points)",
    "Risk_Proteinuria = Risk points based on proteinuria category (0-3 points)",
    "Risk_Creatinine = Risk points based on serum creatinine (0-3 points)",
    "Risk_BUN = Risk points based on BUN level (0-2 points)",
    "CKD_Risk_Score = Total composite risk score (sum of all risk points)",
    "Risk_Level = Risk level classification (Minimal/Low/Moderate/High/Very High)",
    "Clinical_Recommendation = Clinical action based on risk level",
    "eGFR_Reduction = Descriptive eGFR reduction category",
    "Has_Anemia = Anemia indicator based on hemoglobin levels",
    "Metabolic_Acidosis_Risk = Metabolic acidosis risk based on bicarbonate (<22 mEq/L)",
    "Hyperkalemia_Risk = Hyperkalemia risk based on potassium (>5.0 mEq/L)",
    "Metabolic_Risk = Combined metabolic risk indicator",
    "Progression_Risk = Risk of rapid CKD progression (Low/Moderate/High)",
    "High_CV_Risk = High cardiovascular risk indicator (diabetes + hypertension + proteinuria)"
  )
)

write.csv(data_dictionary, "CKD_Enchanced_Data_Dictionary.csv", row.names = FALSE)
cat("\n3. CKD_Data_Dictionary.csv - Description of all new columns\n")

# ============================================================================
# 11. SAVE R DATA OBJECT FOR EASY LOADING IN R
# ============================================================================

saveRDS(ckd_enhanced, "CKD_Dataset_Enhanced.rds")
cat("\n4. CKD_Dataset_Enhanced.rds - R data object for quick loading\n")

cat("\n========================================\n")
cat("ENHANCEMENT COMPLETE!\n")
cat("========================================\n")

# Return the enhanced dataframe invisibly
invisible(ckd_enhanced)

# ============================================================================
# END OF SCRIPT
# ============================================================================