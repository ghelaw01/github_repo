# MIMIC-IV Data Processing for External Validation

This document details the preprocessing steps applied to the MIMIC-IV dataset (version 2.2) to construct the cohort for the external validation study.

## Data Source

The MIMIC-IV dataset is a large, publicly available, de-identified database containing comprehensive clinical data of patients admitted to the Beth Israel Deaconess Medical Center in Boston, Massachusetts.

## Cohort Selection

- **Inclusion Criteria:** Adult patients (age ≥ 18 years) with at least one ICU stay.
- **Exclusion Criteria:** Patients with multiple ICU stays within the same hospital admission.
- **Final Cohort Size:** 53,150 patients.

## Feature Extraction

- **Demographics:** Age, gender, ethnicity
- **Vital Signs:** Heart rate, blood pressure, respiratory rate, temperature, SpO2
- **Laboratory Measurements:** 25 common laboratory tests (e.g., white blood cell count, hemoglobin, creatinine)
- **Comorbidities:** 30 comorbidities derived from ICD-9/10 codes using the Elixhauser comorbidity index.

## Outcome Definition

- **Primary Outcome:** 30-day hospital readmission (unplanned admission within 30 days of discharge).

## Data Preprocessing

- **Missing Data:** Mean imputation for numerical features; separate category for missing categorical features.
- **Feature Scaling:** Standardization of all numerical features (mean=0, std=1).
- **Data Splitting:** 70% training set, 30% testing set, preserving temporal order of admissions.
