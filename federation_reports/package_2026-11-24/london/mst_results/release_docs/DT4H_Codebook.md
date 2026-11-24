# DT4H UC1 Synthetic Dataset -- Codebook

Generated 2026-11-23 by the pipeline. One row per released column. Ranges and category lists are aggregate facts over the training split (also published in the profiling reports). **A null is never 'unknown noise' in this dataset -- its meaning is stated per column.**

Numeric ranges below are **coarsened for disclosure control**: each min/max is rounded outward (min down, max up) to 2 significant figures, so a published endpoint is never an exact single-patient value while still bounding the true range.

| column | type | description | values / range | missing % | null means |
|---|---|---|---|---|---|
| `patient_demographics_gender` | categorical NOMINAL | Gender of the patient | female, male | 0% | n/a (no nulls) |
| `patient_demographics_age` | numeric NUMERIC | Age of the patient at admission date | 25.0 .. 89.0 | 0% | n/a (no nulls) |
| `encounters_encounterClass` | categorical NOMINAL | Type of encounter (emergency, impatient, outpatient, etc) | IMP | 0% | n/a (no nulls) |
| `encounters_admissionYear` | categorical NOMINAL | Year of admission to hospital | 1995, 1997, 2000, 2002, 2003, 2004 … | 0% | n/a (no nulls) |
| `encounters_lengthOfStay` | numeric NUMERIC | The total number of days the patient has been hospitalized | 0.0 .. 130.0 | 0% | n/a (no nulls) |
| `encounters_numOfPreviousHFStays_count` | numeric NUMERIC | Number of previous hospital stays for HF | 0.0 .. 320.0 | 0% | n/a (no nulls) |
| `vital_signs_weight_value_p6mo_first` | numeric NUMERIC | Value of the vital sign | 31.0 .. 230.0 | 7% | not measured |
| `vital_signs_weight_value_p6mo_last` | numeric NUMERIC | Value of the vital sign | 32.0 .. 230.0 | 7% | not measured |
| `vital_signs_height_value_p1a_avg` | numeric NUMERIC | Value of the vital sign | 97.0 .. 200.0 | 15% | not measured |
| `vital_signs_weight_value_last` | numeric NUMERIC | Value of the vital sign | 32.0 .. 230.0 | 9% | not measured |
| `vital_signs_height_value_last` | numeric NUMERIC | Value of the vital sign | 97.0 .. 200.0 | 21% | not measured |
| `vital_signs_diastolicBp_value_last` | numeric NUMERIC | Value of the vital sign | 31.0 .. 130.0 | 0% | not measured |
| `vital_signs_diastolicBp_value_first` | numeric NUMERIC | Value of the vital sign | 38.0 .. 200.0 | 0% | not measured |
| `vital_signs_heartRate_value_last` | numeric NUMERIC | Value of the vital sign | 35.0 .. 160.0 | 0% | n/a (no nulls) |
| `vital_signs_heartRate_value_first` | numeric NUMERIC | Value of the vital sign | 33.0 .. 180.0 | 0% | n/a (no nulls) |
| `vital_signs_systolicBp_value_last` | numeric NUMERIC | Value of the vital sign | 60.0 .. 210.0 | 0% | not measured |
| `vital_signs_systolicBp_value_first` | numeric NUMERIC | Value of the vital sign | 64.0 .. 280.0 | 0% | not measured |
| `vital_signs_oxygenSaturation_value_last` | numeric NUMERIC | Value of the vital sign | 75.0 .. 100.0 | 93% | not measured |
| `vital_signs_oxygenSaturation_value_first` | numeric NUMERIC | Value of the vital sign | 75.0 .. 100.0 | 93% | not measured |
| `lab_results_hemoglobin_value_last` | numeric NUMERIC | Value of the lab result | 59.0 .. 210.0 | 0% | n/a (no nulls) |
| `lab_results_hemoglobin_value_first` | numeric NUMERIC | Value of the lab result | 62.0 .. 200.0 | 0% | n/a (no nulls) |
| `lab_results_ferritin_value_last` | numeric NUMERIC | Value of the lab result | 9.0 .. 15000.0 | 80% | not measured |
| `lab_results_ferritin_value_first` | numeric NUMERIC | Value of the lab result | 9.0 .. 5100.0 | 80% | not measured |
| `lab_results_tfs_value_last` | numeric NUMERIC | Value of the lab result | 3.0 .. 93.0 | 63% | not measured |
| `lab_results_tfs_value_first` | numeric NUMERIC | Value of the lab result | 0.0 .. 93.0 | 63% | not measured |
| `lab_results_ntProBnp_value_last` | numeric NUMERIC | Value of the lab result | 50.0 .. 70000.0 | 27% | not measured |
| `lab_results_ntProBnp_value_first` | numeric NUMERIC | Value of the lab result | 50.0 .. 70000.0 | 27% | not measured |
| `lab_results_crpNonHs_value_last` | numeric NUMERIC | Value of the lab result | 0.6 .. 420.0 | 1% | not measured |
| `lab_results_crpNonHs_value_first` | numeric NUMERIC | Value of the lab result | 0.6 .. 320.0 | 1% | not measured |
| `lab_results_tropTnHs_value_last` | numeric NUMERIC | Value of the lab result | 3.0 .. 4500.0 | 26% | not measured |
| `lab_results_tropTnHs_value_first` | numeric NUMERIC | Value of the lab result | 3.0 .. 7400.0 | 26% | not measured |
| `lab_results_triGly_value_last` | numeric NUMERIC | Value of the lab result | 0.4 .. 3.2 | 87% | not measured |
| `lab_results_triGly_value_first` | numeric NUMERIC | Value of the lab result | 0.4 .. 3.2 | 87% | not measured |
| `lab_results_cholTot_value_last` | numeric NUMERIC | Value of the lab result | 1.7 .. 6.7 | 87% | not measured |
| `lab_results_cholTot_value_first` | numeric NUMERIC | Value of the lab result | 0.5 .. 6.7 | 87% | not measured |
| `lab_results_hdl_value_last` | numeric NUMERIC | Value of the lab result | 0.4 .. 2.7 | 87% | not measured |
| `lab_results_hdl_value_first` | numeric NUMERIC | Value of the lab result | 0.5 .. 2.7 | 87% | not measured |
| `lab_results_creatUS_value_last` | numeric NUMERIC | Value of the lab result | 45.0 .. 2400.0 | 83% | not measured |
| `lab_results_creatUS_value_first` | numeric NUMERIC | Value of the lab result | 45.0 .. 2400.0 | 83% | not measured |
| `lab_results_bun_value_last` | numeric NUMERIC | Value of the lab result | 1.5 .. 56.0 | 16% | not measured |
| `lab_results_bun_value_first` | numeric NUMERIC | Value of the lab result | 1.6 .. 50.0 | 16% | not measured |
| `lab_results_potassium_value_last` | numeric NUMERIC | Value of the lab result | 2.8 .. 7.4 | 1% | not measured |
| `lab_results_potassium_value_first` | numeric NUMERIC | Value of the lab result | 2.6 .. 7.3 | 1% | not measured |
| `lab_results_sodium_value_first` | numeric NUMERIC | Value of the lab result | 100.0 .. 160.0 | 2% | not measured |
| `lab_results_sodium_value_last` | numeric NUMERIC | Value of the lab result | 110.0 .. 170.0 | 2% | not measured |
| `lab_results_albuminBS_value_last` | numeric NUMERIC | Value of the lab result | 17.0 .. 50.0 | 7% | not measured |
| `lab_results_albuminBS_value_first` | numeric NUMERIC | Value of the lab result | 10.0 .. 52.0 | 7% | not measured |
| `lab_results_hba1c%_value_first` | numeric NUMERIC | Value of the lab result | 4.5 .. 13.0 | 95% | not measured |
| `lab_results_hba1c%_value_last` | numeric NUMERIC | Value of the lab result | 4.5 .. 13.0 | 95% | not measured |
| `lab_results_hba1c_value_first` | numeric NUMERIC | Value of the lab result | 25.0 .. 140.0 | 78% | not measured |
| `lab_results_hba1c_value_last` | numeric NUMERIC | Value of the lab result | 25.0 .. 140.0 | 78% | not measured |
| `lab_results_validSerumCreatinine_value_first` | numeric NUMERIC | Value of the lab result | 2.1 .. 23.0 | 8% | not measured |
| `lab_results_valideGFR_value_last` | numeric NUMERIC | Value of the lab result | 2.0 .. 90.0 | 0% | not measured |
| `lab_results_valideGFR_value_first` | numeric NUMERIC | Value of the lab result | 3.0 .. 90.0 | 0% | not measured |
| `symptoms_Ankle_swelling_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Ascites_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Breathlessness_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Cardiac_murmur_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Chest_pain_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Cheyne_stokes_respiration_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Depression_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Dizziness_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Elevated_jugular_venous_pressure_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Fatigue_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Hepatojugular_reflux_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Hepatomegaly_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Intermittent_claudication_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Irregular_pulse_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Loss_of_appetite_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Nocturnal_cough_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Oliguria_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Orthopnoea_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Palpitations_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Paroxysmal_nocturnal_dyspnea_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Peripheral_edema_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Pleural_effusion_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Pulmonary_crepitations_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Reduced_exercise_tolerance_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Syncope_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Tachycardia_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false, true | 0% | n/a (no nulls) |
| `symptoms_Tachypnoea_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Third_heart_sound_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Weight_gain_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `symptoms_Weight_loss_display_pET_any` | categorical BOOLEAN | Whether the symptom was present from 24 hours before admission through discharge | false | 0% | n/a (no nulls) |
| `electrocardiographs_ecg_st_pET` | categorical BOOLEAN | ST-elevation: (SNOMED CT 164931005) value (true/false) observed from 1 months before admission through dischar… | Missing | 100% | explicit 'Missing' category |
| `electrocardiographs_ecg_ischemia_without_st_pET` | categorical BOOLEAN | Ischemia without st-elevation: (SNOMED CT 52674009) value (true/false) observed from 1 months before admission… | Missing | 100% | explicit 'Missing' category |
| `electrocardiographs_ecg_type_of_rhythms_pET_first` | categorical ARRAY[NOMINAL] | Type of rhythm (LOINC 76281-5) value from 1 months before admission through discharge | Missing | 100% | explicit 'Missing' category |
| `electrocardiographs_ecg_type_of_rhythms_pET_last` | categorical ARRAY[NOMINAL] | Type of rhythm (LOINC 76281-5) value from 1 months before admission through discharge | Missing | 100% | explicit 'Missing' category |
| `smoking_status_smoker_last` | categorical BOOLEAN | Determines if the patient is currently smoking. A patient is considered a current smoker if their last recorde… | false, true | 0% | n/a (no nulls) |
| `smoking_status_formerSmoker_last` | categorical BOOLEAN | Whether the patient smoked within 1 year prior to the admission end.The patient is considered to have smoked i… | Missing, false, true | 77% | explicit 'Missing' category |
| `smoking_status_smoker_startTime_count` | numeric NUMERIC | Total number of smoking periods. | 0.0 .. 3.0 | 0% | n/a (no nulls) |
| `nyha_nyha` | numeric NOMINAL | The latest value of the New York Heart Assessment as LOINC Code. If missing, falls back to a value randomly dr… | n/a | 100% | not assessed (sentinel 0) |
| `nyha_nyha_pET` | numeric NOMINAL | New York Heart Association category from 6 months before admission through discharge. If missing, falls back t… | n/a | 100% | not assessed (sentinel 0) |
| `vital_signs_systolicBpDuringEncounter_value_pET` | numeric NUMERIC | Latest value of the systolic blood pressure measured during the encounter, restricted to physiologically plaus… | 60.0 .. 210.0 | 0% | n/a (no nulls) |
| `vital_signs_bmi_value_pET` | numeric NUMERIC | Latest BMI recorded during the encounter. If missing, falls back to the average of the last 2 values recorded … | 14.0 .. 94.0 | 0% | n/a (no nulls) |
| `lab_results_creatBS_value_p3a_avg` | numeric NUMERIC | Average of the last 2 serum creatinine values recorded over the three years preceding the reference time point… | 3.9 .. 150.0 | 0% | n/a (no nulls) |
| `lab_results_validSerumCreatinine_value_pET` | numeric NUMERIC | Latest serum creatinine value recorded during the encounter. If missing, falls back to the average of the last… | 3.9 .. 23.0 | 0% | n/a (no nulls) |
| `hyperkalemia_severity_categorizedValue` | categorical NOMINAL | Severity of hyperkalemia: LOINC 2823-3 | Missing, mild, moderate, normal, severe | 1% | explicit 'Missing' category |
| `ckd_severity_categorizedValue` | categorical NOMINAL | Severity of chronic kidney disease: LOINC 69405-9 | Missing, kidney_failure, mild_to_moderate_decrease, mildly_decreased, moderate_to_severe_decrease, normal_or_high … | 0% | explicit 'Missing' category |
| `conditions_heartFailure_timeFromEarliest_first` | numeric NUMERIC | Time elapsed (in months) since heart failure is observed for the first time until the reference time point | 0.0 .. 67.0 | 0% | n/a (no nulls) |
| `conditions_heart_failure_hf_within_18mo_any` | categorical BOOLEAN | Indicates whether the condition was diagnosed within 18 months before the reference time point, corresponding … | true | 0% | n/a (no nulls) |
| `conditions_heart_failure_occurred_prior_to_18_months_any` | categorical BOOLEAN | Indicates whether the condition was diagnosed 18 months or more before the reference time point, corresponding… | false, true | 0% | n/a (no nulls) |
| `encounter_primary_reason_HF_Disease_f5a_w7d_first` | categorical BOOLEAN | Whether the patient is hospitalized because of heart failure after the discharge | Missing, false, true | 97% | explicit 'Missing' category |
| `encounter_primary_reason_HF_Disease_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of heart failure after the discharge | Missing, false, true | 90% | explicit 'Missing' category |
| `encounter_primary_reason_HF_Disease_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of heart failure after the discharge | Missing, false, true | 82% | explicit 'Missing' category |
| `encounter_primary_reason_HF_Disease_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of heart failure after the discharge | Missing, false, true | 77% | explicit 'Missing' category |
| `encounter_primary_reason_HF_Disease_f5a_w1a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of heart failure after the discharge | Missing, false, true | 73% | explicit 'Missing' category |
| `encounter_primary_reason_HF_Disease_f5a_w3a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of heart failure after the discharge | Missing, false, true | 68% | explicit 'Missing' category |
| `encounter_primary_reason_HF_Disease_f5a_w5a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of heart failure after the discharge | Missing, false, true | 67% | explicit 'Missing' category |
| `encounter_primary_reason_number_of_HF_rehospitalizations_5a_f5a_count` | numeric NUMERIC | Total number of hospitalizations due to heart failure within 5 years after the reference/index event | 0.0 .. 1.0 | 0% | n/a (no nulls) |
| `encounter_primary_reason_CV_Disease_f5a_w7d_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any cardiovascular condition after the discharge | Missing, false, true | 97% | explicit 'Missing' category |
| `encounter_primary_reason_CV_Disease_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any cardiovascular condition after the discharge | Missing, false, true | 90% | explicit 'Missing' category |
| `encounter_primary_reason_CV_Disease_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any cardiovascular condition after the discharge | Missing, false, true | 82% | explicit 'Missing' category |
| `encounter_primary_reason_CV_Disease_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any cardiovascular condition after the discharge | Missing, false, true | 77% | explicit 'Missing' category |
| `encounter_primary_reason_CV_Disease_f5a_w1a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any cardiovascular condition after the discharge | Missing, false, true | 73% | explicit 'Missing' category |
| `encounter_primary_reason_CV_Disease_f5a_w3a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any cardiovascular condition after the discharge | Missing, false, true | 68% | explicit 'Missing' category |
| `encounter_primary_reason_CV_Disease_f5a_w5a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any cardiovascular condition after the discharge | Missing, false, true | 67% | explicit 'Missing' category |
| `encounter_primary_reason_number_of_days_to_rehosp_for_CV_f5a_first` | numeric NUMERIC | Number of days from discharge until rehospitalization due to CV disease | 1.0 .. 1500.0 | 94% | no event occurred (structural) |
| `encounter_primary_reason_number_of_CV_rehospitalizations_5a_f5a_count` | numeric NUMERIC | Total number of hospitalizations due to cardiovascular disease within 5 years after the reference/index event | 0.0 .. 6.0 | 0% | n/a (no nulls) |
| `encounter_primary_reason_non_CV_Disease_f5a_w7d_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any non-cardiovascular condition after the discharge | Missing, false, true | 97% | explicit 'Missing' category |
| `encounter_primary_reason_non_CV_Disease_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any non-cardiovascular condition after the discharge | Missing, false, true | 90% | explicit 'Missing' category |
| `encounter_primary_reason_non_CV_Disease_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any non-cardiovascular condition after the discharge | Missing, false, true | 82% | explicit 'Missing' category |
| `encounter_primary_reason_non_CV_Disease_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any non-cardiovascular condition after the discharge | Missing, false, true | 77% | explicit 'Missing' category |
| `encounter_primary_reason_non_CV_Disease_f5a_w1a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any non-cardiovascular condition after the discharge | Missing, false, true | 73% | explicit 'Missing' category |
| `encounter_primary_reason_non_CV_Disease_f5a_w3a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any non-cardiovascular condition after the discharge | Missing, false, true | 68% | explicit 'Missing' category |
| `encounter_primary_reason_non_CV_Disease_f5a_w5a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any non-cardiovascular condition after the discharge | Missing, false, true | 67% | explicit 'Missing' category |
| `encounter_primary_reason_number_of_days_to_rehosp_for_non_CV_f5a_first` | numeric NUMERIC | Number of days from discharge until rehospitalization due to non-CV disease | 1.0 .. 1800.0 | 72% | no event occurred (structural) |
| `encounter_primary_reason_number_of_non_CV_rehospitalizations_5a_f5a_count` | numeric NUMERIC | Total number of hospitalizations due to non-cardiovascular disease within 5 years after the reference/index ev… | 0.0 .. 39.0 | 0% | not measured |
| `encounter_primary_reason_renal_complications_f5a_w7d_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any renal complication after the discharge | Missing, false | 97% | explicit 'Missing' category |
| `encounter_primary_reason_renal_complications_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any renal complication after the discharge | Missing, false | 90% | explicit 'Missing' category |
| `encounter_primary_reason_renal_complications_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any renal complication after the discharge | Missing, false | 82% | explicit 'Missing' category |
| `encounter_primary_reason_renal_complications_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any renal complication after the discharge | Missing, false | 77% | explicit 'Missing' category |
| `encounter_primary_reason_renal_complications_f5a_w1a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any renal complication after the discharge | Missing, false, true | 73% | explicit 'Missing' category |
| `encounter_primary_reason_renal_complications_f5a_w3a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any renal complication after the discharge | Missing, false, true | 68% | explicit 'Missing' category |
| `encounter_primary_reason_renal_complications_f5a_w5a_first` | categorical BOOLEAN | Whether the patient is hospitalized because of any renal complication after the discharge | Missing, false, true | 67% | explicit 'Missing' category |
| `encounter_primary_reason_number_of_renal_rehospitalizations_5a_f5a_count` | numeric NUMERIC | Total number of hospitalizations due to renal complications within 5 years after the reference/index event | 0.0 .. 2.0 | 0% | n/a (no nulls) |
| `cause_of_death_isCV_f5a_w7d_first` | categorical BOOLEAN | Whether the patient dies because of any cardiovascular condition after the discharge | Missing, false | 99% | explicit 'Missing' category |
| `cause_of_death_isCV_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient dies because of any cardiovascular condition after the discharge | Missing, false | 94% | explicit 'Missing' category |
| `cause_of_death_isCV_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient dies because of any cardiovascular condition after the discharge | Missing, false | 88% | explicit 'Missing' category |
| `cause_of_death_isCV_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient dies because of any cardiovascular condition after the discharge | Missing, false | 83% | explicit 'Missing' category |
| `cause_of_death_isCV_f5a_w1a_first` | categorical BOOLEAN | Whether the patient dies because of any cardiovascular condition after the discharge | Missing, false | 75% | explicit 'Missing' category |
| `cause_of_death_isCV_f5a_w3a_first` | categorical BOOLEAN | Whether the patient dies because of any cardiovascular condition after the discharge | Missing, false | 64% | explicit 'Missing' category |
| `cause_of_death_isCV_f5a_w5a_first` | categorical BOOLEAN | Whether the patient dies because of any cardiovascular condition after the discharge | Missing, false | 61% | explicit 'Missing' category |
| `cause_of_death_isRenal_f5a_w7d_first` | categorical BOOLEAN | Whether the patient dies because of any renal complication after the discharge | Missing, false | 99% | explicit 'Missing' category |
| `cause_of_death_isRenal_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient dies because of any renal complication after the discharge | Missing, false | 94% | explicit 'Missing' category |
| `cause_of_death_isRenal_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient dies because of any renal complication after the discharge | Missing, false | 88% | explicit 'Missing' category |
| `cause_of_death_isRenal_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient dies because of any renal complication after the discharge | Missing, false | 83% | explicit 'Missing' category |
| `cause_of_death_isRenal_f5a_w1a_first` | categorical BOOLEAN | Whether the patient dies because of any renal complication after the discharge | Missing, false | 75% | explicit 'Missing' category |
| `cause_of_death_isRenal_f5a_w3a_first` | categorical BOOLEAN | Whether the patient dies because of any renal complication after the discharge | Missing, false | 64% | explicit 'Missing' category |
| `cause_of_death_isRenal_f5a_w5a_first` | categorical BOOLEAN | Whether the patient dies because of any renal complication after the discharge | Missing, false | 61% | explicit 'Missing' category |
| `cause_of_death_isNonRenalAndNonCV_f5a_w7d_first` | categorical BOOLEAN | Whether the patient dies because of any non-cardiovascular and non-renal condition after the discharge | Missing, false | 99% | explicit 'Missing' category |
| `cause_of_death_isNonRenalAndNonCV_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient dies because of any non-cardiovascular and non-renal condition after the discharge | Missing, false | 94% | explicit 'Missing' category |
| `cause_of_death_isNonRenalAndNonCV_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient dies because of any non-cardiovascular and non-renal condition after the discharge | Missing, false | 88% | explicit 'Missing' category |
| `cause_of_death_isNonRenalAndNonCV_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient dies because of any non-cardiovascular and non-renal condition after the discharge | Missing, false | 83% | explicit 'Missing' category |
| `cause_of_death_isNonRenalAndNonCV_f5a_w1a_first` | categorical BOOLEAN | Whether the patient dies because of any non-cardiovascular and non-renal condition after the discharge | Missing, false | 75% | explicit 'Missing' category |
| `cause_of_death_isNonRenalAndNonCV_f5a_w3a_first` | categorical BOOLEAN | Whether the patient dies because of any non-cardiovascular and non-renal condition after the discharge | Missing, false | 64% | explicit 'Missing' category |
| `cause_of_death_isNonRenalAndNonCV_f5a_w5a_first` | categorical BOOLEAN | Whether the patient dies because of any non-cardiovascular and non-renal condition after the discharge | Missing, false | 61% | explicit 'Missing' category |
| `cause_of_death_isAllCause_f5a_w7d_first` | categorical BOOLEAN | Whether the patient dies because of unspecified condition after the discharge | Missing, true | 99% | explicit 'Missing' category |
| `cause_of_death_isAllCause_f5a_w1mo_first` | categorical BOOLEAN | Whether the patient dies because of unspecified condition after the discharge | Missing, true | 94% | explicit 'Missing' category |
| `cause_of_death_isAllCause_f5a_w3mo_first` | categorical BOOLEAN | Whether the patient dies because of unspecified condition after the discharge | Missing, true | 88% | explicit 'Missing' category |
| `cause_of_death_isAllCause_f5a_w6mo_first` | categorical BOOLEAN | Whether the patient dies because of unspecified condition after the discharge | Missing, true | 83% | explicit 'Missing' category |
| `cause_of_death_isAllCause_f5a_w1a_first` | categorical BOOLEAN | Whether the patient dies because of unspecified condition after the discharge | Missing, true | 75% | explicit 'Missing' category |
| `cause_of_death_isAllCause_f5a_w3a_first` | categorical BOOLEAN | Whether the patient dies because of unspecified condition after the discharge | Missing, true | 64% | explicit 'Missing' category |
| `cause_of_death_isAllCause_f5a_w5a_first` | categorical BOOLEAN | Whether the patient dies because of unspecified condition after the discharge | Missing, true | 61% | explicit 'Missing' category |
| `cause_of_death_number_of_days_to_death_for_all_cause_f5a_first` | numeric NUMERIC | Number of days from discharge until death due to unspecified condition | 0.0 .. 1800.0 | 61% | no event occurred (structural) |
| `eGFR_2021_ckd_epi_creatinine` | numeric NUMERIC | Estimated glomerular filtration rate (eGFR) calculated from serum creatinine, age and gender using the 2021 CK… | 20.0 .. 140.0 | 0% | n/a (no nulls) |
| `ckd_severity_from_calculated_egfr` | categorical NOMINAL | CKD severity stage derived by categorizing the eGFR calculated with the 2021 CKD-EPI creatinine equation | mild_to_moderate_decrease, mildly_decreased, moderate_to_severe_decrease, normal_or_high, severe_decrease | 0% | n/a (no nulls) |
| `ckd_severity_calculated_or_measured` | categorical NOMINAL | Final CKD severity stage: uses the stage derived from the calculated CKD-EPI eGFR when available, otherwise fa… | mild_to_moderate_decrease, mildly_decreased, moderate_to_severe_decrease, normal_or_high, severe_decrease | 0% | n/a (no nulls) |
| `med_acei` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_activeDuringEncounter_ace_inhibitors_arb_use` | categorical  |  | false | 0% | n/a (no nulls) |
| `med_activeDuringEncounter_bb` | categorical  |  | false | 0% | n/a (no nulls) |
| `med_anti_coag` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_anti_plat` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_antiarrhytmic` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_antiinfl` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_arb` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ari` | categorical  |  | false | 0% | n/a (no nulls) |
| `med_arni` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_bb` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ccb` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_cortico_syst` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_digitalis` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_diuretics` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_diuretics_loop` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_inotropes` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_insulins` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ivabradine` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ll` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_mra` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_oral_antidiabetic` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_platelet` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_potassium_binders` | categorical  |  | false | 0% | n/a (no nulls) |
| `med_rasi` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_rdoad` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_rdoad_syst` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_sglt2i` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_thrombolytic` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_vasodil` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_acei_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_anti_coag_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_anti_plat_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_antiarrhytmic_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_antiinfl_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_arb_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ari_history` | categorical  |  | false | 0% | n/a (no nulls) |
| `med_arni_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_bb_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ccb_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_cortico_syst_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_digitalis_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_diuretics_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_diuretics_loop_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_inotropes_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_insulins_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ivabradine_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_ll_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_mra_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_oral_antidiabetic_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_platelet_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_potassium_binders_history` | categorical  |  | false | 0% | n/a (no nulls) |
| `med_rasi_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_rdoad_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_rdoad_syst_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_sglt2i_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_thrombolytic_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `med_vasodil_history` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_af` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_aidshiv` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_ap` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_ckd_chronic` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_cm` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_copd` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_dem` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_dep` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_devices` | categorical  |  | false | 0% | n/a (no nulls) |
| `conditions_dia` | categorical  |  | false | 0% | n/a (no nulls) |
| `conditions_diabetes` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_dysl` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_hf` | categorical  |  | true | 0% | n/a (no nulls) |
| `conditions_hyp` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_hyperthyroid` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_hypothyroid` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_ibd` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_ihd` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_ld` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_mc` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_mi` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_myocarditis` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_osa` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_pad` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_pericardial` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_rd` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_revasc` | categorical  |  | false | 0% | n/a (no nulls) |
| `conditions_stroke` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_substance_abuse` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_tia` | categorical  |  | false, true | 0% | n/a (no nulls) |
| `conditions_vd` | categorical  |  | false, true | 0% | n/a (no nulls) |
