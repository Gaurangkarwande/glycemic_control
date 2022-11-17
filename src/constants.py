META_COLS = ["subject_id", "stay_id", "hr"]
TARGET_COL = ["glucose"]

CONTINUOUS_COVARIATES = [
    "sofa_24hours",
    "liver_24hours",
    "weight_kg",
    "height_cm",
    "bmi",
    "cns_24hours",
    "cardiovascular_24hours",
    "pre_dexPN",
    "dexPN",
    "pos_dexPN",
    "charlson_comorbidity_index",
    "dex5",
    "dex>5",
    "pos_dex5",
    "pos_dex>5",
    "pre_dex>5",
    "pre_dex5",
    "coagulation_24hours",
    "respiration_24hours",
    "renal_24hours",
    "PN_sa_insulin",
    "IV_sa_insulin",
    "pre_PN_sa_insulin",
    "pre_IV_sa_insulin",
    "pos_PN_sa_insulin",
    "pos_IV_sa_insulin",
    "admission_age",  # is this a categorical confounder ?
]

CATEGORICAL_CONFOUNDERS = [
    # "myocardial_infarct",
    "congestive_heart_failure",
    # "peripheral_vascular_disease",
    # "cerebrovascular_disease",
    # "dementia",
    # "chronic_pulmonary_disease",
    # "rheumatic_disease",
    # "peptic_ulcer_disease",
    # "mild_liver_disease",
    # "diabetes_without_cc",
    # "diabetes_with_cc",
    # "paraplegia",
    "renal_disease",
    "malignant_cancer",
    # "severe_liver_disease",
    # "metastatic_solid_tumor",
    "aids",
    "diabetes",
    "diabetes_type",
    "septic",
    "gender",
    "ethnicity",
]

TUBE_FEEDING_COLS = ["pre_dexPN", "dexPN", "pos_dexPN"]
DEXTROSE_COLS = ["dex5", "dex>5", "pos_dex5", "pos_dex>5", "pre_dex>5", "pre_dex5"]
INSULIN_COLS = [
    "PN_sa_insulin",
    "IV_sa_insulin",
    "pre_PN_sa_insulin",
    "pre_IV_sa_insulin",
    "pos_PN_sa_insulin",
    "pos_IV_sa_insulin",
]

CONTINUOUS_COVARIATES_PROCESSED = [
    "sofa_24hours",
    "liver_24hours",
    "cns_24hours",
    "cardiovascular_24hours",
    "charlson_comorbidity_index",
    "coagulation_24hours",
    "respiration_24hours",
    "renal_24hours",
    "tube_feeding",
    "dextrose",
    "insulin",
]

STATIC_COLS = ["weight_kg", "height_cm", "bmi", "admission_age"]

INPUT_COVARIATES = [
    "sofa_24hours",
    "liver_24hours",
    "weight_kg",
    "height_cm",
    "bmi",
    "cns_24hours",
    "admission_age",
    "cardiovascular_24hours",
    "tube_feeding",
    "charlson_comorbidity_index",
    "dextrose",
    "coagulation_24hours",
    "respiration_24hours",
    "renal_24hours",
    "insulin",
]
