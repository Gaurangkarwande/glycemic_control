import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.utils import get_timestamp

META_COLS = [
    "subject_id",
    "stay_id",
    "hr"
    ]
TARGET_COL = ["glucose"]

CONTINUOUS_COVARIATES = [
    "sofa_24hours",
    "liver_24hours",
    "weight_kg",
    "height_cm",
    "bmi",
    "cns_24hours",
    "admission_age",
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
    "admission_age"  # is this a categorical confounder ?
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

COLUMN_SUBSET = META_COLS + TARGET_COL + CONTINUOUS_COVARIATES + CATEGORICAL_CONFOUNDERS

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


def create_base_dataset(df_cohort: pd.DataFrame) -> pd.DataFrame:
    """Create the base dataset from raw cohort"""

    # patient map : {patient_id : [list of stay ids]}
    stay_patient_map = defaultdict(list)
    df_unique_stays = df_cohort[["stay_id", "subject_id"]].drop_duplicates(subset="stay_id")
    for row in df_unique_stays.itertuples():
        stay_patient_map[row.subject_id].append(row.stay_id)

    # select patients with only one ICU stay
    selected_patient_ids = set()
    for key in stay_patient_map:
        if len(stay_patient_map[key]) == 1:
            selected_patient_ids.add(key)

    df = df_cohort[df_cohort.subject_id.isin(selected_patient_ids)]

    # select stays with duration within interquantile range [6, 27]
    # we get these values from box plot of stay counts
    stay_counts_df = df.groupby(["stay_id"])["glucose"].count()
    filtered_stay_count_df = stay_counts_df[(6 <= stay_counts_df) & (stay_counts_df <= 27)]
    stay_id_set = set(filtered_stay_count_df.index)
    filtered_df = df[df.stay_id.isin(stay_id_set)]

    # only keep the columns we are interested in
    raw_covariate_df = filtered_df[COLUMN_SUBSET]

    # derive clinically relevant covariates
    covariate_df = raw_covariate_df.copy()
    covariate_df["tube_feeding"] = raw_covariate_df[TUBE_FEEDING_COLS].mean(axis=1)
    covariate_df["dextrose"] = raw_covariate_df[DEXTROSE_COLS].mean(axis=1)
    covariate_df["insulin"] = raw_covariate_df[INSULIN_COLS].mean(axis=1)
    covariate_df.drop(labels=TUBE_FEEDING_COLS + DEXTROSE_COLS + INSULIN_COLS, inplace=True, axis=1)
    return covariate_df


def parse_args():
    parser = argparse.ArgumentParser(description="Base dataset creation")
    parser.add_argument(
        "--fpath_raw_csv", type=Path, required=True, help="path to raw cohort (currently v2_gsidp)"
    )
    parser.add_argument(
        "--dirpath_out",
        type=Path,
        required=True,
        help="output directory where base dataset will be saved",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    raw_df = pd.read_csv(args.fpath_raw_csv)
    base_df = create_base_dataset(raw_df)
    fpath_out = args.dirpath_out.joinpath(get_timestamp() + "base_dataset.csv")
    base_df.to_csv(fpath_out, index=None)
