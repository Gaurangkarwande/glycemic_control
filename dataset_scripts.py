import argparse
from collections import defaultdict
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.constants import (
    CATEGORICAL_CONFOUNDERS,
    CONTINUOUS_COVARIATES,
    DEXTROSE_COLS,
    INSULIN_COLS,
    META_COLS,
    TARGET_COL,
    TUBE_FEEDING_COLS,
)
from src.utils import get_timestamp

COLUMN_SUBSET = META_COLS + TARGET_COL + CONTINUOUS_COVARIATES + CATEGORICAL_CONFOUNDERS


def create_base_dataset(df_cohort: pd.DataFrame) -> pd.DataFrame:
    """Create the base dataset from raw cohort
    Args:
        df_cohort: the pandas dataframe

    Returns: processed dataframe that will be used as base dataset
    """

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


def train_valid_test_splits_v1(
    df: pd.DataFrame, seed: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset into train valid and test splits. This does patient-wise splitting.
    We split the patients into train, valid and test patients. Underlying assumption each patient
    is a sample is IID.
    Train : valid : test = 60 : 20 : 20

    Args:
        df : the dataframe of all patients
        seed : the random state. Default = 123

    Returns: (train data, valid data, test data)
    """

    patient_ids = df.subject_id.unique()
    train_patients, test_patients = train_test_split(patient_ids, test_size=0.2, random_state=seed)
    train_patients, valid_patients = train_test_split(
        train_patients, test_size=0.25, random_state=seed
    )
    df_train = df[df.subject_id.isin(set(train_patients))]
    df_valid = df[df.subject_id.isin(set(valid_patients))]
    df_test = df[df.subject_id.isin(set(test_patients))]
    return df_train, df_valid, df_test


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


def convert_to_discrete_values(value: int) -> int:
    """Convert continuous glucose range to classification labels

    Args:
        value: the continuous glucose value

    Returns: the label mapping
    """

    if value <= 104:
        label = 0
    elif 104 < value <= 119:
        label = 1
    elif 119 < value <= 136:
        label = 2
    elif 136 < value <= 162:
        label = 3
    else:
        label = 4
    return label


if __name__ == "__main__":
    args = parse_args()
    raw_df = pd.read_csv(args.fpath_raw_csv)
    base_df = create_base_dataset(raw_df)
    fpath_out = args.dirpath_out.joinpath(get_timestamp() + "base_dataset.csv")
    base_df.to_csv(fpath_out, index=None)
