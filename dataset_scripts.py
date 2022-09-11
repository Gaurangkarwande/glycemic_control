import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.constants import (
    CONTINUOUS_COVARIATES,
    CATEGORICAL_CONFOUNDERS,
    META_COLS,
    DEXTROSE_COLS,
    INSULIN_COLS,
    TUBE_FEEDING_COLS,
    TARGET_COL,
)
from src.utils import get_timestamp

COLUMN_SUBSET = META_COLS + TARGET_COL + CONTINUOUS_COVARIATES + CATEGORICAL_CONFOUNDERS


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
