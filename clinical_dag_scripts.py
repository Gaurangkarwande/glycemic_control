import argparse
from enum import Enum
from pathlib import Path

import pandas as pd

from src.constants import INPUT_COVARIATES, TARGET_COL

INPUT_VARIABLES = INPUT_COVARIATES + TARGET_COL


class DagType(str, Enum):
    DIABETIC = "diabetic"
    NON_DIABETIC = "non_diabetic"
    ALL = "all"


def process_excel(fpath_excel: Path, dirpath_save: Path) -> None:
    """Process excel and save the individual sheets

    Args:
        fpath_excel: the path to the excel file
        dirpath_save: the path to where individual csvs will be saved
    """

    excel = pd.ExcelFile(fpath_excel)
    for sheet_name in excel.sheet_names:
        try:
            df_sheet = pd.read_excel(fpath_excel, sheet_name=sheet_name).fillna(0)
            if "Node" in df_sheet.columns:
                df_sheet.set_index("Node", inplace=True)
            elif 'Unnamed: 0' in df_sheet.columns:
                df_sheet.set_index('Unnamed: 0', inplace=True)
            else:
                raise RuntimeError
            df_sheet.index = INPUT_VARIABLES
            df_sheet.columns = INPUT_VARIABLES
            df_sheet.to_csv(dirpath_save.joinpath(sheet_name + ".csv"))
        except Exception as e:
            print(f"Error parsing sheet: {sheet_name}")
            print(e)


def parse_args():
    parser = argparse.ArgumentParser(description="Clinical dags processing")
    parser.add_argument("--fpath_excel", type=Path, required=True, help="path to clinical excel")
    parser.add_argument(
        "--dag_type",
        type=DagType,
        required=True,
        help="the type of DAG - diabetic, non_diabetic or all",
    )
    parser.add_argument(
        "--dirpath_save",
        type=Path,
        required=True,
        help="output directory individual csvs will be stored",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dirpath_save = args.dirpath_save.joinpath(args.dag_type.value)
    dirpath_save.mkdir(exist_ok=False)
    process_excel(args.fpath_excel, dirpath_save)
