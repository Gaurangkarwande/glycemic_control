from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from src.utils import find_hyper_glycemia_hours, find_hypo_glycemia_hours


def visualize_patient_data(
    df_data: pd.DataFrame,
    covariates: List[str],
    shade_hyperglycemia: bool = False,
    shade_hypoglycemia: bool = False,
) -> Tuple[plt.figure, Dict[str, int]]:
    """Visually summarize the patient data

    Args:
        df_data (pd.DataFrame) : the dataframe holding the time series observations for a patient
        covariates (List[str]) : list of (continuous) covariates to be included
        shade_hyperglycemia (bool) : shade regions of hyperglycemia
        shade_hypoglycemia (bool) : shade regions of hypoglycemia

    Returns:
        plt.figure : The patient plot
        Tuple[plt.figure, Dict[str, int]] : the static patient data
    """

    covariates_to_plot = []
    static_patient_info = {}
    ret_figure = None
    for covariate in covariates:
        unique_vals = df_data[covariate].unique()
        if len(unique_vals) > 1:
            covariates_to_plot.append(covariate)
        else:
            static_patient_info[covariate] = unique_vals[0]
    if len(covariates_to_plot) > 0:
        ret_figure, ax_covariates, ax_glucose = plot_covariates_with_glucose(
            df_data, covariates_to_plot
        )
        if shade_hyperglycemia:
            for hour_range in find_hyper_glycemia_hours(df_data[["glucose", "hr"]]):
                ax_covariates.axvspan(hour_range[0], hour_range[1], color="r", alpha=0.05)
                ax_glucose.axvspan(hour_range[0], hour_range[1], color="r", alpha=0.05)
        if shade_hypoglycemia:
            for hour_range in find_hypo_glycemia_hours(df_data[["glucose", "hr"]]):
                ax_covariates.axvspan(hour_range[0], hour_range[1], color="b", alpha=0.05)
                ax_glucose.axvspan(hour_range[0], hour_range[1], color="b", alpha=0.05)
    for ax in ret_figure.get_axes():
        ax.legend()

    return ret_figure, static_patient_info


def plot_covariates_with_glucose(df_data: pd.DataFrame, covariates: List[str]) -> plt.axes:
    """Plots time series

    Args:
        df_data (pd.DataFrame) : the dataframe holding the time series observations.
            Should have all covariates + 'hr' + 'glucose' columns
        covariates (List[str]) : column names to plot

    Returns:
        plt.axes : The patient plot
    """

    fig, (ax_glucose, ax_covariates) = plt.subplots(2, figsize=(15, 10), sharex=True)

    df_data.plot(x="hr", y="glucose", ax=ax_glucose, color="r", marker="o", legend=False)
    ax_glucose.set_ylabel("Glucose (mg/DL")
    ax_glucose.set_xlabel("Time (Hours)")

    df_data.plot(x="hr", y=covariates, ax=ax_covariates, legend=False)
    ax_covariates.set_xlabel("Time (Hours)")
    ax_covariates.set_ylabel("Covariates")

    for ax in fig.get_axes():
        ax.label_outer()

    return fig, ax_covariates, ax_glucose
