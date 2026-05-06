from __future__ import annotations
import pandas as pd
from typing import Optional
import re
from pathlib import Path


def extract_id(path_like) -> int:
    """
    Extract experiment id from file name
    @ Author: André Lasses Armatowski, Madeleine Skeppås
    """
    s = str(path_like)
    m = re.search(r"exp_(\d+)_", s)
    if not m:
        raise ValueError(
            f"One or more stack files missing experiment id pattern 'exp_<id>_': {s}")
    return int(m.group(1))


def extract_roi_num(path_like) -> int:
    """
    Extract roi id from file name
    @ Author: André Lasses Armatowski, Madeleine Skeppås
    """
    s = str(path_like)
    m = re.search(r"roi_(\d+)", s)
    if not m:
        raise ValueError(
            f"One or more stack files missing roi id pattern 'roi_<id>': {s}")
    return int(m.group(1))


def build_stack_table_flex(
    stackfile: list[Path],
    metadata: pd.DataFrame,
    treatment: Optional[tuple[object, object]],
    control: str,
    patient_id: Optional[list[str]],
    sets_by_patient: Optional[dict[str, list[int]]],
) -> pd.DataFrame:
    """
    Build a stack table from stack file paths and metadata, with optional filtering by patient,
    per-patient set selection, and control/treatment perturbation (optionally dose-specific).

    Parameters
    ----------
    stackfile : list[Path]
        List of stack file paths.
    metadata : pd.DataFrame
        Metadata table keyed by `experiment_id` and containing required columns such as `delta_t`.
        If `patient_id` or `sets_by_patient` are used, metadata must contain `patient_id` (and `set`).
        If control/treatment filtering is used, metadata must contain `perturbation` (and `dose` if dose filtering).
    treatment : Optional[tuple[object, object]]
        Optional (treatment_name, treatment_dose). If dose is None, selects all rows matching treatment_name.
        If dose is not None, filters additionally by metadata column `dose`.
    control : str
        Control perturbation value, or "all" to disable perturbation filtering (treatment must then be None).
    patient_id : Optional[list[str]]
        Optional list of patient identifiers to include.
    sets_by_patient : Optional[dict[str, list[int]]]
        Optional mapping from patient_id -> list of allowed set indices to include for that patient.

    Returns
    -------
    pd.DataFrame
        Filtered stack table with columns renamed to include `exp` and `roi`, plus:
          - `is_treatment` boolean flag
          - `treatment_name` and `treatment_dose` when treatment is provided
        The returned table is deduplicated and index-reset.

    @ Author: André Lasses Armatowski, Madeleine Skeppås
    """

    # ------------------------------
    # 0) Metadata: selection checks
    # ------------------------------
    if patient_id is not None:
        if "patient_id" not in metadata.columns:
            raise ValueError(
                "metadata must contain column 'patient_id' when patient_id is provided"
            )

        available_pids = set(
            metadata["patient_id"].dropna().astype(str).unique())
        missing = set(map(str, patient_id)) - available_pids
        if missing:
            raise ValueError(
                f"patient_id contains values not present in metadata['patient_id']: {sorted(missing)}"
            )

    if sets_by_patient is not None:
        if "patient_id" not in metadata.columns:
            raise ValueError(
                "metadata must contain column 'patient_id' when sets_by_patient is provided"
            )
        if "set" not in metadata.columns:
            raise ValueError(
                "metadata must contain a set column 'set' when sets_by_patient is provided"
            )

        set_col = "set"

        available_pids = set(
            metadata["patient_id"].dropna().astype(str).unique())
        missing_keys = set(map(str, sets_by_patient.keys())) - available_pids
        if missing_keys:
            raise ValueError(
                f"sets_by_patient has patient keys not present in metadata['patient_id']: {sorted(missing_keys)}"
            )

        for pid, sets in sets_by_patient.items():
            pid = str(pid)
            existing_sets = set(
                metadata.loc[metadata["patient_id"].astype(
                    str) == pid, set_col]
                .dropna()
                .astype(int, errors="ignore")
                .tolist()
            )
            missing_sets = set(sets) - existing_sets
            if missing_sets:
                raise ValueError(
                    # FIX: f-string referenced {set_col} but set_col was undefined; now it is defined above
                    f"sets_by_patient[{pid}] contains sets not present for that patient in metadata['{set_col}']: {sorted(missing_sets)}"
                )

    # --------------------------------
    # 1) Build stacktable and merge meta
    # --------------------------------
    df = pd.DataFrame({"file_path": [str(p) for p in stackfile]})
    df["experiment_id"] = df["file_path"].apply(extract_id)
    df["roi_id"] = df["file_path"].apply(extract_roi_num)

    stacktable = df.merge(metadata, on="experiment_id", how="left")

    # 1) delta_t must exist
    if "delta_t" not in stacktable.columns:
        raise ValueError("metadata is missing required column 'delta_t'")

    # 2) missing experiments
    missing_ids = stacktable.loc[stacktable["delta_t"].isna(
    ), "experiment_id"].unique()
    if len(missing_ids) > 0:
        raise ValueError(
            f"One or more experiment_id values in stack files were not found in metadata: {missing_ids.tolist()}"
        )

    # 3) simple numeric + >0 check
    dt = pd.to_numeric(stacktable["delta_t"],
                       errors="coerce")  # non-numeric -> NaN
    if dt.isna().any():
        raise ValueError("delta_t must be numeric for all rows")

    if (dt <= 0).any():
        raise ValueError("delta_t must be > 0 for all rows")

    stacktable["delta_t"] = dt.astype(float)

    # ------------------------------
    # 2) Filter by patient_id
    # ------------------------------
    if patient_id is not None:
        stacktable = stacktable[stacktable["patient_id"].isin(
            patient_id)].copy()

    # ----------------------------------------
    # 3) Filter by sets_by_patient
    # ----------------------------------------
    if sets_by_patient is not None:
        keep_mask = pd.Series(True, index=stacktable.index)
        for pid, allowed_sets in sets_by_patient.items():
            pid_mask = stacktable["patient_id"].astype(str) == str(pid)
            keep_mask.loc[pid_mask] = stacktable.loc[pid_mask,
                                                     "set"].isin(allowed_sets)

        stacktable = stacktable.loc[keep_mask].copy()

    # --------------------------------
    # 4) Filter by control / treatment
    # --------------------------------
    if (control != "all") or (treatment is not None):
        if "perturbation" not in stacktable.columns:
            raise ValueError(
                "metadata is missing required column 'perturbation'")

    # If control == "all", no control filtering
    if isinstance(control, str) and control == "all":
        if treatment is not None:
            raise ValueError(
                "treatment was specified, but Perturbation='all' means no contorl specified filtering. "
                "Specify explicit control perturbations instead of 'all'."
            )
        subtable = stacktable.copy()
    else:
        controls = [control]
        subtable = stacktable[stacktable["perturbation"].isin(controls)].copy()

    subtable["is_treatment"] = False

    if treatment is not None:
        treat_name, _ = treatment
        if not isinstance(control, str) or control.strip() == "":
            raise ValueError(
                "control must be a non-empty string when treatment is specified")

        if str(treat_name).strip() == str(control).strip():
            raise ValueError(
                f"treatment perturbation '{str(treat_name).strip()}' cannot equal control perturbation '{str(control).strip()}'."
            )

    # Add treatment rows
    if treatment is not None:
        treat_name, treat_dose = treatment
        treat_mask = (
            stacktable["perturbation"].astype(str).str.strip()
            == str(treat_name).strip()
        )

        if treat_dose is not None:
            if "dose" not in stacktable.columns:
                raise ValueError(
                    "metadata is missing required column 'dose' to filter treatment dose"
                )
            treat_mask &= (
                stacktable["dose"].astype(str).str.strip()
                == str(treat_dose).strip()
            )

        treat_rows = stacktable.loc[treat_mask].copy()
        treat_rows["is_treatment"] = True
        treat_rows["treatment_name"] = str(treat_name).strip()
        treat_rows["treatment_dose"] = treat_dose

        subtable = (
            pd.concat([subtable, treat_rows], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
    else:
        subtable = subtable.reset_index(drop=True)

    if treatment is not None:
        n_control = int((subtable["is_treatment"] == False).sum())
        n_treat = int((subtable["is_treatment"] == True).sum())
        if n_control == 0:
            raise ValueError(
                "treatment was specified, but zero control samples were selected (check your control perturbation filter)."
            )
        if n_treat == 0:
            raise ValueError(
                "treatment was specified, but zero treatment samples were selected (check perturbation/dose filters)."
            )

    # ---------------------------------------------------------
    # 5) Fail if any selected patient has zero selected rows
    # ---------------------------------------------------------
    if patient_id is not None:
        requested = set(patient_id)
        present = set(subtable["patient_id"].dropna().astype(str).unique())
        missing = requested - present
        if missing:
            raise ValueError(
                f"Missing patients after filtering for control and treatment: {sorted(missing)}"
            )

    return subtable.rename(columns={"experiment_id": "exp", "roi_id": "roi"})
