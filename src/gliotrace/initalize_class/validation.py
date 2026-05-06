from pathlib import Path
from numbers import Real

import pandas as pd

from gliotrace.initalize_class.defaults import DEFAULT_HMM, DEFAULT_CHANNEL, DEFAULT_COL
from gliotrace.initalize_class.types import ValidatedInit


def merge_config(defaults: dict, overrides: dict | None, name="config"):
    cfg = defaults.copy()
    if overrides is None:
        return cfg
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise ValueError(
            f"Invalid {name} key(s): {sorted(unknown)}. Allowed: {sorted(defaults)}")
    cfg.update(overrides)
    return cfg


def _validate_npz_file(path: Path, label: str) -> Path:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Invalid stack file (not .npz): {path} ({label})")
    if not path.exists():
        raise FileNotFoundError(f"Stack file does not exist: {path} ({label})")
    if not path.is_file():
        raise ValueError(f"Stack path is not a file: {path} ({label})")
    return path


def validate_stackfile(obj: list | str | Path) -> list[Path]:
    """
    Accepts either:
      - list of .npz paths
      - path to a .txt file containing .npz paths (one per line; tabs allowed)

    Returns:
      (validated_paths, base_dir_if_txt_else_None)
    """
    # Case 1: obj is a .txt list file
    if isinstance(obj, (str, Path)):
        txt_path = Path(obj).expanduser()

        if txt_path.suffix.lower() != ".txt":
            raise ValueError("stackfile path must be a .txt file")
        if not txt_path.exists():
            raise FileNotFoundError(
                f"stackfile list file does not exist: {txt_path}")
        if not txt_path.is_file():
            raise ValueError(f"stackfile list path is not a file: {txt_path}")

        base_dir = txt_path.parent

        raw_lines = txt_path.read_text(encoding="utf-8").splitlines()
        items: list[Path] = []

        for lineno, line in enumerate(raw_lines, start=1):
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # If the file is TSV then take first column as path
            path_str = s.split("\t", 1)[0].strip().strip('\'"')
            if not path_str:
                continue

            p = Path(path_str).expanduser()
            if not p.is_absolute():
                p = base_dir / p

            items.append(_validate_npz_file(p, f"{txt_path.name}:{lineno}"))

        if not items:
            raise ValueError(f"No valid .npz paths found in: {txt_path}")

        return items

    # Case 2: obj is a list of .npz paths
    if isinstance(obj, list):
        if len(obj) == 0:
            raise ValueError("stackfile list cannot be empty")

        items: list[Path] = []
        for i, p in enumerate(obj):
            try:
                path = Path(p).expanduser()
            except TypeError:
                raise TypeError(f"stackfile[{i}] is not path-like")

            items.append(_validate_npz_file(path, f"stackfile[{i}]"))

        return items

    raise TypeError(
        "stackfile must be a path to a .txt file or a list of .npz paths")


def _validate_metadata_columns(metadata: pd.DataFrame) -> None:
    required = {"experiment_id", "delta_t"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(
            f"Missing required metadata columns: {sorted(missing)}")

    if metadata["experiment_id"].isna().any() or metadata["delta_t"].isna().any():
        raise ValueError(
            "metadata contains NA in one or more required columns: experiment_id, delta_t")

    try:
        dt = pd.to_numeric(metadata["delta_t"], errors="raise").astype(float)
    except Exception as e:
        raise ValueError(
            f"metadata column delta_t must be convertible to float ({e})") from e

    if (dt <= 0).any():
        raise ValueError("metadata column delta_t must be > 0 for all rows")


def validate_metadata(metadata) -> pd.DataFrame:
    if isinstance(metadata, pd.DataFrame):
        if metadata.empty:
            raise ValueError("metadata DataFrame is empty")
        _validate_metadata_columns(metadata)
        return metadata

    if metadata is None:
        raise TypeError("metadata is required")

    try:
        p = Path(metadata).expanduser()
    except TypeError:
        raise TypeError(
            "metadata must be a path-like object (str/Path) or a pandas DataFrame")

    if p.suffix.lower() != ".csv":
        raise ValueError(f"metadata must be a .csv file: {p}")
    if not p.exists():
        raise FileNotFoundError(f"metadata file does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"metadata path is not a file: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"metadata file is empty (0 bytes): {p}")

    try:
        df = pd.read_csv(p)
    except Exception as e:
        raise ValueError(f"metadata CSV could not be read: {p} ({e})") from e

    if df.empty:
        raise ValueError(f"metadata CSV has no rows: {p}")

    _validate_metadata_columns(df)

    return df


def validate_detection_sensitivity(x) -> float:
    if not isinstance(x, Real):
        raise TypeError("detection_sensitivity must be a number")
    x = float(x)
    if not (0 <= x <= 1):
        raise ValueError("detection_sensitivity must be in [0, 1]")
    return x


def validate_channel_roles(channel_roles) -> dict:
    if channel_roles is None:
        return DEFAULT_CHANNEL.copy()

    if not isinstance(channel_roles, dict):
        raise TypeError("channel_roles must be a dict")

    allowed_keys = {"blue", "green", "red"}
    unknown = set(channel_roles) - allowed_keys
    if unknown:
        raise ValueError(f"Invalid channel_roles keys: {sorted(unknown)}")

    cfg = merge_config(DEFAULT_CHANNEL, channel_roles, name="channel_roles")

    allowed_vals = {"none", "gbm", "vasc"}
    vals = list(cfg.values())

    bad_vals = sorted({v for v in vals if v not in allowed_vals})
    if bad_vals:
        raise ValueError(
            f"Invalid channel_roles values: {bad_vals} (allowed: {sorted(allowed_vals)})")

    # Must assign exactly one of each role across blue/green/red
    if sorted(vals) != ["gbm", "none", "vasc"]:
        raise ValueError(
            "channel_roles must assign exactly one 'none', one 'gbm', and one 'vasc'")

    return cfg


def validate_fcols(fcols) -> list[str]:
    allowed = set(DEFAULT_COL)
    if fcols is None:
        return list(DEFAULT_COL)  # copy

    if not isinstance(fcols, list):
        raise TypeError("fcols must be a list")

    unknown = set(fcols) - allowed
    if unknown:
        raise ValueError(
            f"Invalid feature column(s): {sorted(unknown)}. Allowed: {sorted(allowed)}")

    return list(fcols)  # copy user list too (prevents outside mutation)


def validate_hmm_param(hmm_param) -> dict:
    if hmm_param is None:
        return DEFAULT_HMM.copy()

    if not isinstance(hmm_param, dict):
        raise TypeError("hmm_param must be a dict")

    cfg = merge_config(DEFAULT_HMM, hmm_param, name="hmm_param")

    if cfg["em_iter"] <= 0:
        raise ValueError("em_iter must be > 0")
    if cfg["glm_iter"] <= 0:
        raise ValueError("glm_iter must be > 0")
    if cfg["eps"] <= 0:
        raise ValueError("eps must be > 0")

    return cfg


def validate_treatment(treatment) -> list | None:
    """
    treatment must be either:
      - None
      - [treatment_name, dosage]
        where:
          - treatment_name is a string (non-empty after stripping)
          - dosage is either a string (non-empty after stripping) or None

    Returns normalized [treatment_name, dosage] or None.
    """
    if treatment is None:
        return None

    if not isinstance(treatment, (list, tuple)):
        raise TypeError(
            "treatment must be None or a 2-element list/tuple: [treatment_name, dosage]")

    if len(treatment) != 2:
        raise ValueError(
            "treatment must have exactly 2 elements: [treatment_name, dosage]")

    name, dosage = treatment

    if not isinstance(name, str):
        raise TypeError("treatment_name must be a string")

    name = name.strip()
    if name == "":
        raise ValueError("treatment_name cannot be an empty string")

    if dosage is None:
        return [name, None]

    if not isinstance(dosage, str):
        raise TypeError("dosage must be a string or None")

    dosage = dosage.strip()
    if dosage == "":
        raise ValueError("dosage cannot be an empty string (use None instead)")

    return [name, dosage]


def validate_perturbations(p) -> str:
    if not isinstance(p, str):
        raise TypeError("perturbations must be a string")
    return p


def validate_patient_id_sets(patients, sets_by_patient):
    # patients: None => all patients (no filter)
    if patients is not None:
        if not isinstance(patients, list):
            raise TypeError("patients must be a list[str] or None")
        if not all(isinstance(p, str) and p for p in patients):
            raise TypeError("patients must be a list of non-empty strings")
        patients_set = set(patients)
    else:
        patients_set = None

    sets_map = None
    if sets_by_patient is not None:
        if not isinstance(sets_by_patient, dict):
            raise TypeError(
                "sets_by_patient must be a dict[str, list[int]] or None")

        sets_map = {}
        for pid, sets in sets_by_patient.items():
            if not isinstance(pid, str) or not pid:
                raise TypeError(
                    "sets_by_patient keys must be non-empty strings (patient ids)")
            if not isinstance(sets, list) or not sets:
                raise TypeError(
                    f"sets_by_patient['{pid}'] must be a non-empty list[int]")
            if not all(isinstance(s, int) for s in sets):
                raise TypeError(
                    f"sets_by_patient['{pid}'] must contain only ints")
            sets_map[pid] = set(sets)

        # Consistency check: dict keys must be within patients if patients is specified
        if patients_set is not None:
            extra = set(sets_map.keys()) - patients_set
            if extra:
                raise ValueError(
                    f"sets_by_patient specifies patients not in patients: {sorted(extra)}"
                )

    return patients_set, sets_map


def _validate_mode(mode: str):
    agg_map = {
        "mean": "mean",
        "median": "median",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "std": "std",
        "count": "count",
    }
    if mode not in agg_map:
        raise ValueError(
            f"Unsupported mode='{mode}' use one of: {sorted(agg_map)}")
    return agg_map[mode]


def _filter_patient(df, patient_id):
    if patient_id is not None:
        if "patient_id" not in df.columns:
            raise ValueError(
                "patient_id was provided, but data has no 'patient_id' column")
        avail = set(df["patient_id"].dropna().astype(str).unique())
        pid = str(patient_id)
        if pid not in avail:
            raise ValueError(
                f"patient_id={patient_id} not found in data. Available: {sorted(avail)}")
        return df[df["patient_id"].astype(str) == pid]
    else:
        if "patient_id" in df.columns:
            raise ValueError(
                "data has 'patient_id' column, so you must specify patient_id")
        return df


def _filter_set(df, set_id, patient_id):
    if set_id is not None:
        if "set" not in df.columns:
            raise ValueError("set was provided, but data has no 'set' column")
        avail = set(df["set"].dropna().unique())
        sid = int(set_id)
        if sid not in avail:
            raise ValueError(
                f"set={set_id} not found for patient_id={patient_id}. "
                f"Available sets for this patient: {sorted(avail)}"
            )
        return df[df["set"] == sid]
    else:
        if "set" in df.columns:
            raise ValueError(
                f"data has 'set' column, so you must specify set (patient_id={patient_id})")
        return df


def _filter_treatment(df, treatment, patient_id, set_id):
    if treatment is None:
        return df
    want = bool(treatment)
    avail = set(df["is_treatment"].dropna().astype(bool).unique())
    if want not in avail:
        raise ValueError(
            f"is_treatment={want} not available for patient_id={patient_id}, set={set_id}. "
            f"Available is_treatment values here: {sorted(avail)}"
        )
    return df[df["is_treatment"] == want]


def _require_cols(df: pd.DataFrame, *cols: str, name: str = "subtable") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} has no {missing!r} column(s)")


def _apply_treated(df: pd.DataFrame, treated: bool | None) -> pd.DataFrame:
    if treated is None:
        return df
    return df[df["is_treatment"] == bool(treated)]


def _apply_patient(df: pd.DataFrame, patient: str | None, *, name: str) -> pd.DataFrame:
    if patient is None:
        return df
    _require_cols(df, "patient_id", name=name)
    out = df[df["patient_id"].astype(str) == str(patient)]
    if out.empty:
        raise ValueError(f"Unknown patient_id={patient}")
    return out


def _apply_set(df: pd.DataFrame, set_id: int | None, *, patient: str | None, name: str) -> pd.DataFrame:
    if set_id is None:
        return df
    if patient is None:
        raise ValueError(
            "set was provided but patient is None; set is only valid within a patient")
    _require_cols(df, "set", name=name)
    out = df[df["set"] == int(set_id)]
    if out.empty:
        raise ValueError(
            f"No experiments for patient_id={patient} and set={set_id}")
    return out


def _validate_exp_roi(subtable, exp, roi):
    try:
        exp = int(exp)
    except Exception:
        raise TypeError("exp must be int-like")

    try:
        roi = int(roi)
    except Exception:
        raise TypeError("roi must be int-like")

    exp_list = sorted(set(subtable["exp"].astype(int).unique()))
    if exp not in exp_list:
        raise ValueError(f"exp={exp} not found. Available exp: {exp_list}")

    roi_list = sorted(
        set(
            subtable.loc[subtable["exp"].astype(int) == exp, "roi"]
            .astype(int)
            .unique()
        )
    )
    if roi not in roi_list:
        raise ValueError(
            f"roi={roi} not found for exp={exp}. Available roi: {roi_list}")

    return exp, roi


def validate_init(
    stackfile,
    metadata,
    detection_sensitivity=0.0,
    channel_roles=None,
    fcols=None,
    hmm_param=None,
    control=None,
    patient_id=None,
    sets_by_patient=None,
    treatment=None
):
    patient_id, sets_by_patient = validate_patient_id_sets(
        patient_id, sets_by_patient)

    return ValidatedInit(
        stackfiles=validate_stackfile(stackfile),
        metadata=validate_metadata(metadata),
        detection_sensitivity=validate_detection_sensitivity(
            detection_sensitivity),
        channel_roles=validate_channel_roles(channel_roles),
        fcols=validate_fcols(fcols),
        hmm_param=validate_hmm_param(hmm_param),
        control=validate_perturbations(control),
        patient_id=patient_id,
        sets_by_patient=sets_by_patient,
        treatment=validate_treatment(treatment),
    )
