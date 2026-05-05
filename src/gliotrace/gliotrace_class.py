from gliotrace.build_tracks_and_vascularity.build_tracks_and_vascularity import build_tracks_and_vascularity
from gliotrace.build_tracks_and_vascularity.weights_and_models.load_networks import load_trained_networks
from gliotrace.visualize.generate_video_compare import generate_video_compare
from gliotrace.feature_and_hmm_pipline.hmm_pipeline import hmm_pipeline
from gliotrace.initalize_class.load_data import build_stack_table_flex
from gliotrace.visualize.generate_video import generate_video
from gliotrace.initalize_class.validation import validate_init, validate_fcols, validate_hmm_param, validate_detection_sensitivity, _filter_patient, _filter_set, _filter_treatment, _validate_mode, _apply_patient, _apply_set, _require_cols, _apply_treated, _validate_exp_roi
from gliotrace.initalize_class.defaults import SOFTMAX_COLUMNS

from gliotrace.visualize.preprocess_stack import prepare_gbm_vasc_arrays


from typing import Optional
from pathlib import Path
from tqdm import tqdm

import pandas as pd
import json
import numpy as np
import joblib

PathLike = str | Path


class GlioTrace:
    def __init__(
        self,
        stackfile,
        metadata,
        detection_sensitivity: float = 0.2,
        channel_roles: dict[str, str] | None = None,
        fcols: list[str] | None = None,
        hmm_param: dict[str, int | float] = None,
        control: str = "all",
        patient_id: Optional[list[str]] = None,
        sets_by_patient: Optional[dict[str, list[str]]] = None,
        treatment: list[str] = None,
    ):
        """
        Initialize a GlioTrace analysis run.

        This constructor validates inputs, builds an internal "stack table" describing all
        stacks to be processed, initializes configuration for detection and GLM-HMM fitting,
        and resets all run-state flags and outputs.

        Parameters
        ----------
        stackfile
            Stack file(s) to analyze.

        metadata
            Metadata used to annotate each stack (e.g., experimental IDs, delta t, patient_id, sets, perturbation).

        detection_sensitivity
            Detection sensitivity for the tracking pipeline (default ``0.2``).

        channel_roles
            Optional mapping from channel keys to semantic roles used by the pipeline,
            e.g. ``{"green": "gbm", "red": "vasc"}``.

        fcols
            Optional list of feature column names used when building the GLM-HMM feature/design
            matrix.

        hmm_param
            Optional dictionary of GLM-HMM hyperparameters/configuration

        control
            Control selection/grouping specifier (default ``"all"``).

        patient_id
            Optional list of patient identifiers used to annotate stacks and/or define grouping

        sets_by_patient
            Optional mapping ``{patient_id: [set_id, ...]}`` used to assign stacks to sets per
            patient. 

        treatment
            Optional list defining treatment label for stacks.

        @ Author: André Lasses Armatowski
        """
        v = validate_init(
            stackfile=stackfile,
            metadata=metadata,
            detection_sensitivity=detection_sensitivity,
            channel_roles=channel_roles,
            fcols=fcols,
            hmm_param=hmm_param,
            control=control,
            patient_id=patient_id,
            sets_by_patient=sets_by_patient,
            treatment=treatment,
        )

        # Build stacktable containing all informaiton about the run data
        self._subtable = build_stack_table_flex(
            v.stackfiles,
            v.metadata,
            v.treatment,
            v.control,
            v.patient_id,
            v.sets_by_patient)

        # Start off with all stacks being untracked
        self._subtable["tracked"] = False

        # Store validated config
        self._detection_sensitivity = v.detection_sensitivity
        self._channel_roles = v.channel_roles
        self._fcols = v.fcols
        self._hmm_param = v.hmm_param

        # Internal state
        self._tracked = False
        self._hmm = False

        # Keep track of output paths
        self._video_paths = []

        # Empty track data
        self._track_data = None

        # Print the configuration initalizing the model
        self.print_configuration()

    # ------------------------------------------------------------------
    # Main Computational Pipline
    # ------------------------------------------------------------------

    def run_tracking(
        self,
        save_point=None,
        load_point=None,
        detection_sensitivity=None,
        detection_backup=None,
    ):
        """
        Run the cell tracking + vascularity/classification pipeline over all untracked stacks.

        Parameters
        ----------
        save_point
            Optional identifier/path used for checkpointing.

        load_point
            Optional identifier/path used to reload a previously saved run.

        detection_sensitivity
            Optional detection sensitivity used by the tracking pipeline. If provided, it is
            validated and stored in ``self._detection_sensitivity``.  If ``None``, the current 
            ``self._detection_sensitivity`` is used.

        detection_backup
            Optional fallback detection sensitivity. If provided, it is validated.
            If the main tracking attempt returns ``None``
            for a stack, the pipeline is re-run once using ``detection_backup``.


        @ Author: André Lasses Armatowski, Madeleine Skeppås
        """

        # First check if the user wants to reload tracking for some point
        lp = load_point
        if lp is not None:
            loaded = self.__class__.load_run(lp)

            # compare your current initialized subtable/stacktable vs loaded one
            self._assert_same_rows_except_tracked(
                self._subtable, loaded.subtable)

            self.__dict__.update(loaded.__dict__)

        # If the loaded point has already been fully tracked
        self._tracked = np.all(self._subtable["tracked"] == True)
        if self._tracked:
            print("All stacks have already been tracked")

        # Update the detection sensitivity:
        if detection_sensitivity is not None:
            ds = validate_detection_sensitivity(detection_sensitivity)
            self._detection_sensitivity = ds
        else:
            ds = self._detection_sensitivity

        # Validate backup detection
        if detection_backup is not None:
            detection_backup = validate_detection_sensitivity(detection_backup)

        # -------------------------------------------
        # ----------- Begin computations ------------
        # -------------------------------------------

        # Order channels
        for key, ch in self._channel_roles.items():
            if ch == "gbm":
                gbm_channel = key
            if ch == "vasc":
                vasc_channel = key

        # Load networks
        gbm_net, tme_net, seg_net = load_trained_networks()
        blocksize = 61

        # indices of rows that are NOT tracked (preserves original row labels)
        untracked_idx = self._subtable.index[~self._subtable["tracked"].astype(
            bool)].to_list()
        total_untracked = len(untracked_idx)

        print("Number of untracked stacks:", total_untracked)

        # -------------------------------------------
        # ----------- Main Pipeline -----------------
        # -------------------------------------------
        for row_i in tqdm(untracked_idx, desc="Tracking and classifying cells", unit="stack"):

            stack_path = Path(self._subtable.loc[row_i, "file_path"])
            dt = self._subtable.loc[row_i, "delta_t"]
            data = np.load(stack_path, allow_pickle=True)

            # Map channels
            channel_data = {
                "blue": data["Bstack"],
                "green": data["Tstack"],
                "red": data["Vstack"],
            }

            # Make sure conditons are met for image tracking and classifcation
            gbm_array = channel_data[gbm_channel]
            vasc_array = channel_data[vasc_channel]

            gbm, vasc = prepare_gbm_vasc_arrays(
                gbm_array, vasc_array, stack_path)

            tracked_stack = build_tracks_and_vascularity(
                gbm=gbm,
                vasc=vasc,
                gbm_net=gbm_net,
                tme_net=tme_net,
                seg_net=seg_net,
                blocksize=blocksize,
                detection_sensitivity=ds,
                i=row_i,
                dt=dt
            )

            # Repeat if no successful tracking
            if tracked_stack is None and detection_backup is not None:
                tracked_stack = build_tracks_and_vascularity(
                    gbm=gbm,
                    vasc=vasc,
                    gbm_net=gbm_net,
                    tme_net=tme_net,
                    seg_net=seg_net,
                    blocksize=blocksize,
                    detection_sensitivity=detection_backup,
                    i=row_i,
                    dt=dt
                )

            # If no cells still not found
            if tracked_stack is None:
                print(
                    f"Warning, dropped row {row_i}: exp {self._subtable.loc[row_i, 'exp']}, roi {self._subtable.loc[row_i, 'roi']}")
                self._subtable = self._subtable.drop([row_i])

                # Important if last tracked_stack happens to be None
                self._tracked = np.all(self._subtable["tracked"] == True)

                # If saving was requested
                if save_point is not None:
                    self.save_run(save_point)
                continue

            # Tag the tracked stack with information about the stack
            tracked_stack["frame_size"] = gbm.shape[0]
            tracked_stack["stack_index"] = row_i
            tracked_stack["delta_t"] = dt
            tracked_stack["exp"] = self._subtable.loc[row_i, "exp"]
            tracked_stack["roi"] = self._subtable.loc[row_i, "roi"]
            tracked_stack["is_treatment"] = self._subtable.loc[row_i,
                                                               "is_treatment"]

            # If patient informaiton exists as well:
            if "patient_id" in self._subtable.columns:
                tracked_stack["patient_id"] = str(
                    self._subtable.loc[row_i, "patient_id"])

            if "set" in self._subtable.columns:
                tracked_stack["set"] = int(self._subtable.loc[row_i, "set"])

            # Append the tracked_stack information to preexisting track_data
            if self._track_data is not None:
                self._track_data = pd.concat([self._track_data, tracked_stack])
            else:
                self._track_data = tracked_stack

            # Mark the stack as tracked
            self._subtable.loc[row_i, "tracked"] = True

            # Update internal flag
            self._tracked = np.all(self._subtable["tracked"] == True)

            if save_point is not None:
                self.save_run(save_point)

    def fit_hmm(self, fcols: Optional[list[str]] = None, hmm_param: Optional[dict] = None):
        """
        Fit a GLM-HMM (Generalized Linear Model Hidden Markov Model) to the tracked data.

        This runs the full GLM-HMM pipeline on ``self._track_data`` and stores all fitted
        artifacts on the instance (features, initial state distribution, per-state GLMs,
        transition matrix, and posterior state probabilities).

        Parameters
        ----------
        fcols
            Optional list of feature column names to use when building the design matrix
            for the GLM-HMM. If provided, the list is validated. If ``None``, the existing 
            ``self._fcols`` is used.

        hmm_param
            Optional dictionary of GLM-HMM hyperparameters / configuration. If provided,
            the dict is validated.

        @ Author: André Lasses Armatowski
        """
        self._require_tracked()

        if fcols is not None:
            self._fcols = validate_fcols(fcols)

        if hmm_param is not None:
            self._hmm_param = validate_hmm_param(hmm_param)

        self._data_feat_unfilt, self._data_feat, self._pi, self.glm_models, self._transition_matrix, self._gammas = hmm_pipeline(
            self._track_data,
            self._fcols,
            self._hmm_param,
        )

        # Update internal flag
        self._hmm = True

    # ------------------------------------------------------------------
    # Return only copies of objects
    # ------------------------------------------------------------------
    @property
    def subtable(self):
        return self._subtable.copy()

    @property
    def transition_matrix(self):
        self._require_hmm()
        return self._transition_matrix.copy()

    @property
    def track_data(self):
        self._require_tracked()
        return self._track_data.copy()

    @property
    def data_feat(self):
        self._require_hmm()
        return self._data_feat.copy()

    @property
    def pi(self):
        self._require_hmm()
        return self._pi.copy()

    @property
    def gammas(self):
        self._require_hmm()
        return self._gammas.copy()

    @property
    def video_paths(self):
        self._require_tracked()
        return self._video_paths.copy()

    # ------------------------------------------------------------------
    # Summarize Results (videos, summary statistics)
    # ------------------------------------------------------------------

    def summary_stats(
        self,
        fcol,
        patient_id=None,
        set_id=None,
        treatment=None,
        mode="mean",
    ):
        """
        Compute per-experiment summary statistics for a single feature over time and ROI.

        The output is a dictionary keyed by experiment id (``exp``). Each value is a
        time-indexed table with ROIs as columns and the aggregated statistic as values.

        Parameters
        ----------
        fcol
            Name of the feature column to summarize. Validated against the allowed feature
            set via ``validate_fcols``. Only a single feature is accepted here.

        patient_id
            Optional patient identifier filter.

        set_id
            Optional set identifier filter.

        treatment
            Optional treatment filter.

        mode
            Aggregation mode determining the statistic computed.
            "mean", "median", "sum", "min", "max", "std", "count"

        @ Author: André Lasses Armatowski
        """
        self._require_hmm()
        fcol = validate_fcols([fcol])[0]
        agg = _validate_mode(mode)

        df = self._data_feat_unfilt
        df = _filter_patient(df, patient_id)
        df = _filter_set(df, set_id, patient_id)
        df = _filter_treatment(df, treatment, patient_id, set_id)

        exp_ids = sorted(df["exp"].dropna().astype(int).unique())
        out = {}
        for exp_id in exp_ids:
            d = df.loc[df["exp"] == exp_id, ["time", "roi", fcol]].copy()
            out[int(exp_id)] = (
                d.groupby(["time", "roi"], dropna=False)[fcol]
                .agg(agg)
                .unstack("roi")
                .sort_index()
            )
        return out

    def video_compare(self, exp, roi, output: Optional[PathLike] = None):
        """
        Generate a comparison video (CNN vs Viterbi) for a given experiment + roi.

        Requires:
        - tracked data (self._track_data)
        - HMM

        output:
        - None => auto folder under ./gliotrace_videos/exp_{exp}/roi_{roi}
        - directory => created if missing
        - file path => parent created if missing

        @ Author: André Lasses Armatowski
        """
        self._require_tracked()
        self._require_hmm()

        exp, roi = _validate_exp_roi(self._subtable, exp, roi)
        out_path = self._coerce_output_path(output, exp, roi)

        result = generate_video_compare(
            self._track_data,
            self._subtable,
            self._data_feat,
            exp,
            roi,
            self._channel_roles,
            out_path,
        )

        # Store saved path
        self._video_paths.append(
            result if result is not None else out_path
        )
        return result

    def video_tracking(self, exp, roi, output: Optional[PathLike] = None):
        """
        Generate a video for a given experiment + roi.

        output:
          - None => auto folder under ./gliotrace_videos/exp_{exp}/roi_{roi}
          - directory => created if missing
          - file path => parent created if missing

        @ Author: André Lasses Armatowski
        """
        self._require_tracked()

        exp, roi = _validate_exp_roi(self._subtable, exp, roi)
        out_path = self._coerce_output_path(output, exp, roi)

        result = generate_video(
            self._subtable,
            self._track_data,
            exp,
            roi,
            self._channel_roles,
            out_path,
        )

        # Store saved path.
        self._video_paths.append(
            result if result is not None else out_path)

    # ------------------------------------------------------------------
    # Public Helper Functions for Sorting
    # ------------------------------------------------------------------

    def patients(self, treated: bool | None = None) -> list[str]:
        """
        List unique patient identifiers present in the current subtable.

        Parameters
        ----------
        treated
            Treatment filter applied via ``_apply_treated``:
            - ``None``: include both control and treated stacks
            - ``False``: include control stacks only
            - ``True``: include treated stacks only

        Returns
        -------
        list[str]
            Sorted unique ``patient_id`` values (as strings), with missing values removed.

        @ Author: André Lasses Armatowski
        """
        st = self._subtable
        _require_cols(st, "patient_id", name="subtable")
        st = _apply_treated(st, treated)
        return sorted(st["patient_id"].dropna().astype(str).unique().tolist())

    def sets(self, patient: str, treated: bool | None = None) -> list[int]:
        """
        List unique set indices for a given patient.

        Parameters
        ----------
        patient
            Patient identifier used to filter the subtable 
        treated
            Treatment filter 

        Returns
        -------
        list[int]
            Sorted unique ``set`` values (as ints).

        @ Author: André Lasses Armatowski
        """
        st = self._subtable
        _require_cols(st, "patient_id", "set", name="subtable")

        st = _apply_patient(st, patient, name="subtable")
        st = _apply_treated(st, treated)

        return sorted(st["set"].dropna().astype(int).unique().tolist())

    def exps(self, patient: str | None = None, set: int | None = None, treated: bool | None = None) -> list[int]:
        st = self._subtable
        _require_cols(st, "exp", name="stacktable")

        st = _apply_patient(st, patient, name="stacktable")
        st = _apply_set(st, set, patient=patient, name="stacktable")
        st = _apply_treated(st, treated)

        return sorted(st["exp"].dropna().astype(int).unique().tolist())

    def print_configuration(self) -> None:
        """
        Print a human-readable summary of the current GlioTrace configuration and dataset scope.

        Returns
        -------
        None

        @ Author: André Lasses Armatowski
        """
        print("\n=== Gliotrace configuration ===")
        st = self._subtable
        print(f"Number of stacks        : {len(st)}")

        def _show(xs, n=10):
            xs = list(xs)
            if len(xs) <= n:
                return xs
            return xs[:n] + [f"... (n={len(xs)})"]

        print("Scope                   :")

        has_patient = "patient_id" in st.columns
        has_set = "set" in st.columns
        has_exp = "exp" in st.columns

        if has_patient and has_set and has_exp:
            # patient -> set -> exps
            grp = (
                st.dropna(subset=["patient_id", "set", "exp"])
                .groupby(["patient_id", "set"])["exp"]
                .apply(lambda s: sorted(s.astype(int).unique().tolist()))
            )
            for pid in _show(sorted(grp.index.get_level_values(0).unique()), n=20):
                if isinstance(pid, str) and pid.startswith("..."):
                    print(f"  {pid}")
                    break
                print(f"  Patient {pid}:")
                sub = grp.loc[pid]  # indexed by set
                for s in _show(sorted(sub.index.unique().tolist()), n=20):
                    if isinstance(s, str) and s.startswith("..."):
                        print(f"    {s}")
                        break
                    print(f"    Set {s}: exps {_show(sub.loc[s], n=12)}")

        elif has_patient and has_exp:
            # patient -> exps
            grp = (
                st.dropna(subset=["patient_id", "exp"])
                .groupby("patient_id")["exp"]
                .apply(lambda s: sorted(s.astype(int).unique().tolist()))
            )
            for pid in _show(sorted(grp.index.unique().tolist()), n=20):
                if isinstance(pid, str) and pid.startswith("..."):
                    print(f"  {pid}")
                    break
                print(f"  Patient {pid}: exps {_show(grp.loc[pid], n=12)}")

        elif has_set and has_exp:
            # set -> exps
            grp = (
                st.dropna(subset=["set", "exp"])
                .groupby("set")["exp"]
                .apply(lambda s: sorted(s.astype(int).unique().tolist()))
            )
            for s in _show(sorted(grp.index.unique().tolist()), n=30):
                if isinstance(s, str) and s.startswith("..."):
                    print(f"  {s}")
                    break
                print(f"  Set {s}: exps {_show(grp.loc[s], n=12)}")

        elif has_exp:
            exps = sorted(st["exp"].dropna().astype(int).unique().tolist())
            print(f"  Exps                  : {_show(exps, n=20)}")
        else:
            print("  (no 'exp' column found)")

        # ----- core config -----
        print(f"Detection sensitivity   : {self._detection_sensitivity}")
        print(f"Channel roles           : {self._channel_roles}")
        print(f"Feature columns         : {self._fcols}")
        print(f"HMM parameters          : {self._hmm_param}")

        # ----- controls / treatment -----
        if "perturbation" in st.columns:
            control = st.loc[st["is_treatment"]
                             == False, "perturbation"].unique()

            if len(control) != 1:
                print(f"Control                 : All")
            else:
                print(f"Control                 : {control[0]}")

        if (st["is_treatment"] == True).any():
            t_name = st.loc[st["is_treatment"]
                            == True, "perturbation"].unique()

            t_dose = []
            if "treatment_dose" in st.columns:
                t_dose = st.loc[st["is_treatment"] == True,
                                "treatment_dose"].dropna().unique()

            if len(t_dose) == 0:
                print(f"Treatment               : {t_name[0]}")
            else:
                print(
                    f"Treatment               : {t_name[0]} (dose={t_dose[0]})")

        n_treat = int((st["is_treatment"] == True).sum())
        n_ctrl = int((st["is_treatment"] == False).sum())
        print(f"Stacks (control/treat)  : {n_ctrl} / {n_treat}")

        print("================================\n")

    def print_exp_roi(self):
        print(self._subtable[["exp", "roi"]])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_tracked(self):
        if not self._tracked:
            raise RuntimeError(
                "run_tracking must be called before this method")

    def _require_hmm(self):
        if not self._hmm:
            raise RuntimeError("fit_hmm must be called before this method")

    def _assert_same_rows_except_tracked(self, current: pd.DataFrame, loaded: pd.DataFrame):
        """
        Internal helper to compare loaded and initalized object
        @ Author: André Lasses Armatowski
        """
        # drop tracked if present
        c = current.drop(columns=["tracked"], errors="ignore").copy()
        l = loaded.drop(columns=["tracked"], errors="ignore").copy()

        # normalize index + column order for a fair comparison
        c = c.reset_index(drop=True)
        l = l.reset_index(drop=True)

        c = c.reindex(sorted(c.columns), axis=1)
        l = l.reindex(sorted(l.columns), axis=1)

        if c.shape != l.shape:
            raise ValueError(
                f"Subtable mismatch; The intialized object differs from the load object: shape current={c.shape}, loaded={l.shape}. If you like to initalize a new object with the same structure as the load-file, use 'GlioTrace.load_run(path)'")

        if not c.equals(l):
            # show a small hint of where it differs
            diff = (c != l) & ~(c.isna() & l.isna())
            first = diff.stack()
            first = first[first].index.tolist()
            where = first[0] if first else None
            raise ValueError(
                f"Subtable mismatch (ignoring 'tracked'). First difference at {where}")

    def _coerce_output_path(self, output: Optional[PathLike], exp: int, roi: int):
        """
        Decide where to write output.

        - If output is None, write to ./gliotrace_videos/exp_{exp}/roi_{roi}
        - If output is a directory, create it
        - If output is a file path, create its parent directory

        @ Author: André Lasses Armatowski
        """
        if output is None:
            out = Path.cwd() / "gliotrace_videos" / f"exp_{exp}" / f"roi_{roi}"
        else:
            out = Path(output)

        # If they passed something with a suffix, treat as file path; else dir.
        if out.suffix:
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out.mkdir(parents=True, exist_ok=True)

        return out

    def _compare_hmm_and_cnn_class(self, threshold=0.1):
        """
        Internal helper to compare cnn and gamma classifaction
        @ Author: André Lasses Armatowski
        """

        self._require_tracked()
        self._require_hmm()

        softmax_cols = SOFTMAX_COLUMNS

        data_feat = self._data_feat.copy()
        gammas_long = self._gammas.copy()

        # ---- convert state_label to softmax label names ----
        data_feat["state_label"] = data_feat["state_label"].apply(
            lambda x: softmax_cols[int(x) - 1])

        # ---- build mask: max CNN softmax prob > threshold ----
        maxprob = data_feat[softmax_cols].max(axis=1)
        data_sel = data_feat.loc[maxprob > threshold].copy()
        if data_sel.empty:
            print("No rows passed the CNN threshold.")
            return pd.DataFrame()

        # ---- merge selected data with gammas ----
        merged = data_sel.merge(
            gammas_long[["exp", "roi", "cellID", "time"] + softmax_cols],
            on=["exp", "roi", "cellID", "time"],
            how="inner",
            validate="one_to_one",
            suffixes=("_cnn", "_gamma"),
        )

        # ---- gamma class from GAMMA columns ----
        gamma_cols = [f"{c}_gamma" for c in softmax_cols]
        merged["gamma_class"] = (
            merged[gamma_cols]
            .idxmax(axis=1)
            .str.replace(r"_gamma$", "", regex=True)
        )

        # ---- contingency table: state_label vs gamma_class ----
        ct = (
            merged
            .groupby(["state_label", "gamma_class"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=softmax_cols, columns=softmax_cols, fill_value=0)
        )

        return ct

    # ------------------------------------------------------------------
    # Save and Load GlioTrace
    # ------------------------------------------------------------------

    def save_run(self, run_dir: str | Path) -> Path:
        """
        Parameters
        ----------
        run_dir
            Target directory for the run. Created if it does not exist.

        Returns
        -------
        pathlib.Path
            Path to the run directory.

        Side Effects
        ------------
        Creates directories and overwrites any existing files with the same names in
        ``run_dir``.

        @ Author: André Lasses Armatowski
        """

        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(exist_ok=True)

        manifest = {
            "format": "gliotrace-run-v1",
            "tracked": bool(self._tracked),
            "hmm": bool(self._hmm),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        cfg = {
            "detection_sensitivity": self._detection_sensitivity,
            "channel_roles": self._channel_roles,
            "fcols": self._fcols,
            "hmm_param": self._hmm_param
        }
        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        # --- tables (pickle handles object columns like segmented_stack) ---
        self._subtable.to_pickle(run_dir / "tables" / "subtable.pkl")

        if self._track_data is not None:
            self._track_data.to_pickle(run_dir / "tables" / "track_data.pkl")

        if self._hmm:

            self._data_feat_unfilt.to_pickle(
                run_dir / "tables" / "data_unfilt.pkl")
            self._data_feat.to_pickle(run_dir / "tables" / "data_feat.pkl")
            self._gammas.to_pickle(run_dir / "tables" / "gammas.pkl")
            self._pi.to_pickle(run_dir / "tables" / "pi.pkl")
            self._transition_matrix.to_pickle(
                run_dir / "tables" / "transition_matrix.pkl")

            joblib.dump(self.glm_models, run_dir / "glm_models.joblib")

        return run_dir

    @classmethod
    def load_run(cls, run_dir: str | Path) -> "GlioTrace":
        """
        Load a GlioTrace instance from a previously saved run directory.

        Parameters
        ----------
        run_dir
            Path to a run directory produced by :meth:`save_run`.

        Returns
        -------
        GlioTrace
            A GlioTrace instance populated with the stored configuration and results.

        Implementation Detail
        ---------------------
        ``_video_paths`` is initialized to an empty list; video exports are not
        restored by this loader.

        @ Author: André Lasses Armatowski
        """
        run_dir = Path(run_dir)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("format") != "gliotrace-run-v1":
            raise ValueError(f"Unknown run format: {manifest.get('format')}")

        cfg = json.loads((run_dir / "config.json").read_text())

        self = cls.__new__(cls)

        # restore config fields
        self._detection_sensitivity = cfg["detection_sensitivity"]
        self._channel_roles = cfg["channel_roles"]
        self._fcols = cfg["fcols"]
        self._hmm_param = cfg["hmm_param"]

        # internal flags
        self._tracked = bool(manifest.get("tracked", False))
        self._hmm = bool(manifest.get("hmm", False))
        self._video_paths = []

        # load tables (pickle)
        self._subtable = pd.read_pickle(run_dir / "tables" / "subtable.pkl")

        track_path = run_dir / "tables" / "track_data.pkl"
        if track_path.exists():
            self._track_data = pd.read_pickle(track_path)
        else:
            self._track_data = None

        if self._hmm:
            pi_path = run_dir / "tables" / "pi.pkl"
            self._pi = pd.read_pickle(pi_path) if pi_path.exists() else None

            transition_path = run_dir / "tables" / "transition_matrix.pkl"
            self._transition_matrix = pd.read_pickle(
                transition_path) if transition_path.exists() else None

            data_unfilt_path = run_dir / "tables" / "data_unfilt.pkl"
            self._data_feat_unfilt = pd.read_pickle(
                data_unfilt_path) if data_unfilt_path.exists() else None

            data_feat_path = run_dir / "tables" / "data_feat.pkl"
            self._data_feat = pd.read_pickle(
                data_feat_path) if data_feat_path.exists() else None

            gammas_path = run_dir / "tables" / "gammas.pkl"
            self._gammas = pd.read_pickle(
                gammas_path) if gammas_path.exists() else None

            glm_path = run_dir / "glm_models.joblib"
            self.glm_models = joblib.load(
                glm_path) if glm_path.exists() else None

        return self
