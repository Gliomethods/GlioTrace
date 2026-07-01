from gliotrace.build_tracks_and_vascularity.build_tracks_and_vascularity import build_tracks_and_vascularity
from gliotrace.build_tracks_and_vascularity.weights_and_models.load_networks import load_trained_networks
from gliotrace.visualize.generate_video_compare import generate_video_compare
from gliotrace.feature_and_hmm_pipline.hmm_pipeline import hmm_pipeline
from gliotrace.feature_and_hmm_pipline.feature_construction import feature_construction
from gliotrace.feature_and_hmm_pipline.hmm_pipeline import filter_features_3
from gliotrace.initalize_class.load_data import build_stack_table_flex
from gliotrace.visualize.generate_video import generate_video
from gliotrace.initalize_class.validation import (
    validate_init, validate_fcols, validate_hmm_param,
    validate_detection_sensitivity,
    _filter_patient, _filter_set, _filter_treatment,
    _validate_mode, _apply_patient, _apply_set,
    _require_cols, _apply_treated, _validate_exp_roi,
)
from gliotrace.initalize_class.defaults import SOFTMAX_COLUMNS, HARD_CODED_FEATURES, NON_SCALE_COLUMNS
from gliotrace.visualize.preprocess_stack import prepare_gbm_vasc_arrays

from gliotrace.feature_and_hmm_pipline.transition_utils import trans_pct, trans_pct_diff

from scipy.special import softmax as scipy_softmax
from sklearn.preprocessing import StandardScaler
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
        detection_backup: float = None,
        channel_roles: dict[str, str] | None = None,
        fcols: list[str] | None = None,
        hmm_param: dict[str, int | float] = None,
        control: str | list[str] = "all",
        patient_id: Optional[list[str]] = None,
        sets_by_patient: Optional[dict[str, list[str]]] = None,
        treatment: list[str] | None = None,
    ):
        """
        Initialize a GlioTrace analysis run.

        Parameters
        ----------
        stackfile
            Stack file(s) to analyze.
        metadata
            Metadata used to annotate each stack.
        detection_sensitivity
            Detection sensitivity for the tracking pipeline (default 0.2).
        detection_backup
            Optional fallback detection sensitivity if primary tracking finds no cells.
        channel_roles
            Mapping from channel keys to semantic roles, e.g. {"green": "gbm", "red": "vasc"}.
        fcols
            Feature column names for the GLM-HMM design matrix.
        hmm_param
            GLM-HMM hyperparameters/configuration.
        control
            Control perturbation label(s). Accepts a string or list of strings (default "all").
        patient_id
            Optional list of patient identifiers to filter on.
        sets_by_patient
            Optional mapping {patient_id: [set_id, ...]} for per-patient set filtering.
        treatment
            Optional 2-element list ``[perturbation, dose]`` specifying the treatment
            perturbation and dose. Pass ``[perturbation, None]`` to include all doses.

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

        # Build stacktable containing all information about the run data
        self._subtable = build_stack_table_flex(
            v.stackfiles,
            v.metadata,
            v.treatment,
            v.control,
            v.patient_id,
            v.sets_by_patient,
        )

        # Start off with all stacks being untracked
        self._subtable["tracked"] = False

        # Store metadata path for use by downstream methods
        self._metadata = v.metadata

        # Store validated config
        self._detection_sensitivity = v.detection_sensitivity
        self._detection_backup = detection_backup
        self._channel_roles = v.channel_roles
        self._fcols = v.fcols
        self._hmm_param = v.hmm_param

        # Internal state flags
        self._tracked = False
        self._feat = False
        self._hmm = False

        # Keep track of output paths
        self._video_paths = []

        # Empty data containers
        self._track_data = None
        self._data_feat = None
        self._data_feat_unfilt = None

        self.print_configuration()

    # ------------------------------------------------------------------
    # Main Computational Pipeline
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
            Optional path for checkpointing after each stack.
        load_point
            Optional path to reload a previously saved run.
        detection_sensitivity
            Overrides instance detection sensitivity for this run.
        detection_backup
            Overrides instance detection_backup for this run.

        @ Author: André Lasses Armatowski, Madeleine Skeppås
        """
        lp = load_point
        if lp is not None:
            loaded = self.__class__.load_run(lp)
            self._assert_same_rows_except_tracked(self._subtable, loaded._subtable)
            self.__dict__.update(loaded.__dict__)

        self._tracked = np.all(self._subtable["tracked"] == True)
        if self._tracked:
            print("All stacks have already been tracked")
            return

        if detection_sensitivity is not None:
            ds = validate_detection_sensitivity(detection_sensitivity)
            self._detection_sensitivity = ds
        else:
            ds = self._detection_sensitivity

        if detection_backup is not None:
            db = validate_detection_sensitivity(detection_backup)
        else:
            db = self._detection_backup

        for key, ch in self._channel_roles.items():
            if ch == "gbm":
                gbm_channel = key
            if ch == "vasc":
                vasc_channel = key

        gbm_net, tme_net, seg_net = load_trained_networks()
        blocksize = 61

        untracked_idx = self._subtable.index[
            ~self._subtable["tracked"].astype(bool)
        ].to_list()
        print("Number of untracked stacks:", len(untracked_idx))

        for row_i in tqdm(untracked_idx, desc="Tracking and classifying cells", unit="stack"):

            stack_path = Path(self._subtable.loc[row_i, "file_path"])
            dt = self._subtable.loc[row_i, "delta_t"]
            data = np.load(stack_path, allow_pickle=True)

            channel_data = {
                "blue": data["Bstack"],
                "green": data["Tstack"],
                "red": data["Vstack"],
            }

            gbm_array = channel_data[gbm_channel]
            vasc_array = channel_data[vasc_channel]
            gbm, vasc = prepare_gbm_vasc_arrays(gbm_array, vasc_array, stack_path)

            tracked_stack = build_tracks_and_vascularity(
                gbm=gbm, vasc=vasc,
                gbm_net=gbm_net, tme_net=tme_net, seg_net=seg_net,
                blocksize=blocksize,
                detection_sensitivity=ds,
                i=row_i, dt=dt,
            )

            if tracked_stack is None and db is not None:
                tracked_stack = build_tracks_and_vascularity(
                    gbm=gbm, vasc=vasc,
                    gbm_net=gbm_net, tme_net=tme_net, seg_net=seg_net,
                    blocksize=blocksize,
                    detection_sensitivity=db,
                    i=row_i, dt=dt,
                )

            if tracked_stack is None:
                print(
                    f"Warning, dropped row {row_i}: "
                    f"exp {self._subtable.loc[row_i, 'exp']}, "
                    f"roi {self._subtable.loc[row_i, 'roi']}"
                )
                self._subtable = self._subtable.drop([row_i])
                self._tracked = np.all(self._subtable["tracked"] == True)
                if save_point is not None:
                    self.save_run(save_point)
                continue

            tracked_stack["frame_size"] = gbm.shape[0]
            tracked_stack["stack_index"] = row_i
            tracked_stack["delta_t"] = dt
            tracked_stack["exp"] = self._subtable.loc[row_i, "exp"]
            tracked_stack["roi"] = self._subtable.loc[row_i, "roi"]
            tracked_stack["is_treatment"] = self._subtable.loc[row_i, "is_treatment"]

            if "patient_id" in self._subtable.columns:
                tracked_stack["patient_id"] = str(self._subtable.loc[row_i, "patient_id"])
            if "set" in self._subtable.columns:
                tracked_stack["set"] = int(self._subtable.loc[row_i, "set"])

            if self._track_data is not None:
                self._track_data = pd.concat([self._track_data, tracked_stack])
            else:
                self._track_data = tracked_stack

            self._subtable.loc[row_i, "tracked"] = True
            self._tracked = np.all(self._subtable["tracked"] == True)

            if save_point is not None:
                self.save_run(save_point)

    def create_and_filter_features(self, fcols: Optional[list[str]] = None):
        """
        Build features for all tracked experiments and filter out unusable tracks.

        Parameters
        ----------
        fcols
            Optional override of feature columns. Updates self._fcols if provided.

        @ Author: André Lasses Armatowski
        """
        self._require_tracked()

        if fcols is not None:
            self._fcols = validate_fcols(fcols)

        data = self._track_data.copy()
        frames = []
        for exp in data["exp"].unique():
            exp_data = data.loc[data["exp"] == exp].copy()
            frames.append(feature_construction(exp_data))

        data_feat = pd.concat(frames, ignore_index=True)
        self._data_feat_unfilt = data_feat.copy()

        self._data_feat = filter_features_3(
            data_feat,
            fcols=self._fcols,
            hard_coded_features=HARD_CODED_FEATURES,
            min_timepoints=2,
            verbose=True,
        )

        if self._data_feat is None:
            print("Warning: No usable tracks after filtering")

        self._feat = True

    def fit_hmm(
        self,
        fcols: Optional[list[str]] = None,
        hmm_param: Optional[dict] = None,
        filter_by: Optional[dict] = None,
        scalers: Optional[dict] = None,
    ):
        """
        Fit a GLM-HMM to the tracked data.

        Parameters
        ----------
        fcols
            Feature columns for the design matrix.
        hmm_param
            GLM-HMM hyperparameters.
        filter_by
            Optional dict of column -> values to restrict which rows are modelled,
            e.g. {"exp": [1, 2, 3]}.
        scalers
            Optional pre-fitted StandardScaler objects keyed by feature name.
            If None, hmm_pipeline fits its own scaling.
        create_features
            If True (default), runs feature construction internally.
            Set to False if create_and_filter_features was already called.

        @ Author: André Lasses Armatowski
        """
        self._require_tracked()

        if fcols is not None:
            self._fcols = validate_fcols(fcols)
        if hmm_param is not None:
            self._hmm_param = validate_hmm_param(hmm_param)

        (
            self._data_feat_unfilt,
            self._data_feat,
            self._pi,
            self.glm_models,
            self._transition_matrix,
            self._gammas,
        ) = hmm_pipeline(
            self._track_data,
            self._fcols,
            self._hmm_param,
            filter_by=filter_by,
            scalers=scalers,
        )

        self._feat = True
        self._hmm = True

    def get_matched_control_exp(
        self,
        treatment: str,
        cell_line: Optional[str] = None,
        dose=None,
        control: str | list[str] = ["control"],
    ) -> np.ndarray:
        """
        Return treatment experiment IDs plus matched control experiment IDs.

        For non-in-vivo data, controls are matched by set (and cell line if
        the ``patient_id`` column is present in the metadata). For in-vivo data,
        all control experiments are included regardless of set and cell line.

        Parameters
        ----------
        treatment
            Treatment perturbation name. If this is itself a control label,
            all matching control experiments are returned.
        cell_line
            Optional patient_id/cell-line identifier. If ``None`` and the metadata
            has no ``patient_id`` column, all data is treated as one cell line.
            If ``None`` and the metadata has a ``patient_id`` column, no cell-line
            filter is applied (all cell lines are included).
        dose
            Optional list of allowed doses. If None, all doses are included.
        control
            Names of perturbations considered controls.

        Returns
        -------
        np.ndarray
            Array of experiment IDs covering both treatment and matched controls.

        @ Author: Linnea Hallin
        """
        metadata_df = self._get_metadata_df()
        control = [control] if isinstance(control, str) else list(control)
        has_patient_id = "patient_id" in metadata_df.columns

        # Cell line filter — only applied when the patient_id column exists
        def _cl_mask(df):
            if has_patient_id and cell_line is not None:
                return df["patient_id"] == cell_line
            return pd.Series(True, index=df.index)

        if treatment in control:
            return metadata_df[
                _cl_mask(metadata_df)
                & metadata_df["perturbation"].isin(control)
            ]["experiment_id"].unique()

        treat_idx = _cl_mask(metadata_df) & (
            metadata_df["perturbation"] == treatment
        )
        if dose is not None:
            treat_idx = treat_idx & (metadata_df["dose"].isin(dose))

        matched_exp_treat = metadata_df[treat_idx]["experiment_id"].unique()

        if "set" not in metadata_df.columns:
            matched_exp_control = metadata_df[
                _cl_mask(metadata_df) & metadata_df["perturbation"].isin(control)
            ]|["experiment_id"].unique()
        else:
            sets = metadata_df[
                metadata_df["experiment_id"].isin(matched_exp_treat)
            ]["set"].unique()
            matched_exp_control = metadata_df[
                _cl_mask(metadata_df)
                & metadata_df["perturbation"].isin(control)
                & metadata_df["set"].isin(sets)
            ]["experiment_id"].unique()

        if len(matched_exp_control)==0:
            control_idx = _cl_mask(metadata_df) & metadata_df["perturbation"].isin(control)
            matched_exp_control = metadata_df[control_idx]["experiment_id"].unique()

        return np.concatenate([matched_exp_treat, matched_exp_control])
    
    def _get_metadata_df(self):
        if self._metadata is None:
            raise RuntimeError(
                "No metadata available. Re-create the GlioTrace object with metadata."
            )
        return self._metadata.copy()

    def run_hmm_by(
        self,
        treatment: str,
        cell_line: Optional[str] = None,
        dose=None,
        control: str | list[str] = ["control"],
        save_path: str = "runs/modelled/default_run/",
        filter_by: dict = {},
        scalers: Optional[dict] = None,
        fcols: Optional[list[str]] = None,
        **kwargs,
    ):
        """
        Fit a GLM-HMM on matched treatment/control experiments and save the fitted run.

        Fetches matched experiment IDs via ``get_matched_control_exp`` and combines
        them with any additional ``filter_by`` constraints before calling ``fit_hmm``.
        If ``filter_by`` already contains an ``"exp"`` key, only the intersection of
        the requested and matched experiments is used.

        Feature construction is expected to have been run beforehand via
        ``create_and_filter_features``; this method always calls ``fit_hmm`` with
        ``create_features=False``.

        Parameters
        ----------
        treatment
            Treatment perturbation name.
        cell_line
            Optional patient_id/cell-line identifier. If None, no cell-line filter is applied.
        dose
            Optional list of allowed doses.
        control
            Names of perturbations considered controls.
        save_path
            Directory to save the fitted run.
        filter_by
            Additional column-level filters passed to ``fit_hmm``,
            e.g. ``{"set": [1, 2]}``.
        scalers
            Pre-fitted ``StandardScaler`` objects keyed by feature name,
            passed directly to ``fit_hmm``.
        fcols
            Feature columns. If None, ``self._fcols`` is used.

        @ Author: Linnea Hallin
        """
        exps = self.get_matched_control_exp(
            treatment,
            cell_line=cell_line,
            dose=dose, control=control,
        )

        filter_by_fcn = filter_by.copy()
        if "exp" in filter_by_fcn:
            filter_by_fcn["exp"] = list(set(exps) & set(filter_by_fcn["exp"]))
        else:
            filter_by_fcn["exp"] = exps

        self.fit_hmm(
            fcols=fcols,
            filter_by=filter_by_fcn,
            scalers=scalers,
            **kwargs,
        )
        self.save_run(save_path)

    def get_scalers(
        self,
        treatment: str,
        cell_line: Optional[str] = None,
        fcols: list[str] = None,
        treat_interactions: Optional[list[str]] = None,
        dose=None,
        control: str | list[str] = ["control"],
        non_scale_columns=NON_SCALE_COLUMNS,
    ) -> dict:
        """
        Fit StandardScaler objects for model features on matched experiment data.

        Scalers are fitted on the unfiltered feature data (``_data_feat_unfilt``)
        restricted to the matched treatment and control experiments. Interaction
        features are scaled using the same distribution as their source feature,
        but stored under the interaction column name so they can be applied
        independently at inference time.

        Requires ``create_and_filter_features`` to have been called first.

        Parameters
        ----------
        treatment
            Treatment perturbation name.
        cell_line
            Optional patient_id/cell-line identifier. If None, no cell-line filter is applied.
        fcols
            Feature columns to fit scalers for.
        treat_interactions
            Features for which a treatment interaction column (``{feat}_treat``)
            should also receive a scaler. The interaction scaler is fitted on the
            same raw feature values as the base feature.
        dose
            Optional list of allowed doses.
        control
            Names of perturbations considered controls.
        non_scale_columns
            Columns that should never be scaled (e.g. binary flags).

        Returns
        -------
        dict
            Mapping of feature name -> fitted ``StandardScaler``.

        @ Author: Linnea Hallin
        """
        self._require_feat()

        exps = self.get_matched_control_exp(
            treatment,
            cell_line=cell_line,
            dose=dose, control=control,
        )

        data_feat_unscaled = self._data_feat.loc[
            self._data_feat["exp"].isin(exps)
        ]

        treat_interactions = treat_interactions or []
        scalers = {}

        for feat in fcols:
            if feat not in non_scale_columns:
                scalers[feat] = StandardScaler().fit(data_feat_unscaled[[feat]])

                if feat in treat_interactions:
                    name = f"{feat}_treat"
                    scalers[name] = StandardScaler().fit(data_feat_unscaled[[feat]])
                    scalers[name].feature_names_in_ = [name]

        return scalers

    def _run_loo_fits(
        self,
        treatment: str,
        fcols: list[str],
        cell_line: Optional[str] = None,
        dose=None,
        control: str | list[str] = ["control"],
        save_path: str = "runs/modelled/",
        run_by: Optional[str] = None,
        treat_interactions: Optional[list[str]] = None,
        run_full: bool = True,
        filter_by: dict = {},
        **kwargs,
    ) -> dict:
        """
        Internal helper: fit the full model and all LOO replicates and save them to disk.

        Called by ``loo_hmm``. Handles feature construction, scaler fitting, and
        iterating over LOO groups. One model is fitted per LOO group by excluding
        one value of ``run_by`` at a time (typically one experimental set).
        If ``run_full=True``, an additional model is fitted on all matched experiments.

        All fitted models are saved to subdirectories under ``save_path``:
        ``{save_path}full`` for the full model and ``{save_path}excl{s}`` for each
        LOO replicate excluding group ``s``.

        Parameters
        ----------
        treatment
            Treatment perturbation name.
        fcols
            Base feature columns. Interaction columns (``{feat}_treat``) are
            appended automatically if ``treat_interactions`` is provided.
        cell_line
            Optional patient_id/cell-line identifier. If None, no cell-line filter is applied.
        dose
            Optional list of allowed doses.
        control
            Names of perturbations considered controls.
        save_path
            Root directory under which per-model subdirectories are created.
        run_by
            Column in ``track_data`` defining the LOO grouping. Defaults to ``"set"`` if that column exists in ``track_data``, otherwise ``"experiment_id"``. Can be overridden explicitly.
        treat_interactions
            Features for which a treatment interaction term (``{feat}_treat``)
            should be constructed and included in the model. Requires
            ``"is_treatment"`` to be present in ``fcols``.
        run_full
            If True (default), also fit and save a model on all matched experiments.
        filter_by
            Additional column-level filters applied to every model fit,
            e.g. ``{"roi": [1, 2]}``.

        Returns
        -------
        dict
            Mapping of run label (``"full"``, ``"excl{s}"``) -> save path string.

        @ Author: Linnea Hallin
        """

        if fcols is None or len(fcols) == 0:
            raise ValueError("fcols cannot be None or empty.")

        treat_interactions = treat_interactions or []

        is_control_only = treatment in control

        if is_control_only:
            treat_interactions = []
            fcols = [f for f in fcols if f != "is_treatment"]
            fcols_full = fcols
        else:
            # Build interaction feature names
            if treat_interactions:
                if "is_treatment" not in fcols:
                    raise ValueError(
                        "Cannot create interaction terms if 'is_treatment' is not in fcols."
                    )
                for feat in treat_interactions:
                    if feat not in fcols:
                        raise ValueError(
                            f"Cannot create interaction term for '{feat}' "
                            f"because it is not in fcols."
                        )
                fcols_full = fcols + [f"{feat}_treat" for feat in treat_interactions]
            else:
                fcols_full = fcols

        print("-- Creating features --")
        self.create_and_filter_features(fcols=fcols_full)

        exps = self.get_matched_control_exp(
            treatment,
            cell_line=cell_line,
            dose=dose, control=control,
        )

        data_feat_unscaled = self._data_feat.loc[
            self._data_feat["exp"].isin(exps)
        ].copy()

        print("-- Fitting scalers --")
        scalers = self.get_scalers(
            treatment,
            cell_line=cell_line,
            fcols=fcols,
            treat_interactions=treat_interactions,
            dose=dose, control=control,
        )

        save_paths = {}
        filter_by_fcn = filter_by.copy()

        if run_full:
            print("-- Running full model --")
            save_path_full = f"{save_path}full"
            save_paths["full"] = save_path_full
            self.run_hmm_by(
                treatment,
                cell_line=cell_line,
                dose=dose, control=control,
                save_path=save_path_full,
                scalers=scalers,
                fcols=fcols_full,
                filter_by=filter_by,
                **kwargs,
            )

        loo_groups = self._track_data.loc[
            self._track_data["exp"].isin(exps), run_by
        ].unique()

        if run_by in filter_by:
            loo_groups = list(set(loo_groups) & set(filter_by[run_by]))

        for i, s in enumerate(loo_groups):
            groups_used = np.delete(loo_groups.copy(), i)
            print(f"-- Running model excluding {run_by}={s}, using: {groups_used} --")

            filter_by_fcn[run_by] = groups_used
            save_path_excl = f"{save_path}excl{s}"
            save_paths[f"excl{s}"] = save_path_excl

            self.run_hmm_by(
                treatment,
                cell_line=cell_line,
                dose=dose, control=control,
                save_path=save_path_excl,
                scalers=scalers,
                fcols=fcols_full,
                filter_by=filter_by_fcn,
                **kwargs,
            )

        return save_paths, data_feat_unscaled, scalers

    def _loo_hmm_single(
        self,
        treatment: str,
        fcols: list[str],
        cell_line: Optional[str] = None,
        dose=None,
        control: str | list[str] = ["control"],
        treat_interactions: Optional[list[str]] = None,
        run_by: Optional[str] = None,
        save_path: str = "runs/modelled/",
        **kwargs,
    ) -> dict:
        """
        Core single-run LOO HMM workflow for one (cell_line, treatment, dose) combination.

        Called internally by ``loo_hmm``. Fits the full model and all LOO replicates
        via ``_run_loo_fits``, then reloads the saved runs to compute jackknife
        estimates of the global transition matrix and per-feature transition differences.

        Parameters
        ----------
        treatment
            Treatment perturbation name.
        fcols
            Base feature columns.
        cell_line
            Optional patient_id/cell-line identifier. If None, no cell-line filter is applied.
        dose
            Optional list of allowed doses.
        control
            Names of perturbations considered controls.
        treat_interactions
            Features for which treatment interaction terms should be included.
        run_by
            Column in ``track_data`` defining the LOO grouping. Defaults to ``"set"`` if that column exists in ``track_data``, otherwise ``"experiment_id"``. Can be overridden explicitly.
        save_path
            Root directory for saving fitted runs.

        Returns
        -------
        dict
            Dictionary with keys ``global_a``, ``global_a_mean``, ``global_a_var``,
            ``trans_diff``, ``trans_diff_mean``, ``trans_diff_var``.

        @ Author: Linnea Hallin
        """
        
        treat_interactions = treat_interactions or []

        is_control_only = treatment in control

        if is_control_only:
            treat_interactions = []
            fcols = [f for f in fcols if f != "is_treatment"]
            fcols_full = fcols
        else:
            # Build interaction feature names
            if treat_interactions:
                if "is_treatment" not in fcols:
                    raise ValueError(
                        "Cannot create interaction terms if 'is_treatment' is not in fcols."
                    )
                for feat in treat_interactions:
                    if feat not in fcols:
                        raise ValueError(
                            f"Cannot create interaction term for '{feat}' "
                            f"because it is not in fcols."
                        )
                fcols_full = fcols + [f"{feat}_treat" for feat in treat_interactions]
            else:
                fcols_full = fcols

        save_paths, data_feat_unscaled, scalers = self._run_loo_fits(
            treatment,
            cell_line=cell_line,
            fcols=fcols,
            dose=dose, 
            control=control,
            save_path=save_path,
            run_by=run_by,
            treat_interactions=treat_interactions,
            run_full=True,
            **kwargs,
        )

        fcols_full = fcols + [f"{feat}_treat" for feat in treat_interactions]

        full = self.__class__.load_run(save_paths["full"])
        jackknives = [
            self.__class__.load_run(save_paths[key])
            for key in save_paths
            if key.startswith("excl")
        ]
        n = len(jackknives)

        global_a = full.transition_matrix
        global_a_mean = np.mean([jk.transition_matrix for jk in jackknives], axis=0)
        global_a_var = (n - 1) * np.var(
            [jk.transition_matrix for jk in jackknives], axis=0, ddof=0
        )

        trans_diff = {}
        trans_diff_mean = {}
        trans_diff_var = {}

        for feat in fcols_full:
            trans_diff[feat] = trans_pct_diff(
                full.glm_models, data_feat_unscaled, scalers,
                diff_feat=feat, pct1=90, pct2=10,
                fcols=fcols_full,
                treat_interactions=treat_interactions,
                silence_warning=True,
            )

            trans_10s = [
                trans_pct(
                    jk.glm_models, data_feat_unscaled, scalers,
                    diff_feat=feat, pct=10,
                    fcols=fcols_full,
                    treat_interactions=treat_interactions,
                    silence_warning=True,
                )
                for jk in jackknives
            ]
            trans_90s = [
                trans_pct(
                    jk.glm_models, data_feat_unscaled, scalers,
                    diff_feat=feat, pct=90,
                    fcols=fcols_full,
                    treat_interactions=treat_interactions,
                    silence_warning=True,
                )
                for jk in jackknives
            ]

            trans_diff_mean[feat] = np.mean(
                [trans_90s[i] - trans_10s[i] for i in range(n)], axis=0
            )
            trans_diff_var[feat] = (n - 1) * np.var(
                [trans_90s[i] - trans_10s[i] for i in range(n)], axis=0, ddof=0
            )

        return {
            "global_a": global_a,
            "global_a_mean": global_a_mean,
            "global_a_var": global_a_var,
            "trans_diff": trans_diff,
            "trans_diff_mean": trans_diff_mean,
            "trans_diff_var": trans_diff_var,
        }

    def loo_hmm(
        self,
        fcols: list[str],
        cell_line: Optional[str] = None,
        treatment: Optional[str] = None,
        dose=None,
        control: str | list[str] = ["control"],
        treat_interactions: Optional[list[str]] = None,
        run_by: Optional[str] = None,
        save_path: str = "runs/modelled/",
        **kwargs,
    ) -> dict:
        """
        Run the full LOO HMM workflow and return transition matrix results.

        If ``cell_line`` and ``treatment`` are both specified, a single model is
        fitted and a single results dict is returned.

        If either is unspecified, the method auto-infers the combinations to run:

        - ``cell_line=None``: if the metadata has no ``patient_id`` column, all data is
          treated as one cell line; if the column exists, the workflow runs once
          per unique cell line found in the metadata.
        - ``treatment=None``: runs once per unique non-control perturbation, combined
          with each unique dose for that perturbation. If no ``dose`` column exists,
          dose is ignored.

        Results are always returned as a nested dict keyed by
        ``(cell_line, treatment, dose)``. When only one combination is run,
        the dict has a single entry under that key.

        Parameters
        ----------
        fcols
            Base feature columns.
        cell_line
            Optional patient_id/cell-line identifier. If None, all cell lines are run.
        treatment
            Optional treatment perturbation name. If None, all non-control
            perturbations are run.
        dose
            Optional list of allowed doses. Only used when ``treatment`` is
            specified explicitly. When auto-inferring treatments, doses are
            iterated over automatically.
        control
            Names of perturbations considered controls.
        treat_interactions
            Features for which treatment interaction terms should be included.
            Interaction columns are automatically appended to ``fcols`` when
            evaluating transition differences.
        run_by
            Column in ``track_data`` defining the LOO grouping. Defaults to ``"set"`` if that column exists in ``track_data``, otherwise ``"experiment_id"``. Can be overridden explicitly.
        save_path
            Root directory for saving fitted runs. When multiple combinations
            are run, each is saved under ``{save_path}{cell_line}_{treatment}_{dose}/``.

        Returns
        -------
        dict
            Nested dict keyed by ``(cell_line, treatment, dose)`` tuples. Each value
            is a dict with keys:

            - ``global_a``: transition matrix from the full model.
            - ``global_a_mean``: jackknife mean of LOO transition matrices.
            - ``global_a_var``: jackknife variance of LOO transition matrices.
            - ``trans_diff``: per-feature 90th-vs-10th-percentile transition
              matrix difference from the full model.
            - ``trans_diff_mean``: jackknife mean of per-feature LOO differences.
            - ``trans_diff_var``: jackknife variance of per-feature LOO differences.

        @ Author: Linnea Hallin
        """
        metadata_df = self._get_metadata_df()
        control = [control] if isinstance(control, str) else list(control)
        has_patient_id = "patient_id" in metadata_df.columns
        has_dose = "dose" in metadata_df.columns
        run_by = run_by if run_by is not None else ("set" if "set" in self._track_data.columns else "experiment_id")

        # Filter metadata to only experiments actually present in track_data
        tracked_exps = self._track_data["exp"].unique()
        tracked_meta = metadata_df[metadata_df["experiment_id"].isin(tracked_exps)]

        # --- resolve cell lines to iterate over ---
        if cell_line is not None:
            cell_lines = [cell_line]
        elif has_patient_id:
            cell_lines = sorted(tracked_meta["patient_id"].dropna().unique().tolist())
        else:
            cell_lines = [None]  # sentinel: no cell-line filter

        # --- resolve treatments to iterate over ---
        if treatment is not None:
            treatments = [(treatment, dose)]
        else:
            if has_dose:
                combos = (
                    tracked_meta[["perturbation", "dose"]]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )
                treatments = [(p, [d]) for p, d in combos]
            else:
                treatments = [
                    (p, None)
                    for p in tracked_meta["perturbation"].unique().tolist()
                ]

        results = {}
        for cl in cell_lines:
            for treat, d in treatments:
                # build a safe subdirectory name when iterating
                cl_tag = cl if cl is not None else "all"
                d_tag = str(d[0]) if (d is not None and len(d) == 1) else str(d)
                sub_path = (
                    save_path
                    if (len(cell_lines) == 1 and len(treatments) == 1)
                    else f"{save_path}{cl_tag}_{treat}_{d_tag}/"
                )

                print(f"-- Running LOO HMM: cell_line={cl}, treatment={treat}, dose={d} --")

                results[(cl, treat, d_tag)] = self._loo_hmm_single(
                    treat,
                    cell_line=cl,
                    fcols=fcols,
                    dose=d,
                    control=control,
                    treat_interactions=treat_interactions,
                    run_by=run_by,
                    save_path=sub_path,
                    **kwargs,
                )

        return results

    # ------------------------------------------------------------------
    # Properties (read-only copies)
    # ------------------------------------------------------------------

    @property
    def subtable(self):
        """
        DataFrame listing all stacks with their file paths, experiment/ROI ids,
        metadata annotations, and tracking status.
        """
        return self._subtable.copy()

    @property
    def transition_matrix(self):
        """
        Global transition probability matrix fitted by the GLM-HMM.
        Shape ``(K, K)`` where ``K`` is the number of hidden states.
        Requires ``fit_hmm`` to have been called.
        """
        self._require_hmm()
        return self._transition_matrix.copy()

    @property
    def track_data(self):
        """
        Raw per-cell tracking DataFrame containing position, classification,
        and vascularity features for every tracked cell across all stacks.
        Requires ``run_tracking`` to have been called.
        """
        self._require_tracked()
        return self._track_data.copy()

    @property
    def data_feat(self):
        """
        Filtered feature DataFrame used as input to the GLM-HMM.
        Tracks that did not pass quality filters are excluded.
        Requires ``fit_hmm`` or ``create_and_filter_features`` to have been called.
        """
        self._require_feat()
        return self._data_feat.copy()

    @property
    def pi(self):
        """
        Initial state distribution vector of length ``K`` (number of hidden states).
        Requires ``fit_hmm`` to have been called.
        """
        self._require_hmm()
        return self._pi.copy()

    @property
    def gammas(self):
        """
        Posterior state probability DataFrame (gamma values) from the GLM-HMM
        E-step. One row per cell per timepoint, with one column per hidden state.
        Requires ``fit_hmm`` to have been called.
        """
        self._require_hmm()
        return self._gammas.copy()

    @property
    def video_paths(self):
        """
        List of output paths for all videos generated during this session via
        ``video_tracking`` or ``video_compare``.
        Requires ``run_tracking`` to have been called.
        """
        self._require_tracked()
        return self._video_paths.copy()

    # ------------------------------------------------------------------
    # Video generation
    # ------------------------------------------------------------------

    def video_compare(self, exp, roi, output: Optional[PathLike] = None):
        """
        Generate a side-by-side comparison video of CNN classification vs Viterbi
        decoded states for a given experiment and ROI.

        Parameters
        ----------
        exp
            Experiment ID to visualize.
        roi
            ROI index within the experiment.
        output
            Optional output path. If a directory, the file is written inside it.
            If omitted, defaults to ``<cwd>/gliotrace_videos/exp_{exp}/roi_{roi}/``.

        Returns
        -------
        Path
            Path to the generated video file.

        @ Author: André Lasses Armatowski
        """
        self._require_tracked()
        self._require_hmm()

        exp, roi = _validate_exp_roi(self._subtable, exp, roi)
        out_path = self._coerce_output_path(output, exp, roi)

        result = generate_video_compare(
            self._track_data, self._subtable, self._data_feat,
            exp, roi, self._channel_roles, out_path,
        )
        self._video_paths.append(result if result is not None else out_path)
        return result

    def video_tracking(self, exp, roi, output: Optional[PathLike] = None):
        """
        Generate a tracking video overlaying detected cell tracks on the raw
        fluorescence stack for a given experiment and ROI.

        Parameters
        ----------
        exp
            Experiment ID to visualize.
        roi
            ROI index within the experiment.
        output
            Optional output path. If a directory, the file is written inside it.
            If omitted, defaults to ``<cwd>/gliotrace_videos/exp_{exp}/roi_{roi}/``.

        @ Author: André Lasses Armatowski
        """
        self._require_tracked()

        exp, roi = _validate_exp_roi(self._subtable, exp, roi)
        out_path = self._coerce_output_path(output, exp, roi)

        result = generate_video(
            self._subtable, self._track_data,
            exp, roi, self._channel_roles, out_path,
        )
        self._video_paths.append(result if result is not None else out_path)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------

    def generate_summary_statistics(
        self,
        fcol,
        patient_id=None,
        set_id=None,
        treatment=None,
        mode="mean",
    ):
        """
        Compute per-experiment summary statistics for a single feature over time and ROI.

        Parameters
        ----------
        fcol
            Feature column to summarize.
        patient_id
            Optional patient filter.
        set_id
            Optional set filter.
        treatment
            Optional treatment filter (True/False/None).
        mode
            Aggregation mode: "mean", "median", "sum", "min", "max", "std", "count".

        Returns
        -------
        dict
            Mapping of ``experiment_id`` -> DataFrame of shape ``(time, roi)``
            containing the aggregated feature value at each timepoint and ROI.

        @ Author: André Lasses Armatowski
        """
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

    # ------------------------------------------------------------------
    # Sorting helpers
    # ------------------------------------------------------------------

    def patients(self, treated: bool | None = None) -> list[str]:
        """List unique patient identifiers, optionally filtered by treatment status."""
        st = self._subtable
        _require_cols(st, "patient_id", name="subtable")
        st = _apply_treated(st, treated)
        return sorted(st["patient_id"].dropna().astype(str).unique().tolist())

    def sets(self, patient: str, treated: bool | None = None) -> list[int]:
        """List unique set indices for a given patient."""
        st = self._subtable
        _require_cols(st, "patient_id", "set", name="subtable")
        st = _apply_patient(st, patient, name="subtable")
        st = _apply_treated(st, treated)
        return sorted(st["set"].dropna().astype(int).unique().tolist())

    def exps(
        self,
        patient: str | None = None,
        set: int | None = None,
        treated: bool | None = None,
    ) -> list[int]:
        """List unique experiment IDs, optionally filtered by patient, set, or treatment."""
        st = self._subtable
        _require_cols(st, "exp", name="stacktable")
        st = _apply_patient(st, patient, name="stacktable")
        st = _apply_set(st, set, patient=patient, name="stacktable")
        st = _apply_treated(st, treated)
        return sorted(st["exp"].dropna().astype(int).unique().tolist())

    # ------------------------------------------------------------------
    # Printing / debug
    # ------------------------------------------------------------------

    def print_configuration(self) -> None:
        """Print a human-readable summary of the current configuration and dataset scope."""
        print("\n=== Gliotrace configuration ===")
        st = self._subtable
        print(f"Number of stacks        : {len(st)}")

        def _show(xs, n=10):
            xs = list(xs)
            return xs if len(xs) <= n else xs[:n] + [f"... (n={len(xs)})"]

        print("Scope                   :")

        has_patient = "patient_id" in st.columns
        has_set = "set" in st.columns
        has_exp = "exp" in st.columns

        if has_patient and has_set and has_exp:
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
                sub = grp.loc[pid]
                for s in _show(sorted(sub.index.unique().tolist()), n=20):
                    if isinstance(s, str) and s.startswith("..."):
                        print(f"    {s}")
                        break
                    print(f"    Set {s}: exps {_show(sub.loc[s], n=12)}")

        elif has_patient and has_exp:
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

        print(f"Detection sensitivity   : {self._detection_sensitivity}")
        print(f"Detection backup        : {self._detection_backup}")
        print(f"Channel roles           : {self._channel_roles}")
        print(f"Feature columns         : {self._fcols}")
        print(f"HMM parameters          : {self._hmm_param}")

        if "perturbation" in st.columns:
            control = st.loc[st["is_treatment"] == False, "perturbation"].unique()
            print(f"Control                 : {'All' if len(control) != 1 else control[0]}")

        if (st["is_treatment"] == True).any():
            t_name = st.loc[st["is_treatment"] == True, "perturbation"].unique()
            t_dose = []
            if "treatment_dose" in st.columns:
                t_dose = st.loc[
                    st["is_treatment"] == True, "treatment_dose"
                ].dropna().unique()
            if len(t_dose) == 0:
                print(f"Treatment               : {t_name[0]}")
            else:
                print(f"Treatment               : {t_name[0]} (dose={t_dose[0]})")

        n_treat = int((st["is_treatment"] == True).sum())
        n_ctrl = int((st["is_treatment"] == False).sum())
        print(f"Stacks (control/treat)  : {n_ctrl} / {n_treat}")
        print("================================\n")

    def print_exp_roi(self):
        """Print a table of all experiment IDs and ROI indices currently loaded."""
        print(self._subtable[["exp", "roi"]])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_tracked(self):
        """Raise RuntimeError if tracking has not been completed."""
        if not self._tracked:
            raise RuntimeError("run_tracking must be called before this method.")

    def _require_hmm(self):
        """Raise RuntimeError if the GLM-HMM has not been fitted."""
        if not self._hmm:
            raise RuntimeError("fit_hmm must be called before this method.")

    def _require_feat(self):
        """Raise RuntimeError if features have not been constructed."""
        if not self._feat:
            raise RuntimeError(
                "create_and_filter_features or fit_hmm must be called before this method."
            )

    def _assert_same_rows_except_tracked(
        self, current: pd.DataFrame, loaded: pd.DataFrame
    ):
        """
        Assert that two subtables are identical aside from the ``tracked`` column.

        Used when resuming a partially tracked run via ``load_point`` to ensure
        the loaded checkpoint matches the current configuration.

        Raises
        ------
        ValueError
            If the shapes differ or any non-tracked column values disagree.
        """
        c = current.drop(columns=["tracked"], errors="ignore").reset_index(drop=True)
        l = loaded.drop(columns=["tracked"], errors="ignore").reset_index(drop=True)
        c = c.reindex(sorted(c.columns), axis=1)
        l = l.reindex(sorted(l.columns), axis=1)

        if c.shape != l.shape:
            raise ValueError(
                f"Subtable mismatch: shape current={c.shape}, loaded={l.shape}. "
                "Use GlioTrace.load_run(path) to initialize from a saved run."
            )
        if not c.equals(l):
            diff = (c != l) & ~(c.isna() & l.isna())
            first = diff.stack()
            first = first[first].index.tolist()
            where = first[0] if first else None
            raise ValueError(
                f"Subtable mismatch (ignoring 'tracked'). First difference at {where}"
            )

    def _coerce_output_path(self, output: Optional[PathLike], exp: int, roi: int):
        """
        Resolve and create an output path for a video file.

        If ``output`` is None, defaults to ``<cwd>/gliotrace_videos/exp_{exp}/roi_{roi}/``.
        If ``output`` has a file extension it is treated as a full file path;
        otherwise it is treated as a directory and created accordingly.

        Returns
        -------
        Path
            Resolved and created output path.
        """
        if output is None:
            out = Path.cwd() / "gliotrace_videos" / f"exp_{exp}" / f"roi_{roi}"
        else:
            out = Path(output)
        if out.suffix:
            out.parent.mkdir(parents=True, exist_ok=True)
        else:
            out.mkdir(parents=True, exist_ok=True)
        return out

    def _compare_hmm_and_cnn_class(self, threshold=0.1):
        """Internal helper to compare CNN and gamma classification."""
        self._require_tracked()
        self._require_hmm()

        softmax_cols = SOFTMAX_COLUMNS
        data_feat = self._data_feat.copy()
        gammas_long = self._gammas.copy()

        data_feat["state_label"] = data_feat["state_label"].apply(
            lambda x: softmax_cols[int(x) - 1]
        )

        maxprob = data_feat[softmax_cols].max(axis=1)
        data_sel = data_feat.loc[maxprob > threshold].copy()
        if data_sel.empty:
            print("No rows passed the CNN threshold.")
            return pd.DataFrame()

        merged = data_sel.merge(
            gammas_long[["exp", "roi", "cellID", "time"] + softmax_cols],
            on=["exp", "roi", "cellID", "time"],
            how="inner",
            validate="one_to_one",
            suffixes=("_cnn", "_gamma"),
        )

        gamma_cols = [f"{c}_gamma" for c in softmax_cols]
        merged["gamma_class"] = (
            merged[gamma_cols]
            .idxmax(axis=1)
            .str.replace(r"_gamma$", "", regex=True)
        )

        return (
            merged
            .groupby(["state_label", "gamma_class"])
            .size()
            .unstack(fill_value=0)
            .reindex(index=softmax_cols, columns=softmax_cols, fill_value=0)
        )

    # ------------------------------------------------------------------
    # Save and Load
    # ------------------------------------------------------------------

    def save_run(self, run_dir: str | Path) -> Path:
        """
        Save the current GlioTrace instance to disk.

        Parameters
        ----------
        run_dir
            Target directory. Created if it does not exist.

        Returns
        -------
        Path
            Resolved path to the saved run directory.

        @ Author: André Lasses Armatowski
        """
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tables").mkdir(exist_ok=True)

        manifest = {
            "format": "gliotrace-run-v1",
            "tracked": bool(self._tracked),
            "feat": bool(self._feat),
            "hmm": bool(self._hmm),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        cfg = {
            "detection_sensitivity": self._detection_sensitivity,
            "detection_backup": self._detection_backup,
            "channel_roles": self._channel_roles,
            "fcols": self._fcols,
            "hmm_param": self._hmm_param,
        }

        (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

        self._subtable.to_pickle(run_dir / "tables" / "subtable.pkl")

        if self._metadata is not None:
            self._metadata.to_pickle(run_dir / "tables" / "metadata.pkl")

        if self._track_data is not None:
            self._track_data.to_pickle(run_dir / "tables" / "track_data.pkl")

        if self._feat:
            if self._data_feat_unfilt is not None:
                self._data_feat_unfilt.to_pickle(run_dir / "tables" / "data_unfilt.pkl")
            if self._data_feat is not None:
                self._data_feat.to_pickle(run_dir / "tables" / "data_feat.pkl")

        if self._hmm:
            if self._gammas is not None:
                self._gammas.to_pickle(run_dir / "tables" / "gammas.pkl")
            if self._pi is not None:
                self._pi.to_pickle(run_dir / "tables" / "pi.pkl")
            if self._transition_matrix is not None:
                self._transition_matrix.to_pickle(
                    run_dir / "tables" / "transition_matrix.pkl"
                )
            joblib.dump(self.glm_models, run_dir / "glm_models.joblib")

        return run_dir

    @classmethod
    def load_run(cls, run_dir: str | Path) -> "GlioTrace":
        """
        Load a GlioTrace instance from a saved run directory.

        Parameters
        ----------
        run_dir
            Path to a directory produced by save_run.

        @ Author: André Lasses Armatowski
        """
        run_dir = Path(run_dir)

        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("format") != "gliotrace-run-v1":
            raise ValueError(f"Unknown run format: {manifest.get('format')}")

        cfg = json.loads((run_dir / "config.json").read_text())

        self = cls.__new__(cls)

        self._detection_sensitivity = cfg["detection_sensitivity"]
        self._detection_backup = cfg.get("detection_backup", None)
        self._channel_roles = cfg["channel_roles"]
        self._fcols = cfg["fcols"]
        self._hmm_param = cfg["hmm_param"]
        
        metadata_path = run_dir / "tables" / "metadata.pkl"
        self._metadata = pd.read_pickle(metadata_path) if metadata_path.exists() else None

        self._tracked = bool(manifest.get("tracked", False))
        self._feat = bool(manifest.get("feat", False))
        self._hmm = bool(manifest.get("hmm", False))
        self._video_paths = []

        self._subtable = pd.read_pickle(run_dir / "tables" / "subtable.pkl")

        track_path = run_dir / "tables" / "track_data.pkl"
        self._track_data = pd.read_pickle(track_path) if track_path.exists() else None

        self._data_feat = None
        self._data_feat_unfilt = None
        self._pi = None
        self._transition_matrix = None
        self._gammas = None
        self.glm_models = None

        if self._feat:
            data_feat_path = run_dir / "tables" / "data_feat.pkl"
            self._data_feat = (
                pd.read_pickle(data_feat_path) if data_feat_path.exists() else None
            )
            data_unfilt_path = run_dir / "tables" / "data_unfilt.pkl"
            self._data_feat_unfilt = (
                pd.read_pickle(data_unfilt_path) if data_unfilt_path.exists() else None
            )

        if self._hmm:
            pi_path = run_dir / "tables" / "pi.pkl"
            self._pi = pd.read_pickle(pi_path) if pi_path.exists() else None

            transition_path = run_dir / "tables" / "transition_matrix.pkl"
            self._transition_matrix = (
                pd.read_pickle(transition_path) if transition_path.exists() else None
            )

            gammas_path = run_dir / "tables" / "gammas.pkl"
            self._gammas = (
                pd.read_pickle(gammas_path) if gammas_path.exists() else None
            )

            glm_path = run_dir / "glm_models.joblib"
            self.glm_models = joblib.load(glm_path) if glm_path.exists() else None

        return self
