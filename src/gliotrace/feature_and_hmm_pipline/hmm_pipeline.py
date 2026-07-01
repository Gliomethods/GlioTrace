from gliotrace.feature_and_hmm_pipline.feature_construction import feature_construction
from gliotrace.feature_and_hmm_pipline.hmm_glm import hmm_glm
from gliotrace.feature_and_hmm_pipline.format_data import format_data, add_universal_time_to_gammas
from gliotrace.feature_and_hmm_pipline.clean_tracks import filter_features_3
from gliotrace.feature_and_hmm_pipline.viterbi_paths import viterbi_paths_all_tracks, map_viterbi_t_to_time_and_merge

from gliotrace.initalize_class.defaults import HARD_CODED_FEATURES, NON_SCALE_COLUMNS, SOFTMAX_COLUMNS

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

import numpy as np
import pandas as pd


def _init_sticky_A(K, stay=0.95):
    "Intial A, can be changed"
    move = (1.0 - stay) / (K - 1)
    A = np.full((K, K), move, dtype=float)
    np.fill_diagonal(A, stay)
    return A


def hmm_pipeline(data, fcols, hmm_param, filter_by = None, scalers = None):
    """
    Prepares features and runs HMM model with given features and logits from the CNN

    Author: Linnea Hallin, André Lasses Armatowski (adapted)
    """

    if fcols is None or len(fcols) == 0:
        raise ValueError("fcols must be a non-empty list of feature columns")

    if (scalers is not None) and (not all([scaler in fcols for scaler in scalers.keys()])):
        missing_cols = set(scalers.keys()) - set(fcols)
        raise ValueError(f"Scalers provided for columns not in fcols: {missing_cols}")

    if filter_by is not None:
        idx = np.ones(len(data), dtype=bool)
        for col, values in filter_by.items():
            if col not in data.columns:
                raise ValueError(f"Column '{col}' specified in filter_by not found in data")
            idx = idx & data[col].isin(values)
        data = data.loc[list(idx)].copy()
        if data.empty:
            raise ValueError("No data left after filtering by filter_by. Check the filter_by values provided.")

    # ----------- Create features -----------
    print("--- Creating features ---")
    frames = []
    for exp in data["exp"].unique():
        exp_data = data.loc[data["exp"] == exp].copy()
        features_exp = feature_construction(exp_data)
        frames.append(features_exp)

    data_feat = pd.concat(frames, ignore_index=True)

    data_feat_unfiltered = data_feat.copy()

    # ----------- Filter tracks on non-NA and size -----------
    data_feat = filter_features_3(
        data_feat,
        fcols=fcols,
        hard_coded_features=HARD_CODED_FEATURES,
        min_timepoints=2,
        verbose=True,
    )
    if data_feat is None:
        raise RuntimeError("No usable tracks after filtering")
    
    # ---------- Constant columns warning ---------------
    exclude = set(NON_SCALE_COLUMNS) | set(HARD_CODED_FEATURES)
    check_cols = [
        c for c in fcols if c in data_feat.columns and c not in exclude]

    const = []
    for c in check_cols:
        # count distinct non-NaN values
        nunq = data_feat[c].nunique(dropna=True)
        if nunq <= 1:
            const.append(c)

    if const:
        vals = {c: data_feat[c].dropna().iloc[0] if data_feat[c].notna(
        ).any() else None for c in const}
        raise ValueError(
            f"Constant feature columns found: {const}. Values: {vals}")

    # ----------- Define number of states -----------
    softmax_columns = SOFTMAX_COLUMNS
    n_states = len(softmax_columns)

    # ----------- Create initial A -----------
    # NOTE: Can be changed
    print("--- Intialize transitions ---")
    A = _init_sticky_A(n_states, stay=0.95)
    #smoothed_data = smooth_tracks(data_feat);
    #A = count_label_transitions(smoothed_data)

    # ----------- Scale and format -----------
    print("--- Final Preperations for GLM-HMM ---")

    exclude = set(NON_SCALE_COLUMNS) | set(HARD_CODED_FEATURES)
    scale_cols = [c for c in fcols if c not in exclude]

    if scalers is not None:
        for col, scaler in scalers.items():
            if col in data_feat.columns:
                data_feat.loc[:, col] = scaler.transform(data_feat[[col]])
                if col in scale_cols:
                    scale_cols.remove(col)
                else: 
                    print(f"Warning: Column {col} provided in scalers is not usually scaled. Check if this is intentional.")

    if len(scale_cols) > 0:
        data_feat.loc[:, scale_cols] = StandardScaler(
        ).fit_transform(data_feat[scale_cols])

    # Put data into correct format for HMM (trajectories-only version)
    track_data, cnn_outputs = format_data(
        trajectories=data_feat,
        columns=fcols,
        softmax_cols=softmax_columns,
    )

    # ----------- hyperparameters for glm ----------
    max_iter = hmm_param["em_iter"]
    glm_iter = hmm_param["glm_iter"]
    eps = hmm_param["eps"]

    # ----------- Initialize pi -----------
    # NOTE: Can be changed
    pi0 = np.ones(n_states, dtype=float) / float(n_states)

    # ----------- Fit HMM -----------
    print("--- Running GLM-HMM ---")
    pi, glm_models, A_global, gammas = hmm_glm(
        track_data,
        cnn_outputs,
        pi=pi0,
        max_iter=max_iter,
        glm_iters=glm_iter,
        eps_conv=eps,
        A=A,
        state_names=softmax_columns,
        patience=5
    )

    # Align gamma and data_feat times
    gammas = add_universal_time_to_gammas(gammas, data_feat)

    print("--- Computing Viterbi Paths ---")

    viterbi_df = viterbi_paths_all_tracks(
        trajectories=track_data,
        cnn_outputs_log=cnn_outputs,
        pi=np.asarray(pi),
        glm_models=glm_models,
        K=n_states
    )

    data_feat = map_viterbi_t_to_time_and_merge(
        data_feat, viterbi_df, K=n_states)

    return data_feat_unfiltered, data_feat, pi, glm_models, A_global, gammas

