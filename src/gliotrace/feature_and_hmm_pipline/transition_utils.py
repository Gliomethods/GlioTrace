import numpy as np
import pandas as pd
from scipy.special import softmax as scipy_softmax
from gliotrace.initalize_class.defaults import NON_SCALE_COLUMNS

def trans_mat_from_model(glm_models, feature_values, silence_warning=False):
    """
    Compute a transition probability matrix from GLM-HMM models at given feature values.

    Each state's GLM produces logits for transitions from that state to all next states.
    A row-wise softmax converts those logits into transition probabilities.

    Parameters
    ----------
    glm_models
        List of fitted GLM models, one per HMM state. Each model must expose
        ``intercept_`` and ``coef_`` attributes (e.g. sklearn ``LogisticRegression``).
    feature_values
        1-D array of feature values at which to evaluate the transition probabilities.
        The order must match the feature order used during model fitting.
    silence_warning
        If False (default), prints a reminder that feature order cannot be verified
        automatically.

    Returns
    -------
    trans_matrix : np.ndarray, shape (K, K)
        Row-stochastic transition probability matrix.
    logits_matrix : np.ndarray, shape (K, K)
        Raw logits before softmax.
    intercepts : np.ndarray, shape (K, K)
        Intercept terms for each state-to-state transition.
    coefs : np.ndarray, shape (K, K, F)
        Coefficient arrays for each state-to-state transition and feature.

    @ Author: Linnea Hallin
    """
    if not silence_warning:
        print(
            "Warning: there is no way to check if the feature order matches "
            "the one in the models. Be careful."
        )

    if len(feature_values) != glm_models[0].n_features_in_:
        raise ValueError(
            "Length of feature_values does not match number of features in the model."
        )

    k = len(glm_models)
    f = len(feature_values)

    intercepts = np.zeros((k, k))
    coefs = np.zeros((k, k, f))

    for i in range(k):
        intercepts[i, :] = glm_models[i].intercept_
        coefs[i, :, :] = glm_models[i].coef_

    logits_matrix = intercepts + np.dot(coefs, feature_values)
    trans_matrix = scipy_softmax(logits_matrix, axis=1)

    return trans_matrix, logits_matrix, intercepts, coefs


def scale_data(data_feat_unscaled, scalers, non_scale_columns=NON_SCALE_COLUMNS):
    """
    Apply fitted scalers to feature columns, leaving non-scaled columns unchanged.

    Parameters
    ----------
    data_feat_unscaled
        DataFrame of raw (unscaled) feature values.
    scalers
        Mapping of feature name -> fitted ``StandardScaler``.
    non_scale_columns
        Column names that should be passed through without scaling.

    Returns
    -------
    pd.DataFrame
        Copy of the input with scaled feature columns.

    @ Author: Linnea Hallin
    """
    data_feat_scaled = data_feat_unscaled.copy()
    for feat, scaler in scalers.items():
        if feat not in non_scale_columns:
            data_feat_scaled[feat] = scaler.transform(data_feat_unscaled[[feat]])
    return data_feat_scaled


def feat_q(data_feat, pct, fcol):
    """
    Return the requested percentile value for one feature column.

    Parameters
    ----------
    data_feat
        DataFrame containing the feature.
    pct
        Percentile to compute (0-100).
    fcol
        Feature column name.

    Returns
    -------
    float

    @ Author: Linnea Hallin
    """
    return np.percentile(data_feat[fcol], pct)


def mean_feat_interaction(data_feat_scaled, fcols, treat_interactions):
    """
    Compute mean feature values across the dataset for baseline model evaluation.

    All features in ``fcols`` are set to their dataset mean. Interaction columns
    (``{feat}_treat``) are also included for any feature in ``treat_interactions``.
    The returned dictionary preserves insertion order, which determines the feature
    vector passed to ``trans_mat_from_model``.

    Parameters
    ----------
    data_feat_scaled
        DataFrame of scaled feature values.
    fcols
        Base feature column names.
    treat_interactions
        Features for which a ``{feat}_treat`` interaction column exists.

    Returns
    -------
    dict
        Ordered mapping of feature name -> mean value.

    @ Author: Linnea Hallin
    """
    means = {}
    for feat in fcols:
        means[feat] = data_feat_scaled[feat].mean()
    for feat in treat_interactions:
        name = f"{feat}_treat"
        means[name] = data_feat_scaled[name].mean()
    return means


def trans_pct(
    glm_models,
    data_feat_unscaled,
    scalers,
    diff_feat,
    fcols,
    pct,
    treat_interactions=None,
    non_scale_columns=NON_SCALE_COLUMNS,
    silence_warning=False,
):
    """
    Compute the transition matrix when one feature is fixed at a chosen percentile.

    All other features are held at their dataset mean. The behaviour for the focal
    feature depends on its type:

    - ``is_treatment``: interpreted as a binary flag; pct < 50 sets it to 0
      (control) and zeros all interaction terms; pct >= 50 sets it to 1
      (treated) and sets interaction terms to the treated-group feature mean.
    - Other binary features: set to their min (pct < 50) or max (pct >= 50).
    - Continuous features: set to the requested percentile of the scaled data.

    Parameters
    ----------
    glm_models
        List of fitted per-state GLM models.
    data_feat_unscaled
        DataFrame of raw feature values used to compute percentiles and fit scalers.
    scalers
        Mapping of feature name -> fitted ``StandardScaler``.
    diff_feat
        The feature to vary; all others are held at their mean.
    fcols
        All feature column names (base + any interaction columns).
    pct
        Percentile (0-100) at which to set ``diff_feat``.
    treat_interactions
        Features with treatment interaction columns.
    non_scale_columns
        Columns that should not be scaled.
    silence_warning
        Passed to ``trans_mat_from_model``.

    Returns
    -------
    np.ndarray, shape (K, K)
        Transition probability matrix evaluated at the specified feature values.

    @ Author: Linnea Hallin
    """
    treat_interactions = treat_interactions or []

    if diff_feat not in data_feat_unscaled.columns:
        raise ValueError(f"{diff_feat} is not a column in data_feat_unscaled.")
    if (diff_feat not in scalers) and (diff_feat not in non_scale_columns):
        raise ValueError(f"No scaler found for {diff_feat}.")

    data_feat_scaled = scale_data(
        data_feat_unscaled, scalers, non_scale_columns=non_scale_columns
    )
    feat_values = mean_feat_interaction(data_feat_scaled, fcols, treat_interactions)

    if diff_feat == "is_treatment":
        if pct / 100.0 < 0.5:
            feat_values[diff_feat] = 0
            for treat_feat in treat_interactions:
                feat_values[f"{treat_feat}_treat"] = 0
        else:
            feat_values[diff_feat] = 1
            for treat_feat in treat_interactions:
                feat_values[f"{treat_feat}_treat"] = data_feat_scaled[
                    data_feat_scaled["is_treatment"] == 1
                ][treat_feat].mean()

    elif data_feat_unscaled[diff_feat].nunique() == 2:
        values = data_feat_unscaled[diff_feat].unique()
        if pct / 100.0 < 0.5:
            feat_values[diff_feat] = values.min()
        else:
            feat_values[diff_feat] = values.max()
        if diff_feat in treat_interactions:
            feat_values[f"{diff_feat}_treat"] = feat_values[diff_feat]

    else:
        feat_values[diff_feat] = feat_q(data_feat_scaled, pct, diff_feat)
        if diff_feat in treat_interactions:
            feat_values[f"{diff_feat}_treat"] = feat_values[diff_feat]

    feature_values = np.array(list(feat_values.values()))
    trans_mat, _, _, _ = trans_mat_from_model(
        glm_models, feature_values, silence_warning=silence_warning
    )
    return trans_mat


def trans_pct_diff(
    glm_models,
    data_feat_unscaled,
    scalers,
    diff_feat,
    fcols,
    pct1=90,
    pct2=10,
    treat_interactions=None,
    **kwargs,
):
    """
    Compare transition matrices between two percentile values of a feature.

    Computes the transition matrix at ``pct1`` and ``pct2`` via ``trans_pct``
    and returns their difference. The default (90th minus 10th percentile)
    captures the effect of moving from a low to a high value of ``diff_feat``
    on state transition probabilities.

    Parameters
    ----------
    glm_models
        List of fitted per-state GLM models.
    data_feat_unscaled
        DataFrame of raw feature values.
    scalers
        Mapping of feature name -> fitted ``StandardScaler``.
    diff_feat
        The feature whose effect is being evaluated.
    fcols
        All feature column names.
    pct1
        Upper percentile (default 90).
    pct2
        Lower percentile (default 10).
    treat_interactions
        Features with treatment interaction columns.
    **kwargs
        Passed to ``trans_pct``.

    Returns
    -------
    np.ndarray, shape (K, K)
        Element-wise difference of the two transition matrices (pct1 minus pct2).

    @ Author: Linnea Hallin
    """
    treat_interactions = treat_interactions or []

    t1 = trans_pct(
        glm_models, data_feat_unscaled, scalers, diff_feat, fcols, pct1,
        treat_interactions=treat_interactions, **kwargs,
    )
    t2 = trans_pct(
        glm_models, data_feat_unscaled, scalers, diff_feat, fcols, pct2,
        treat_interactions=treat_interactions, **kwargs,
    )
    return t1 - t2


def trans_mat_jackknife(trans_matrix_list):
    """
    Compute jackknife mean and variance for a list of transition matrices.

    Parameters
    ----------
    trans_matrix_list
        List of transition matrices (np.ndarray, shape (K, K)) from LOO replicates.

    Returns
    -------
    mean_trans_mat : np.ndarray, shape (K, K)
        Element-wise mean across replicates.
    var_trans_mat : np.ndarray, shape (K, K)
        Jackknife variance: ``(n - 1) * population_variance``.

    @ Author: Linnea Hallin
    """
    n = len(trans_matrix_list)
    if n == 0:
        raise ValueError("Empty list of transition matrices.")
    mean_trans_mat = np.mean(trans_matrix_list, axis=0)
    var_trans_mat = (n - 1) * np.var(trans_matrix_list, axis=0, ddof=0)
    return mean_trans_mat, var_trans_mat