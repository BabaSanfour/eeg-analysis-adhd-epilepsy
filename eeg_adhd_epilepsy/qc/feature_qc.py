"""Quality control logic for structured feature extraction."""

import logging

import numpy as np
import pandas as pd
from coco_pipe.descriptors import KNOWN_FAMILY_TOKENS
from coco_pipe.descriptors.qc import aggregate_family_qc, classify_descriptor_columns
from coco_pipe.io import DataContainer
from coco_pipe.io.quality import (
    GROUP_BY_COLUMN,
    QCResult,
    compute_feature_missingness,
    drop_epoch_outliers,
    drop_subject_outliers,
)

logger = logging.getLogger(__name__)


def _build_grouped_qc_summary(
    scoring_container: DataContainer,
    family_masks: dict[str, np.ndarray],
    group_by: str,
) -> pd.DataFrame:
    """Family-level QC summary, each family on the rows kept for that family.

    ``aggregate_family_qc`` reports per family; when ``group_by`` is finer than
    ``family``, a family's "kept rows" are the rows clean across *all* of that
    family's groups (logical AND of the relevant masks).
    """
    descriptor_names = (
        np.asarray(scoring_container.coords.get("feature", []), dtype=object).astype(str).tolist()
    )
    feature_schema = scoring_container.feature_schema()
    classification = classify_descriptor_columns(
        descriptor_names, known_families=KNOWN_FAMILY_TOKENS, feature_schema=feature_schema
    )
    column_family = classification["family"].fillna("unknown").astype(str).to_numpy()
    column_group = classification[GROUP_BY_COLUMN.get(group_by, "family")].fillna("unknown").astype(str).to_numpy()
    matrix = np.asarray(scoring_container.X)
    n_rows = matrix.shape[0]
    summaries = []
    for family in dict.fromkeys(column_family.tolist()):
        column_indices = np.flatnonzero(column_family == family)
        if column_indices.size == 0:
            continue
        groups_in_family = dict.fromkeys(column_group[column_indices].tolist())
        masks = [family_masks[group] for group in groups_in_family if group in family_masks]
        keep = np.logical_and.reduce(masks) if masks else np.ones(n_rows, dtype=bool)
        columns = [str(descriptor_names[index]) for index in column_indices]
        kept_rows = matrix[np.flatnonzero(keep)][:, column_indices]
        summaries.append(
            aggregate_family_qc(
                pd.DataFrame(kept_rows, columns=columns),
                columns,
                known_families=KNOWN_FAMILY_TOKENS,
                feature_schema=classification.iloc[column_indices].reset_index(drop=True),
            )
        )
    if not summaries:
        return aggregate_family_qc(
            pd.DataFrame(matrix, columns=list(descriptor_names)),
            list(descriptor_names),
            known_families=KNOWN_FAMILY_TOKENS,
            feature_schema=classification,
        )
    return pd.concat(summaries, ignore_index=True)


def compute_grouped_qc_masks(
    scoring_container: DataContainer,
    subject_col: str,
    qc_config: dict,
) -> tuple[dict[str, np.ndarray], QCResult]:
    """Compute grouped quality control masks based on the provided configuration."""
    outlier = qc_config.get("outlier", {})
    z_threshold = float(outlier.get("z_threshold", 5.0))
    epoch_fraction = float(outlier.get("epoch_outlier_fraction", 0.30))
    subject_fraction = float(outlier.get("subject_outlier_fraction", 0.20))
    group_by = str(outlier.get("group_by", "family"))
    min_obs = qc_config.get("min_obs")

    subject_values = np.asarray(
        scoring_container.coords.get(subject_col, scoring_container.ids),
        dtype=object,
    )
    subject_aggregated = pd.Index(subject_values.astype(str)).nunique() == len(subject_values)

    subject_masks, subject_result = drop_subject_outliers(
        scoring_container,
        z_threshold=z_threshold,
        outlier_fraction_threshold=subject_fraction,
        subject_col=subject_col,
        group_by=group_by,
    )
    assert isinstance(subject_masks, dict)

    epoch_dropped: dict[str, list] = {}
    epoch_threshold: float | None = None
    epoch_fraction_used: float | None = None
    if subject_aggregated:
        family_masks = dict(subject_masks)
    else:
        epoch_masks, epoch_result = drop_epoch_outliers(
            scoring_container,
            z_threshold=z_threshold,
            outlier_fraction_threshold=epoch_fraction,
            subject_col=subject_col,
            group_by=group_by,
            min_obs=None,
        )
        assert isinstance(epoch_masks, dict)
        family_masks = {
            family: epoch_masks[family]
            & subject_masks.get(family, np.ones_like(epoch_masks[family]))
            for family in epoch_masks
        }
        epoch_dropped = epoch_result.per_family_dropped
        epoch_threshold = z_threshold
        epoch_fraction_used = epoch_fraction

    per_family_dropped = {
        group: [
            *epoch_dropped.get(group, []),
            *subject_result.per_family_dropped.get(group, []),
        ]
        for group in family_masks
    }
    if min_obs is not None:
        for group, mask in family_masks.items():
            kept = int(mask.sum())
            if kept < int(min_obs):
                logger.warning(
                    "Family-scoped QC left %d observation(s) for %s, below "
                    "min_obs=%d; keeping them (min_obs is enforced at "
                    "extraction, not decode).",
                    kept,
                    group,
                    int(min_obs),
                )

    n_obs = scoring_container.X.shape[0]
    n_subjects = pd.Index(subject_values.astype(str)).nunique()
    meta = scoring_container.meta or {}
    feature_names = (
        np.asarray(scoring_container.coords.get("feature", []), dtype=object).astype(str).tolist()
    )
    result = QCResult(
        n_rows_entering_qc=meta.get("n_rows_entering_qc"),
        n_dropped_nan_inf=int(meta.get("n_dropped_nan_inf", 0)),
        n_dropped_extreme=int(meta.get("dropped_extreme_rows", 0)),
        n_obs_in=n_obs,
        n_obs_out=n_obs,
        n_subjects_in=n_subjects,
        n_subjects_out=n_subjects,
        epoch_drop_threshold=epoch_threshold,
        epoch_outlier_fraction_threshold=epoch_fraction_used,
        subject_drop_threshold=z_threshold,
        subject_outlier_fraction_threshold=subject_fraction,
        per_family_dropped=per_family_dropped,
        subject_outlier_burden=subject_result.subject_outlier_burden,
        feature_missingness=compute_feature_missingness(
            pd.DataFrame(scoring_container.X, columns=feature_names),
            feature_names,
        ),
        feature_columns_dropped=meta.get("dropped_feature_columns"),
        thresholds={
            "epoch_z_threshold": epoch_threshold,
            "epoch_outlier_fraction_threshold": epoch_fraction_used,
            "subject_z_threshold": z_threshold,
            "subject_outlier_fraction_threshold": subject_fraction,
            "group_by": group_by,
            "subject_aggregated": bool(subject_aggregated),
        },
    )
    result.family_qc = _build_grouped_qc_summary(scoring_container, family_masks, group_by)
    return family_masks, result
