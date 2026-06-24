import numpy as np
from collections.abc import Sequence

from coco_pipe.io import DataContainer
from coco_pipe.io.quality import group_labels


def families_for_analysis_unit(
    source: DataContainer,
    unit: dict,
    descriptor_families: Sequence[str] | None = None,
) -> list[str]:
    """QC grouping labels covered by one analysis unit (see coco_pipe.group_labels)."""
    group_by = (source.meta or {}).get("family_qc_group_by", "family")
    labels = group_labels(unit["container"], group_by)
    if descriptor_families and group_by == "family":
        wanted = {str(v) for v in descriptor_families}
        labels = [lbl for lbl in labels if lbl in wanted]
    return labels or ([str(unit["family"])] if unit.get("family") else [])


def apply_family_qc_mask(
    container: DataContainer,
    families: Sequence[str],
) -> tuple[DataContainer, np.ndarray]:
    """Filter observations using only the requested families' QC decisions."""
    bad_ids_by_family = dict(container.meta.get("family_qc_bad_ids", {}))
    if not bad_ids_by_family:
        indices = np.arange(container.X.shape[0], dtype=int)
        return container, indices
    if container.ids is None:
        raise ValueError("Family-scoped QC requires observation IDs.")
    bad_ids: set[str] = set()
    for family in families:
        bad_ids.update(str(value) for value in bad_ids_by_family.get(str(family), []))
    ids = np.asarray(container.ids, dtype=object).astype(str)
    keep_indices = np.flatnonzero(~np.isin(ids, list(bad_ids)))
    return container.isel(obs=keep_indices), keep_indices
