"""Shared column definitions for the ESMFold "Geo-24" geometric feature set.

Single, dependency-free source of truth for the 24 numeric geometric-feature
columns computed from ESMFold structures (pLDDT, compactness, secondary
structure, SASA, sequence descriptors, curvature/torsion).

This list was previously duplicated verbatim in three places:
  - ``gnn/data_utils.py``           (as ``DEFAULT_GEO_FEATURE_COLS``)
  - ``nn_pipeline/feature_dataset.py`` (as the ``GEOMETRIC_FEATURE_COLS`` concat)
  - ``debug_checks/check_feature_overlap*.py`` (as inline ``geo_feats``)

Kept torch-free on purpose: ``gnn.data_utils`` needs torch_geometric and
``nn_pipeline.feature_dataset`` needs torch, but the pandas-only debug scripts
must be able to import this list without either. Everyone imports from here.

IMPORTANT: this is deliberately NOT the same as
``features.geometric_features.get_feature_names()``, which appends a 25th
categorical ``ss_method`` marker. That marker is not a model input; including
it here would change ``geo_feature_dim`` from 24 to 25 and break the GNN/MLP
models.
"""

# Semantic sub-groups (used by nn_pipeline.feature_dataset for readability).
PLDDT_COLS = ["plddt_mean", "plddt_std", "plddt_min", "plddt_max"]
COMPACTNESS_COLS = [
    "radius_gyration", "end_to_end_distance", "max_pairwise_distance",
    "centroid_distance_mean", "centroid_distance_std",
]
SECONDARY_STRUCTURE_COLS = ["fraction_helix", "fraction_sheet", "fraction_coil"]
SASA_COLS = ["total_sasa", "hydrophobic_sasa", "fraction_hydrophobic_sasa"]
SEQUENCE_COLS = ["length", "net_charge", "mean_hydrophobicity", "hydrophobic_moment"]
CURVATURE_COLS = [
    "curvature_mean", "curvature_std", "curvature_max",
    "torsion_mean", "torsion_std",
]

# Flat 24-column list (order matters — it defines the geo feature dimension).
GEO_FEATURE_COLS = (
    PLDDT_COLS + COMPACTNESS_COLS + SECONDARY_STRUCTURE_COLS
    + SASA_COLS + SEQUENCE_COLS + CURVATURE_COLS
)

assert len(GEO_FEATURE_COLS) == 24, "Geo feature set must be 24 numeric columns"
