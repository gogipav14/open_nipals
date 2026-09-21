"""
SIMCA (Soft Independent Modelling of Class Analogy) module.

Provides SIMCA classification with:
- Per-class PCA models
- Hotelling's T² and DModX for classification
- R² and Q² metrics for component selection
- Built-in scaling with pre-scaled data detection
- Multi-class support with configurable unknown handling
"""

from .simca import SIMCA, SIMCAClass, ComponentSelector
from .metrics import (
    calc_r2_x,
    calc_r2_y,
    calc_r2_cumulative_pca,
    calc_r2_cumulative_pls,
    calc_press,
    calc_q2,
    calc_q2_cumulative_pca,
    calc_q2_cumulative_pls,
)
from .cross_validation import (
    KFoldCV,
    LeaveOneOutCV,
    VenetianBlindsCV,
    cross_val_predict_pca,
    cross_val_predict_pls,
)

__all__ = [
    # Main classifier
    "SIMCA",
    "SIMCAClass",
    "ComponentSelector",
    # Metrics
    "calc_r2_x",
    "calc_r2_y",
    "calc_r2_cumulative_pca",
    "calc_r2_cumulative_pls",
    "calc_press",
    "calc_q2",
    "calc_q2_cumulative_pca",
    "calc_q2_cumulative_pls",
    # Cross-validation
    "KFoldCV",
    "LeaveOneOutCV",
    "VenetianBlindsCV",
    "cross_val_predict_pca",
    "cross_val_predict_pls",
]
