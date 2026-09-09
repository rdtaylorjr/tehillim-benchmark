"""Still and White (1981) nuisance control: regress on covariates and keep the residual."""

import numpy as np


def residualize_on_covariates(response: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """OLS-residualizes response on one or more covariate columns, jointly (Still-White 1981)."""
    design = np.column_stack([np.ones(len(response)), covariates])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return np.asarray(response - design @ coefficients)


def residualize_by_length(distances: np.ndarray, length_diff: np.ndarray) -> np.ndarray:
    """OLS-residualizes distances on |length difference|, a Still-White (1981) nuisance fix."""
    return residualize_on_covariates(distances, length_diff.reshape(-1, 1))
