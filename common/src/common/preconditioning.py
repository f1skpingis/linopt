"""Diagonal scaling for the bounded-row LP formulation."""

import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF


def apply_diagonal_preconditioners(
    lp: lp_problem.BoundedFormLP, d1: ArrayF, d2: ArrayF
) -> lp_problem.BoundedFormLP:
    """For positive diagonal D1, D2, form A_scaled = D1 A D2.

    x_scaled = D2^-1 x, y_scaled = D1^-1 y, r_scaled = D2 r.
    Recover original coordinates with x = D2 x_scaled, y = D1 y_scaled,
    r = D2^-1 r_scaled. Objectives are unchanged. This applies supplied
    diagonals only; choosing Ruiz/Pock-Chambolle diagonals remains separate.
    """
    m, n = lp.constraint_matrix.shape
    if d1.shape != (m,) or d2.shape != (n,):
        raise ValueError("Expected row scaling (m,) and column scaling (n,).")
    if any(not np.all(np.isfinite(d)) or np.any(d <= 0) for d in (d1, d2)):
        raise ValueError("Scaling diagonals must be finite and strictly positive.")
    return lp_problem.BoundedFormLP(
        constraint_matrix=d1[:, None] * lp.constraint_matrix * d2,
        objective=d2 * lp.objective,
        row_lower=d1 * lp.row_lower,
        row_upper=d1 * lp.row_upper,
        variable_lower=lp.variable_lower / d2,
        variable_upper=lp.variable_upper / d2,
    )


# Keep the existing misspelled entry point usable during migration.
apply_diagonal_precoditioners = apply_diagonal_preconditioners


def pock_chambolle_preconditioning(
    lp: lp_problem.BoundedFormLP,
) -> lp_problem.BoundedFormLP:
    raise NotImplementedError(
        "Pock-Chambolle diagonal selection is not implemented yet."
    )


def ruiz_preconditioning(lp: lp_problem.BoundedFormLP) -> lp_problem.BoundedFormLP:
    raise NotImplementedError("Ruiz diagonal selection is not implemented yet.")
