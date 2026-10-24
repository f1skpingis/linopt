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

    total_d1 = (
        d1
        if lp.diagonal_preconditioning is None
        else d1 * lp.diagonal_preconditioning[0]
    )
    total_d2 = (
        d2
        if lp.diagonal_preconditioning is None
        else d2 * lp.diagonal_preconditioning[1]
    )

    return lp_problem.BoundedFormLP(
        constraint_matrix=d1[:, None] * lp.constraint_matrix * d2,
        objective=d2 * lp.objective,
        row_lower=d1 * lp.row_lower,
        row_upper=d1 * lp.row_upper,
        variable_lower=lp.variable_lower / d2,
        variable_upper=lp.variable_upper / d2,
        diagonal_preconditioning=(total_d1, total_d2),
    )


def pock_chambolle_preconditioning(
    lp: lp_problem.BoundedFormLP, alpha: float
) -> lp_problem.BoundedFormLP:
    """Applies the Pock-Chambolle diagonal preconditioning to the problem `lp`

    The diagonal preconditioners are the column and row `alpha` and `2 - alpha` norms,
    respectively. See `Diagonal Preconditioning` in the paper 2501.07018v2 for details.
    """
    if not 0 < alpha < 2:
        raise ValueError(
            "Choice of alpha would require calculating p-norm with negative p which is not well defined for any matrix A"
        )

    if lp.constraint_matrix.size == 0:
        return lp

    a_column_norms = np.sum(np.abs(lp.constraint_matrix) ** alpha, axis=0) ** (
        1 / alpha
    )
    a_row_norms = np.sum(np.abs(lp.constraint_matrix) ** (2.0 - alpha), axis=1) ** (
        1 / (2.0 - alpha)
    )

    # Leave zero/tiny norms unscaled rather than amplifying possible noise.
    # Their coefficients are retained; equilibration to unit norm is not promised.
    non_zero_tol = 1e-6
    non_zero_mask_d1 = a_row_norms > non_zero_tol
    non_zero_mask_d2 = a_column_norms > non_zero_tol

    d1 = np.ones_like(a_row_norms)
    d2 = np.ones_like(a_column_norms)

    d1[non_zero_mask_d1] = 1 / np.sqrt(a_row_norms[non_zero_mask_d1])
    d2[non_zero_mask_d2] = 1 / np.sqrt(a_column_norms[non_zero_mask_d2])

    return apply_diagonal_preconditioners(lp, d1, d2)


def ruiz_preconditioning(lp: lp_problem.BoundedFormLP) -> lp_problem.BoundedFormLP:
    """Apply one Ruiz pass using inverse square roots of row/column max norms.

    Repeated passes equilibrate nonzero rows and columns toward unit infinity
    norm, except those skipped by the small-norm safeguard below.
    """

    # If constraint matrix is empty
    if lp.constraint_matrix.size == 0:
        return lp

    a_column_max = np.max(np.abs(lp.constraint_matrix), axis=0)
    a_row_max = np.max(np.abs(lp.constraint_matrix), axis=1)

    # Leave zero/tiny norms unscaled rather than amplifying possible noise.
    # Their coefficients are retained; equilibration to unit norm is not promised.
    non_zero_tol = 1e-6
    non_zero_mask_d1 = a_row_max > non_zero_tol
    non_zero_mask_d2 = a_column_max > non_zero_tol

    d1 = np.ones_like(a_row_max)
    d2 = np.ones_like(a_column_max)

    d1[non_zero_mask_d1] = 1 / np.sqrt(a_row_max[non_zero_mask_d1])
    d2[non_zero_mask_d2] = 1 / np.sqrt(a_column_max[non_zero_mask_d2])

    return apply_diagonal_preconditioners(lp, d1, d2)


def pdlp_preconditioning(lp: lp_problem.BoundedFormLP) -> lp_problem.BoundedFormLP:
    """Applies the iterative preconditioning described in 2501.07018v2"""

    ruiz_iterations = 10
    while ruiz_iterations > 0:
        lp = ruiz_preconditioning(lp)
        ruiz_iterations -= 1

    pdlp_alpha = 1.0
    return pock_chambolle_preconditioning(lp, alpha=pdlp_alpha)
