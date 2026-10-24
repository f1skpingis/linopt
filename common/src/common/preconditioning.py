"""Diagonal scaling for the bounded-row LP formulation."""

import numpy as np
from common import lp_problem
from common.diagonal_scaling import DiagonalScaling, PreconditionTransformation


def apply_diagonal_preconditioners(
    lp: lp_problem.BoundedFormLP, scaling: DiagonalScaling
) -> lp_problem.BoundedFormLP:
    """For positive diagonal D1, D2, form A_scaled = D1 A D2.

    x_scaled = D2^-1 x, y_scaled = D1^-1 y, r_scaled = D2 r.
    Recover original coordinates with x = D2 x_scaled, y = D1 y_scaled,
    r = D2^-1 r_scaled. Objectives are unchanged. This applies supplied
    diagonals only; choosing Ruiz/Pock-Chambolle diagonals remains separate.
    """
    rows = scaling.row_factors
    columns = scaling.column_factors
    m, n = lp.constraint_matrix.shape
    if rows.shape != (m,) or columns.shape != (n,):
        raise ValueError("Expected row scaling (m,) and column scaling (n,).")

    # Apply only the new factors to the current LP; retain the composed map
    # for recovery all the way back to the problem before any scaling.
    total_scaling = scaling
    if lp.diagonal_preconditioning is not None:
        total_scaling = DiagonalScaling(
            row_factors=rows * lp.diagonal_preconditioning.row_factors,
            column_factors=columns * lp.diagonal_preconditioning.column_factors,
        )

    return lp_problem.BoundedFormLP(
        constraint_matrix=rows[:, None] * lp.constraint_matrix * columns[None, :],
        objective=columns * lp.objective,
        row_lower=rows * lp.row_lower,
        row_upper=rows * lp.row_upper,
        variable_lower=scaling.transform_primal(
            lp.variable_lower, direction=PreconditionTransformation.TO
        ),
        variable_upper=scaling.transform_primal(
            lp.variable_upper, direction=PreconditionTransformation.TO
        ),
        diagonal_preconditioning=total_scaling,
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

    return apply_diagonal_preconditioners(
        lp, DiagonalScaling(row_factors=d1, column_factors=d2)
    )


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

    return apply_diagonal_preconditioners(
        lp, DiagonalScaling(row_factors=d1, column_factors=d2)
    )


def pdlp_preconditioning(lp: lp_problem.BoundedFormLP) -> lp_problem.BoundedFormLP:
    """Applies the iterative preconditioning described in 2501.07018v2"""

    ruiz_iterations = 10
    while ruiz_iterations > 0:
        lp = ruiz_preconditioning(lp)
        ruiz_iterations -= 1

    pdlp_alpha = 1.0
    return pock_chambolle_preconditioning(lp, alpha=pdlp_alpha)
