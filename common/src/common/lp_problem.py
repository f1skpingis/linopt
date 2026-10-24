"""LP formulations with explicit, solver-independent mathematical contracts."""

from dataclasses import dataclass
from functools import cached_property

from common.linear_algebra import DenseLinearOperator, LinearOperator
import numpy as np
from scipy import sparse

from common.numpy_type_aliases import ArrayF
from common.diagonal_scaling import DiagonalScaling


def _validate_matrix_objective(a: ArrayF, c: ArrayF) -> None:
    if a.ndim != 2 or c.shape != (a.shape[1],):
        raise ValueError("Expected matrix (m, n) and objective (n,).")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(c)):
        raise ValueError("Matrix and objective must be finite.")


def _validate_bounds(lower: ArrayF, upper: ArrayF, size: int) -> None:
    if lower.shape != (size,) or upper.shape != (size,):
        raise ValueError(f"Expected bound vectors of shape ({size},).")
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(np.isposinf(lower))
        or np.any(np.isneginf(upper))
        or np.any(lower > upper)
    ):
        raise ValueError(
            "Bounds must satisfy -inf <= lower <= upper <= inf, with no +inf lower or -inf upper."
        )


@dataclass(frozen=True)
class StandardFormLP:
    """Minimize c.T x subject to A x = b and x >= 0.

    Used by the current simplex and interior-point implementations.
    """

    constraint_matrix: ArrayF
    rhs: ArrayF
    objective: ArrayF

    def __post_init__(self) -> None:
        _validate_matrix_objective(self.constraint_matrix, self.objective)
        if self.rhs.shape != (self.constraint_matrix.shape[0],) or not np.all(
            np.isfinite(self.rhs)
        ):
            raise ValueError("Expected a finite RHS vector of shape (m,).")

    @property
    def num_variables(self) -> int:
        return self.objective.size

    @cached_property
    def sparse_constraint_matrix(self) -> sparse.csr_array:
        return sparse.csr_array(self.constraint_matrix)

    def to_bounded_form(self) -> "BoundedFormLP":
        """Embed standard form without changing coordinates or adding rows."""
        return BoundedFormLP(
            self.constraint_matrix,
            self.objective,
            self.rhs,
            self.rhs,
            np.zeros(self.num_variables),
            np.full(self.num_variables, np.inf),
        )


@dataclass(frozen=True)
class PrimalFeasibilityStatus:
    variable_bounds: float
    row_bounds: float
    tolerance: float

    def feasible(self) -> bool:
        return max(self.variable_bounds, self.row_bounds) <= self.tolerance


def dual_domain_projection(lower: ArrayF, upper: ArrayF, values: ArrayF) -> ArrayF:
    """Project onto Y or R in Section 2, Eq. (1) of the newer PDLP paper.

    Free bounds require zero; lower-only requires nonnegative; upper-only
    requires nonpositive; two finite bounds allow either sign.
    """
    free = np.isneginf(lower) & np.isposinf(upper)
    lower_only = np.isfinite(lower) & np.isposinf(upper)
    upper_only = np.isneginf(lower) & np.isfinite(upper)
    result = values.copy()
    result[free] = 0.0
    result[lower_only] = np.maximum(0.0, result[lower_only])
    result[upper_only] = np.minimum(0.0, result[upper_only])
    return result


@dataclass(frozen=True)
class DualFeasibilityStatus:
    row_multipliers: float
    reduced_costs: float
    stationarity: float
    tolerance: float

    def feasible(self) -> bool:
        return (
            max(self.row_multipliers, self.reduced_costs, self.stationarity)
            <= self.tolerance
        )


@dataclass(frozen=True)
class BoundedFormLP:
    """Minimize c.T x with row_lower <= A x <= row_upper,
    variable_lower <= x <= variable_upper (new PDLP paper, Section 2, Eq. 1).

    Equal row bounds encode equalities. Infinite bounds encode absent bounds.
    Explicit bounds avoid any implicit nonnegativity convention.
    """

    constraint_matrix: ArrayF
    objective: ArrayF
    row_lower: ArrayF
    row_upper: ArrayF
    variable_lower: ArrayF
    variable_upper: ArrayF

    # Accumulated coordinate map, if this problem has been preconditioned.
    diagonal_preconditioning: DiagonalScaling | None = None

    def __post_init__(self) -> None:
        _validate_matrix_objective(self.constraint_matrix, self.objective)
        m, n = self.constraint_matrix.shape
        _validate_bounds(self.row_lower, self.row_upper, m)
        _validate_bounds(self.variable_lower, self.variable_upper, n)
        if self.diagonal_preconditioning is not None and (
            self.diagonal_preconditioning.row_factors.shape != (m,)
            or self.diagonal_preconditioning.column_factors.shape != (n,)
        ):
            raise ValueError("Scaling dimensions must match the constraint matrix.")

    @property
    def num_variables(self) -> int:
        return self.objective.size

    @cached_property
    def lin_op_a(self) -> LinearOperator:
        # Should be easy to switch out for Sparse linear operator
        return DenseLinearOperator(self.constraint_matrix)

    def primal_feasibility_status(
        self, x: ArrayF, tolerance: float
    ) -> PrimalFeasibilityStatus:
        if x.shape != (self.num_variables,):
            raise ValueError("Expected x of shape (n,).")
        if not np.all(np.isfinite(x)):
            return PrimalFeasibilityStatus(np.inf, np.inf, tolerance)
        ax = self.constraint_matrix @ x
        return PrimalFeasibilityStatus(
            float(
                np.max(
                    np.maximum(self.variable_lower - x, x - self.variable_upper),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.maximum(self.row_lower - ax, ax - self.row_upper), initial=0.0
                )
            ),
            tolerance,
        )

    def dual_feasibility_status(
        self,
        row_multipliers: ArrayF,
        tolerance: float,
        reduced_costs: ArrayF | None = None,
    ) -> DualFeasibilityStatus:
        """Check y in Y, r in R and c - A.T y = r (Eq. 1).

        If omitted, recover r by projection onto R. This checks dual feasibility,
        not complementarity or optimality with a particular primal point.
        """
        if row_multipliers.shape != self.row_lower.shape:
            raise ValueError("Expected row multipliers of shape (m,).")
        if reduced_costs is not None and reduced_costs.shape != self.objective.shape:
            raise ValueError("Expected reduced costs of shape (n,).")
        if not np.all(np.isfinite(row_multipliers)) or (
            reduced_costs is not None and not np.all(np.isfinite(reduced_costs))
        ):
            return DualFeasibilityStatus(np.inf, np.inf, np.inf, tolerance)
        raw = self.objective - self.constraint_matrix.T @ row_multipliers
        if reduced_costs is None:
            reduced_costs = dual_domain_projection(
                self.variable_lower, self.variable_upper, raw
            )
        return DualFeasibilityStatus(
            float(
                np.max(
                    np.abs(
                        row_multipliers
                        - dual_domain_projection(
                            self.row_lower, self.row_upper, row_multipliers
                        )
                    ),
                    initial=0.0,
                )
            ),
            float(
                np.max(
                    np.abs(
                        reduced_costs
                        - dual_domain_projection(
                            self.variable_lower, self.variable_upper, reduced_costs
                        )
                    ),
                    initial=0.0,
                )
            ),
            float(np.max(np.abs(raw - reduced_costs), initial=0.0)),
            tolerance,
        )
