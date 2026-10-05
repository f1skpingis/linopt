from dataclasses import dataclass
from functools import cached_property

import jaxtyping
import numpy as np
from scipy import sparse

from common.numpy_type_aliases import ArrayF


@dataclass(frozen=True)
class PrimalFeasibilityStatus:
    lower_bounds: float
    upper_bounds: float
    equalities: float
    inequalities: float
    tolerance: float

    def feasible(self) -> bool:
        return all(
            violation <= self.tolerance
            for violation in (
                self.lower_bounds,
                self.upper_bounds,
                self.equalities,
                self.inequalities,
            )
        )


@dataclass(frozen=True)
class DualFeasibilityStatus:
    inequality_multipliers: float
    lower_bound_reduced_costs: float
    upper_bound_reduced_costs: float
    free_variable_reduced_costs: float
    stationarity: float
    tolerance: float

    def feasible(self) -> bool:
        return all(
            violation <= self.tolerance
            for violation in (
                self.inequality_multipliers,
                self.lower_bound_reduced_costs,
                self.upper_bound_reduced_costs,
                self.free_variable_reduced_costs,
                self.stationarity,
            )
        )


@dataclass(frozen=True)
class LpProblem:
    constraint_matrix: jaxtyping.Float[ArrayF, "m n"]
    rhs: jaxtyping.Float[ArrayF, " m"]
    objective: jaxtyping.Float[ArrayF, " n"]

    inequality_constraint_matrix: jaxtyping.Float[ArrayF, "q n"] | None = None
    inequality_rhs: jaxtyping.Float[ArrayF, " q"] | None = None

    lower_bounds: jaxtyping.Float[ArrayF, " n"] | None = None
    upper_bounds: jaxtyping.Float[ArrayF, " n"] | None = None

    @property
    def num_variables(self) -> int:
        return len(self.objective)

    @cached_property
    def sparse_constraint_matrix(self) -> sparse.csr_array:
        return sparse.csr_array(self.constraint_matrix)

    @cached_property
    def effective_inequality_constraint_matrix(self) -> ArrayF:
        if self.inequality_constraint_matrix is None:
            return np.empty((0, self.num_variables))
        return self.inequality_constraint_matrix

    @cached_property
    def effective_inequality_rhs(self) -> ArrayF:
        if self.inequality_rhs is None:
            return np.empty(0)
        return self.inequality_rhs

    @cached_property
    def effective_lower_bounds(self) -> ArrayF:
        if self.lower_bounds is None:
            return np.full(self.num_variables, -np.inf)
        return self.lower_bounds

    @cached_property
    def effective_upper_bounds(self) -> ArrayF:
        if self.upper_bounds is None:
            return np.full(self.num_variables, np.inf)
        return self.upper_bounds

    @cached_property
    def num_inequality_constraints(self) -> int:
        return self.effective_inequality_rhs.size

    def primal_feasibility_status(
        self, x: ArrayF, tolerance: float
    ) -> PrimalFeasibilityStatus:
        return PrimalFeasibilityStatus(
            lower_bounds=float(
                np.max(np.maximum(self.effective_lower_bounds - x, 0.0), initial=0.0)
            ),
            upper_bounds=float(
                np.max(np.maximum(x - self.effective_upper_bounds, 0.0), initial=0.0)
            ),
            equalities=float(
                np.max(np.abs(self.constraint_matrix @ x - self.rhs), initial=0.0)
            ),
            inequalities=float(
                np.max(
                    np.maximum(
                        self.effective_inequality_rhs
                        - self.effective_inequality_constraint_matrix @ x,
                        0.0,
                    ),
                    initial=0.0,
                )
            ),
            tolerance=tolerance,
        )

    def dual_feasibility_status(
        self,
        equality_multipliers: ArrayF,
        inequality_multipliers: ArrayF,
        tolerance: float,
        reduced_costs: ArrayF | None = None,
    ) -> DualFeasibilityStatus:
        """Check c - A.T @ y_eq - G.T @ y_ineq = r with y_ineq >= 0.

        Reduced costs r must be nonnegative for lower-only bounds, nonpositive
        for upper-only bounds, and zero for free variables. When omitted, r is
        chosen as the projection of c - A.T @ y_eq - G.T @ y_ineq onto these
        sets. Equality multipliers are unrestricted.
        """
        raw_reduced_costs = (
            self.objective
            - self.constraint_matrix.T @ equality_multipliers
            - self.effective_inequality_constraint_matrix.T @ inequality_multipliers
        )
        finite_lower = np.isfinite(self.effective_lower_bounds)
        finite_upper = np.isfinite(self.effective_upper_bounds)
        lower_only = finite_lower & ~finite_upper
        upper_only = ~finite_lower & finite_upper
        free = ~finite_lower & ~finite_upper

        if reduced_costs is None:
            reduced_costs = raw_reduced_costs.copy()
            reduced_costs[lower_only] = np.maximum(reduced_costs[lower_only], 0.0)
            reduced_costs[upper_only] = np.minimum(reduced_costs[upper_only], 0.0)
            reduced_costs[free] = 0.0

        return DualFeasibilityStatus(
            inequality_multipliers=float(
                np.max(np.maximum(-inequality_multipliers, 0.0), initial=0.0)
            ),
            lower_bound_reduced_costs=float(
                np.max(np.maximum(-reduced_costs[lower_only], 0.0), initial=0.0)
            ),
            upper_bound_reduced_costs=float(
                np.max(np.maximum(reduced_costs[upper_only], 0.0), initial=0.0)
            ),
            free_variable_reduced_costs=float(
                np.max(np.abs(reduced_costs[free]), initial=0.0)
            ),
            stationarity=float(
                np.max(np.abs(raw_reduced_costs - reduced_costs), initial=0.0)
            ),
            tolerance=tolerance,
        )
