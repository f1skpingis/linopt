from dataclasses import dataclass
from functools import cached_property

import jaxtyping
import numpy as np
from scipy import sparse

from common.numpy_type_aliases import ArrayF


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
