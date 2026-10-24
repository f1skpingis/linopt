# PDHG learning baseline

This implementation follows Section 2 of the local newer PDLP paper,
[`2501.07018v2.pdf`](papers/2501.07018v2.pdf). It solves

```text
min c.T x
s.t. row_lower <= A x <= row_upper
     variable_lower <= x <= variable_upper
```

Use `common.lp_problem.BoundedFormLP`. All four bound vectors are explicit;
use `-np.inf` and `np.inf` for absent bounds. Equal row bounds encode an
equality. A row with two distinct finite bounds remains one row and has one
signed dual multiplier.

```python
import numpy as np
from common.lp_problem import BoundedFormLP
from pdhg.pdlp import PdlpSolver

problem = BoundedFormLP(
    constraint_matrix=np.array([[1., 1.]]),
    objective=np.array([1., 2.]),
    row_lower=np.array([1.]),
    row_upper=np.array([np.inf]),
    variable_lower=np.zeros(2),
    variable_upper=np.full(2, np.inf),
)
x, y = PdlpSolver().solve(problem, (np.zeros(2), np.zeros(1)), 10_000)
```

## Paper-to-code map

- Eq. (1): `BoundedFormLP`; `common.lp_problem.dual_domain_projection` implements the same
  sign-domain projection for row multipliers `y` and reduced costs `r`.
- The paper's `p(v; lower, upper)` is `bound_support`. Its negative part is
  **nonnegative**, `v_minus = -min(v, 0)`. Zero entries are masked to avoid
  multiplying zero by infinite bounds.
- Eq. (1): dual objective is `-p(-y; row bounds) - p(-r; variable bounds)`.
  We recover `r` by projecting `c - A.T y` onto `R`, as described in Section 2.
- Eqs. (3a), (3b): `primal_update` and `dual_update` in `pdhg_tools.py`.
  The latter projects onto `[-row_upper, -row_lower]`; it is no longer a
  projection of a prefix of `y` onto nonnegative numbers.
- Eq. (4): `PdhgStepSizeParameters` retains `tau = eta / omega` and
  `sigma = eta * omega`.

`PdlpSolver` retains its existing name but currently implements **baseline
PDHG**, with weight 1 and step size `0.9 / estimated_norm(A)`. Power iteration
is an estimate, not a certified upper bound. A zero operator uses step size 1.

The stopping policy remains a baseline relative L2 residual/gap check. The
row scale is the L2 norm of the largest finite absolute bound per row. This
is an explicit extension of the old RHS scale, not a reproduction of the
new paper's Section 7 polishing experiment (tight feasibility, 1% gap).
Residuals include variable-bound and dual-domain violations as well as row
feasibility and stationarity. Checks on a scaled LP are in scaled units;
recover original coordinates before assessing original-unit tolerances.

## Preconditioning and next exercises

`common.preconditioning.apply_diagonal_preconditioners` applies supplied,
strictly positive row and column diagonals:

```text
A_scaled = D1 A D2        c_scaled = D2 c
row_bounds_scaled = D1 row_bounds
variable_bounds_scaled = D2^-1 variable_bounds
x_original = D2 x_scaled
y_original = D1 y_scaled
r_original = D2^-1 r_scaled
```

The existing misspelled `apply_diagonal_preconditioners` name remains an alias.
Ruiz and Pock-Chambolle diagonal selection are still unimplemented exercises.
Adaptive step sizes, primal-weight updates, restarts, feasibility polishing,
and infeasibility detection are also left for subsequent work.

The earlier paper remains useful for its ablations and development of the
individual enhancements: [`2106.04756v2.pdf`](papers/2106.04756v2.pdf).
