# Shared LP representations

`StandardFormLP(constraint_matrix, rhs, objective)` means exactly
`min c.T x, A x = b, x >= 0`. The current simplex and IPM solvers accept it.

`BoundedFormLP(constraint_matrix, objective, row_lower, row_upper,
variable_lower, variable_upper)` means `min c.T x` subject to bounded rows
and bounded variables. Bounds are explicit vectors, permitting infinities;
equalities have matching finite row bounds. PDHG accepts this form.

The classes have separate contracts and do not inherit from one another.
Both validate dimensions and finite coefficients; bounded form also validates
bound order and infinity conventions. Supply NumPy arrays with vector shape
`(n,)` or `(m,)`, not column matrices. Frozen dataclasses prevent field
reassignment, but their array contents should be treated as immutable.

`standard.to_bounded_form()` embeds a standard-form problem without changing
variables or dual coordinates. A general conversion in the other direction
is not implemented: it would need variable shifts/splitting, slack variables,
objective offsets, and solution recovery. The existing Netlib helper only
adds slack/surplus variables and retains its existing limitations.

See `pdhg/README.md` for paper conventions, scaling, and the learning roadmap.

`common.diagonal_scaling.DiagonalScaling` holds named `row_factors` and
`column_factors`, validates positive finite vectors, and owns the coordinate
transformations. Use `transform_primal` or `transform_dual` with the required
keyword `direction=PreconditionTransformation.TO` (divide by the factors) or
`.FROM` (multiply). Identity scaling uses vectors of ones.

`BoundedFormLP.diagonal_preconditioning` optionally stores this object as the
accumulated map from the problem before preprocessing. The solver keeps a
separate map for only its own preprocessing: warm starts, returned solutions,
and optimality checks are always in the supplied problem's coordinates.
