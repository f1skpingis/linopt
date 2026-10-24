import logging

import numpy as np
import pytest
from common import linear_algebra, lp_problem
from common.diagonal_scaling import DiagonalScaling, PreconditionTransformation
from common.linear_algebra import DenseLinearOperator
from common.preconditioning import (
    apply_diagonal_preconditioners,
    pock_chambolle_preconditioning,
    ruiz_preconditioning,
)
from scipy.optimize import linprog

# TODO(martin): Typo, call something else, maybe pdlp
from pdhg import pdhg_tools, pdlp

logger = logging.getLogger(__name__)


class TestPowerIteration:
    def test_dense(self) -> None:
        v1 = np.array([1.0, -1.0, 1.0])
        v2 = np.array([1.0, 1.0, 1.0])
        v3 = np.array([0.0, 0.0, 1.0])

        l1 = 1.0
        l2 = 2.0
        l3 = 10.0

        k = l1 * np.outer(v1, v1) + l2 * np.outer(v2, v2) + l3 * np.outer(v3, v3)

        spectral_norm = np.linalg.norm(k, 2)

        power_iteration_result = pdhg_tools.iterative_spectral_norm(
            DenseLinearOperator(k)
        )

        assert np.isclose(
            spectral_norm,
            power_iteration_result.spectral_norm,
        )

        assert (
            power_iteration_result.status == pdhg_tools.PowerIterationStatus.CONVERGED
        )
        logger.info(
            f"Power iteration converged after {power_iteration_result.iterations} iterations with norm {power_iteration_result.spectral_norm}"
        )


def test_vstack() -> None:

    a1 = np.array(
        [
            [1, 2],
            [3, 4],
        ]
    )
    a2 = np.array(
        [
            [4, 3],
            [2, 1],
        ]
    )

    stacked_op = linear_algebra.vstack(DenseLinearOperator(a1), DenseLinearOperator(a2))
    manually_stacked_op = np.array([*a1, *a2])

    test_vec_r = np.array([1, -2, 3, -4])
    test_vec = np.array([1, -2])

    assert np.allclose(manually_stacked_op @ test_vec, stacked_op.matvec(test_vec))
    assert np.allclose(
        manually_stacked_op.T @ test_vec_r, stacked_op.rmatvec(test_vec_r)
    )


class TestEasyPdhgSolve:
    num_variables = 2
    a = np.empty((0, num_variables))
    b = np.empty(0)
    g = np.array([[1.0, 1.0]])
    h = np.array([1.0])
    lower_bounds = np.zeros(num_variables)
    c = np.array([1.0, 1.0])
    problem = lp_problem.BoundedFormLP(
        constraint_matrix=g,
        row_lower=h,
        row_upper=np.full(1, np.inf),
        variable_lower=lower_bounds,
        variable_upper=np.full(num_variables, np.inf),
        objective=c,
    )

    def test_solve(self) -> None:
        solver = pdlp.PdlpSolver()
        x = np.array([10.0, 10.0])
        y = np.array([0.0])
        x, y = solver.solve(self.problem, (x, y), 10_000)

        assert np.isclose(self.problem.objective @ x, 1.0, atol=1e-3)
        assert np.all(x >= -1e-6)
        assert x.sum() >= 1.0 - 1e-3

    def test_solve_from_infeasible(self) -> None:
        solver = pdlp.PdlpSolver()
        x = np.array([-10.0, -10.0])
        y = np.array([0.0])
        x, y = solver.solve(self.problem, (x, y), 10_000)

        assert np.isclose(self.problem.objective @ x, 1.0, atol=1e-3)
        assert np.all(x >= -1e-6)
        assert x.sum() >= 1.0 - 1e-3


def test_dual_domain_projection() -> None:
    """Check the four sign domains used for both row duals and reduced costs.

    A free primal coordinate requires a zero reduced cost. A lower-only bound
    allows nonnegative reduced costs, an upper-only bound allows nonpositive
    ones, and two finite bounds allow either sign. Thus [1, 1, 1, 40] projects
    to [0, 1, 0, 40]. In particular, 40 is not clipped to the primal upper
    bound of 1: this is projection onto a dual sign domain, not a primal box.
    """
    lower_bounds = np.array(
        [
            -np.inf,
            -1.0,
            -np.inf,
            -1.0,
        ]
    )
    upper_bounds = np.array(
        [
            np.inf,
            np.inf,
            1.0,
            1.0,
        ]
    )

    s = np.array(
        [
            1.0,
            1.0,
            1.0,
            40.0,
        ]
    )
    lam_expected = np.array(
        [
            0.0,  # {0} is the \Lambda set for free vars
            1.0,  # R^+ for lower bounded vars
            0.0,  # R^- for upper bounded vars
            40.0,  # R for boxed vars
        ]
    )
    lam = lp_problem.dual_domain_projection(
        lower_bounds,
        upper_bounds,
        s,
    )
    assert np.allclose(lam, lam_expected)


class TestLpPreconditioning:
    a = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    b = np.array([1.0, 1.0])
    c = np.array([1.0, 0.0])

    simple_lp = lp_problem.BoundedFormLP(
        a, c, b, b, np.full(2, -np.inf), np.full(2, np.inf)
    )

    def test_preconditioning_with_arbitrary_matrices(self) -> None:
        """Isolate row scaling before testing simultaneous row/column scaling.

        With A = I and b = [1, 1], the equalities fix x = [1, 1]. Multiplying
        each equation and its bound by the same positive factor preserves this
        solution. D2 = I means neither variable coordinates nor costs change.
        """
        preconditioning_matrix_d1 = np.array([10.0, 20.0])
        preconditioning_matrix_d2 = np.array([1.0, 1.0])

        preconditioned_lp = apply_diagonal_preconditioners(
            self.simple_lp,
            DiagonalScaling(preconditioning_matrix_d1, preconditioning_matrix_d2),
        )

        # Since D2 is identity, variables should be the same
        feasible_x = np.array([1.0, 1.0])

        assert self.simple_lp.primal_feasibility_status(
            x=feasible_x, tolerance=1e-8
        ).feasible()
        assert preconditioned_lp.primal_feasibility_status(
            x=feasible_x, tolerance=1e-8
        ).feasible()

        assert np.allclose(
            np.diag(preconditioning_matrix_d1) @ self.simple_lp.constraint_matrix,
            preconditioned_lp.constraint_matrix,
        )
        assert np.allclose(
            preconditioning_matrix_d1 * self.simple_lp.row_lower,
            preconditioned_lp.row_lower,
        )
        assert np.allclose(preconditioned_lp.objective, self.simple_lp.objective)


def test_dual_update_all_row_types() -> None:
    """Check Eq. (3b) against hand-computed updates for each kind of row.

    A = I decouples the rows. Starting from y = 0 with sigma = 1/2,
    the supplied extrapolated point violates the lower, upper, equality and
    ranged rows. Their updates should have the corresponding signed values;
    the unconstrained row must contribute no dual multiplier.
    """
    # Row order: lower-only, upper-only, equality, ranged, unconstrained.
    problem = lp_problem.BoundedFormLP(
        np.eye(5),
        np.zeros(5),
        np.array([1.0, -np.inf, 2.0, -1.0, -np.inf]),
        np.array([np.inf, 3.0, 2.0, 1.0, np.inf]),
        np.full(5, -np.inf),
        np.full(5, np.inf),
    )
    op = DenseLinearOperator(problem.constraint_matrix)
    actual = pdhg_tools.dual_update(
        problem, op, np.array([0.0, 4.0, 0.0, 3.0, 20.0]), np.zeros(5), 0.5
    )
    # Lower: (1 - 0)/2 = .5; upper: (3 - 4)/2 = -.5;
    # equality: (2 - 0)/2 = 1; ranged, above its upper bound: (1 - 3)/2 = -1.
    # For the free row the interval projection cancels y_hat completely.
    np.testing.assert_allclose(actual, [0.5, -0.5, 1.0, -1.0, 0.0])


def test_bounded_solver_against_highs() -> None:
    """Compare the complete iteration and stopping logic with another solver.

    This LP combines all row types so a projection or row-order mistake can
    affect the answer. The equality gives x1 = 2 - x2, making the objective
    2 - 3*x2. The ranged row implies x2 <= 1.25, giving the unique minimizer
    (0.75, 1.25). HiGHS supplies an independent numerical reference.
    """
    # An equality, lower-only, upper-only and ranged row, and a free row.
    a = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]])
    problem = lp_problem.BoundedFormLP(
        a,
        np.array([1.0, -2.0]),
        np.array([2.0, 0.25, -np.inf, -0.5, -np.inf]),
        np.array([2.0, np.inf, 1.5, 0.75, np.inf]),
        np.array([-1.0, 0.0]),
        np.array([3.0, 3.0]),
    )
    # SciPy expects <= inequalities: negate a lower-only row and split the
    # ranged row into two inequalities. The free row imposes no constraint.
    reference = linprog(
        problem.objective,
        A_eq=a[:1],
        b_eq=[2.0],
        A_ub=np.array([-a[1], a[2], a[3], -a[3]]),
        b_ub=[-0.25, 1.5, 0.75, 0.5],
        bounds=[(-1.0, 3.0), (0.0, 3.0)],
        method="highs",
    )
    assert reference.success
    x0, y0 = np.zeros(2), np.zeros(5)
    x, y = pdlp.PdlpSolver().solve(problem, (x0, y0), 20000)
    np.testing.assert_allclose(x, reference.x, atol=1e-6)
    assert problem.primal_feasibility_status(x, 1e-6).feasible()
    metrics = pdhg_tools.compute_optimality_metrics(problem, x, y)
    # Agreement of objectives checks recovered reduced costs and dual signs,
    # beyond merely reaching a primal solution close to the reference.
    assert abs(metrics.primal_objective - metrics.dual_objective) < 1e-6
    # Solving should not overwrite the caller's warm-start arrays.
    np.testing.assert_array_equal(x0, np.zeros(2))
    np.testing.assert_array_equal(y0, np.zeros(5))


def test_preconditioning_give_correct_unpreconditioned_optimum() -> None:
    """Optimize preconditioned and unpreconditioned and check that solution
    is reported equal in the unpreconditioned coordinates.

    First we solve the LP with HiGHS and then we solve it, preconditioned
    with PDHG and compare original coordinate system variable values
    """
    # An equality, lower-only, upper-only and ranged row, and a free row.
    a = np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [2.0, 1.0]])
    problem = lp_problem.BoundedFormLP(
        constraint_matrix=a,
        objective=np.array([1.0, -2.0]),
        row_lower=np.array([2.0, 0.25, -np.inf, -0.5, -np.inf]),
        row_upper=np.array([2.0, np.inf, 1.5, 0.75, np.inf]),
        variable_lower=np.array([-1.0, 0.0]),
        variable_upper=np.array([3.0, 3.0]),
    )

    reference = linprog(
        problem.objective,
        A_eq=a[:1],
        b_eq=[2.0],
        A_ub=np.array([-a[1], a[2], a[3], -a[3]]),
        b_ub=[-0.25, 1.5, 0.75, 0.5],
        bounds=[(-1.0, 3.0), (0.0, 3.0)],
        method="highs",
    )
    assert reference.success
    x0, y0 = np.zeros(2), np.zeros(5)

    # Some arbitrary preconditioning.
    d1 = np.array([2.0, 1.3, 0.4, 0.95, 1.4])
    d2 = np.array([1.0, 2.0])

    preconditioned_problem = apply_diagonal_preconditioners(
        problem, DiagonalScaling(d1, d2)
    )

    assert preconditioned_problem.diagonal_preconditioning is not None
    assert np.allclose(preconditioned_problem.diagonal_preconditioning.row_factors, d1)
    assert np.allclose(
        preconditioned_problem.diagonal_preconditioning.column_factors, d2
    )

    x_preconditioned, y = pdlp.PdlpSolver(apply_preconditioning=False).solve(
        preconditioned_problem, (x0, y0), 20000
    )
    x = preconditioned_problem.diagonal_preconditioning.transform_primal(
        x_preconditioned, direction=PreconditionTransformation.FROM
    )
    np.testing.assert_allclose(x, reference.x, atol=1e-6)
    assert problem.primal_feasibility_status(x, 1e-6).feasible()
    metrics = pdhg_tools.compute_optimality_metrics(
        preconditioned_problem, x_preconditioned, y
    )
    # Agreement of objectives checks recovered reduced costs and dual signs,
    # beyond merely reaching a primal solution close to the reference.
    assert abs(metrics.primal_objective - metrics.dual_objective) < 1e-6
    # Solving should not overwrite the caller's warm-start arrays.
    np.testing.assert_array_equal(x0, np.zeros(2))
    np.testing.assert_array_equal(y0, np.zeros(5))


def test_bound_only_solver_and_dual_objective() -> None:
    """Exercise the zero-operator case with an analytic box-only optimum.

    Minimize 2*x1 - x2: choose x1 = -2 and x2 = 3 for objective -7. The free
    zero-cost coordinate stays at its initial zero. With no rows, ||A|| = 0
    and y is empty, so this also checks the step-size fallback. Reduced costs
    are simply c; their zero entries must not produce 0*infinity in the dual
    objective when evaluating absent bounds.
    """
    problem = lp_problem.BoundedFormLP(
        np.empty((0, 3)),
        np.array([2.0, -1.0, 0.0]),
        np.empty(0),
        np.empty(0),
        np.array([-2.0, -np.inf, -np.inf]),
        np.array([np.inf, 3.0, np.inf]),
    )
    x, y = pdlp.PdlpSolver().solve(problem, (np.zeros(3), np.empty(0)), 100)
    np.testing.assert_allclose(x, [-2.0, 3.0, 0.0])
    assert pdhg_tools.compute_dual_objective(problem, y, problem.objective) == -7.0


def test_rectangular_scaling_preserves_primal_and_dual() -> None:
    """Verify coordinate changes, not just the entries of the scaled matrix.

    A is 2-by-3 so confusing row and column diagonals cannot hide behind
    matching dimensions. Both diagonals are nontrivial: x scales by D2^-1,
    y by D1^-1, and r by D2. A feasible primal point must stay feasible and
    both objective expressions must be invariant under these changes.

    The chosen y and r lie in their sign domains but need not satisfy
    stationarity: objective invariance is an algebraic identity, not an
    optimality claim. The final assertion separately checks how c - A.T*y
    transforms, which is the quantity used to recover reduced costs.
    """
    problem = lp_problem.BoundedFormLP(
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        np.array([2.0, -1.0, 3.0]),
        np.array([1.0, -np.inf]),
        np.array([1.0, 3.0]),
        np.array([0.0, -np.inf, -2.0]),
        np.array([np.inf, 4.0, 5.0]),
    )
    d1, d2 = np.array([2.0, 3.0]), np.array([4.0, 5.0, 6.0])
    scaled = apply_diagonal_preconditioners(problem, DiagonalScaling(d1, d2))
    x, y, r = (
        np.array([1.0, 0.0, 2.0]),
        np.array([2.0, -1.0]),
        np.array([1.0, -2.0, 3.0]),
    )
    assert scaled.primal_feasibility_status(x / d2, 1e-12).feasible()
    np.testing.assert_allclose(scaled.objective @ (x / d2), problem.objective @ x)
    np.testing.assert_allclose(
        pdhg_tools.compute_dual_objective(scaled, y / d1, d2 * r),
        pdhg_tools.compute_dual_objective(problem, y, r),
    )
    np.testing.assert_allclose(
        scaled.objective - scaled.constraint_matrix.T @ (y / d1),
        d2 * (problem.objective - problem.constraint_matrix.T @ y),
    )


def test_invalid_scaling_diagonals() -> None:
    """Reject scaling inputs that violate the coordinate-change assumptions.

    Row/column lengths must match A. Zero makes recovery noninvertible,
    negative factors would require swapping bound directions, and infinite
    factors do not define a finite invertible scaling. This helper promises
    strictly positive finite diagonals, so each case must raise ValueError.
    """
    problem = lp_problem.StandardFormLP(
        np.eye(2), np.ones(2), np.zeros(2)
    ).to_bounded_form()
    for d1, d2 in [
        (np.ones(1), np.ones(2)),
        (np.ones(2), np.ones(3)),
        (np.array([0.0, 1.0]), np.ones(2)),
        (np.ones(2), np.array([-1.0, 1.0])),
        (np.ones(2), np.array([np.inf, 1.0])),
    ]:
        with pytest.raises(ValueError):
            apply_diagonal_preconditioners(problem, DiagonalScaling(d1, d2))


def test_simple_pock_chambolle_preconditioning() -> None:
    problem = lp_problem.BoundedFormLP(
        np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]),
        np.array([2.0, -1.0, 3.0]),
        np.array([1.0, -np.inf]),
        np.array([1.0, 3.0]),
        np.array([0.0, -np.inf, -2.0]),
        np.array([np.inf, 4.0, 5.0]),
    )
    column_norms_alpha_1 = np.array([1.0, 2.0, 1.0])
    row_norms_alpha_1 = np.array([2.0, 2.0])

    d1 = 1 / np.sqrt(row_norms_alpha_1)
    d2 = 1 / np.sqrt(column_norms_alpha_1)

    alpha_1_problem = pock_chambolle_preconditioning(problem, alpha=1.0)

    assert alpha_1_problem.diagonal_preconditioning is not None
    assert np.allclose(d1, alpha_1_problem.diagonal_preconditioning.row_factors)
    assert np.allclose(d2, alpha_1_problem.diagonal_preconditioning.column_factors)

    # sqrt(1.0) ** 2, (sqrt(1.0) + sqrt(1.0)) ** 2, sqrt(1.0) ** 2
    column_norms_alpha_15 = np.array([1.0, 2.0 ** (1 / 1.5), 1.0])
    row_norms_alpha_15 = np.array([4.0, 4.0])

    d1_15 = 1 / np.sqrt(row_norms_alpha_15)
    d2_15 = 1 / np.sqrt(column_norms_alpha_15)

    alpha_15_problem = pock_chambolle_preconditioning(problem, alpha=1.5)

    assert alpha_15_problem.diagonal_preconditioning is not None
    assert np.allclose(d1_15, alpha_15_problem.diagonal_preconditioning.row_factors)
    assert np.allclose(d2_15, alpha_15_problem.diagonal_preconditioning.column_factors)


def test_simple_ruiz_preconditioning() -> None:
    problem = lp_problem.BoundedFormLP(
        np.array([[10.0, -4.0, 0.0], [0.0, 1.0, 2.0]]),
        np.array([2.0, -1.0, 3.0]),
        np.array([1.0, -np.inf]),
        np.array([1.0, 3.0]),
        np.array([0.0, -np.inf, -2.0]),
        np.array([np.inf, 4.0, 5.0]),
    )
    column_inf_norms = np.array([10.0, 4.0, 2.0])
    row_inf_norms = np.array([10.0, 2.0])

    d1 = 1 / np.sqrt(row_inf_norms)
    d2 = 1 / np.sqrt(column_inf_norms)

    preconditioned_problem = ruiz_preconditioning(problem)

    assert preconditioned_problem.diagonal_preconditioning is not None
    assert np.allclose(d1, preconditioned_problem.diagonal_preconditioning.row_factors)
    assert np.allclose(
        d2, preconditioned_problem.diagonal_preconditioning.column_factors
    )


def test_adaptive_step_calculation() -> None:
    # Construct a really simple lp,
    # minimize 0, x <= 1.
    # Say we choose primal weight 1, we should see that a
    problem = lp_problem.BoundedFormLP(
        constraint_matrix=np.array([[1.0]]),
        objective=np.array([0.0]),
        row_lower=np.array([-np.inf]),
        row_upper=np.array([1.0]),
        variable_lower=np.array([-np.inf]),
        variable_upper=np.array([np.inf]),
    )
    # solver = pdlp.PdlpSolver()
    # (x,y) = solver.solve(problem, start_solution=(np.array([2.0]),np.array([2.0])), max_iterations=100)
    # assert x[0] <= 1.0
    # Take the point (x=0.0, y=1.0)
    # The set \mathcal{X} is all of R so
    # x' <- 10 * y = 10.0    (I have simplified the algorithm equations with ones and zeros inserted where they belong)
    # y' <- y - 20x' - 10*max(-1, y/10 - 2x') = 1 - 200 - max(-1, -19.9) = -199 - (-1) = -198
    # Omega norm is sqrt(10^2 + 198^2) ~ 198.25
    # Condition is eta_bar = -198.25 / ((-199) * 1 * (10)) ~ 19.8
    # Now, eta_bar is certainly not smaller than eta which is 10, so we should expect
    # a smaller step size to emerge from the adaptive step
    # The step size should be smaller that that which it, obviously, is not.
    # This we expect to see a smaller stepsize than one after one step
    step_size_params = pdhg_tools.PdhgStepSizeParameters(
        primal_weight=1.0, step_size=10.0
    )
    (_x, _y), step_size_used, _prediction = pdhg_tools.adaptive_pdhg_step(
        problem=problem,
        step_size_parameters=step_size_params,
        x=np.array([0.0]),
        y=np.array([1.0]),
        iteration=1,
        logging=True,
    )
    assert step_size_used.step_size < step_size_params.step_size
