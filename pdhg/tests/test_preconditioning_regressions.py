"""Scaling safeguards and the solver's input-coordinate contract."""

from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from common.lp_problem import BoundedFormLP
from common.preconditioning import (
    apply_diagonal_preconditioners,
    pdlp_preconditioning,
    pock_chambolle_preconditioning,
    ruiz_preconditioning,
)

from pdhg import pdhg_tools, pdlp


def _pock_chambolle(problem: BoundedFormLP) -> BoundedFormLP:
    return pock_chambolle_preconditioning(problem, alpha=1.0)


@pytest.mark.parametrize(
    "scale", [ruiz_preconditioning, _pock_chambolle, pdlp_preconditioning]
)
def test_zero_and_tiny_norms_are_left_unscaled(
    scale: Callable[[BoundedFormLP], BoundedFormLP],
) -> None:
    """A diagonal example isolates each row and column's scaling decision.

    Zero, subthreshold, and threshold-sized entries must survive unchanged,
    even after ten Ruiz passes. The ordinary entry 4 should become 1. Exact
    comparison on tiny entries avoids an absolute tolerance hiding shrinkage.
    """
    matrix = np.diag([0.0, 1e-8, 1e-6, 4.0])
    problem = BoundedFormLP(
        matrix, np.zeros(4), np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)
    )
    scaled = scale(problem)
    assert scaled.diagonal_preconditioning is not None
    for diagonal in scaled.diagonal_preconditioning:
        np.testing.assert_array_equal(diagonal[:3], np.ones(3))
    np.testing.assert_array_equal(scaled.constraint_matrix[:3, :3], matrix[:3, :3])
    assert scaled.constraint_matrix[3, 3] == pytest.approx(1.0)


def _equality_problem() -> BoundedFormLP:
    # min 3*x with 4*x = 8, x >= 0: x*=2 and y*=3/4.
    return BoundedFormLP(
        np.array([[4.0]]),
        np.array([3.0]),
        np.array([8.0]),
        np.array([8.0]),
        np.zeros(1),
        np.full(1, np.inf),
    )


@pytest.mark.parametrize("already_scaled", [False, True])
@pytest.mark.parametrize("automatic_scaling", [False, True])
def test_nonzero_optimal_warm_start_stays_fixed(
    already_scaled: bool,
    automatic_scaling: bool,
) -> None:
    """A primal-dual optimum must remain fixed through one PDHG iteration.

    Nonzero starts expose missing forward coordinate transforms that zero
    starts hide. Prior scaling tests that the solver only undoes its own
    scaling, without changing or discarding the caller's recovery metadata.
    """
    problem = _equality_problem()
    x0, y0 = np.array([2.0]), np.array([0.75])
    if already_scaled:
        problem = apply_diagonal_preconditioners(
            problem, np.array([2.0]), np.array([3.0])
        )
        x0, y0 = x0 / 3.0, y0 / 2.0
    before_x, before_y = x0.copy(), y0.copy()
    metadata = problem.diagonal_preconditioning
    x, y = pdlp.PdlpSolver(automatic_scaling).solve(problem, (x0, y0), max_iterations=1)
    np.testing.assert_allclose(x, before_x, atol=1e-12)
    np.testing.assert_allclose(y, before_y, atol=1e-12)
    np.testing.assert_array_equal(x0, before_x)
    np.testing.assert_array_equal(y0, before_y)
    assert problem.diagonal_preconditioning is metadata


def test_automatically_scaled_solve_returns_input_coordinates() -> None:
    """Solve an already-scaled LP from zero, then explicitly recover its parent.

    This checks convergence as well as the fixed-point warm-start test above.
    Returning coordinates of the earliest ancestor would give x=2 rather than
    the input problem's x=2/3 and would fail input feasibility.
    """
    problem = apply_diagonal_preconditioners(
        _equality_problem(), np.array([2.0]), np.array([3.0])
    )
    x, y = pdlp.PdlpSolver().solve(problem, (np.zeros(1), np.zeros(1)), 10000)
    np.testing.assert_allclose(x, [2.0 / 3.0], atol=1e-7)
    np.testing.assert_allclose(y, [0.375], atol=1e-7)
    assert problem.primal_feasibility_status(x, 1e-6).feasible()
    np.testing.assert_allclose(
        problem.to_unpreconditioned_primal_variables(x), [2.0], atol=1e-7
    )
    np.testing.assert_allclose(
        problem.to_unpreconditioned_dual_variables(y), [0.75], atol=1e-7
    )


def test_scaled_feasibility_does_not_terminate_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small scaled residual must not mask a meaningful input residual.

    For x=1+1e-5 in the equality x=1, scaling the row by 1e-8 reduces the
    violation to 1e-13. Feed that iterate followed by the exact solution to
    isolate termination policy from the convergence speed of PDHG. The first
    check must continue rather than return or raise; the second must succeed.
    """
    problem = BoundedFormLP(
        np.ones((1, 1)),
        np.zeros(1),
        np.ones(1),
        np.ones(1),
        np.zeros(1),
        np.full(1, np.inf),
    )
    scaled = apply_diagonal_preconditioners(problem, np.array([1e-8]), np.ones(1))
    bad_x = np.array([1.0 + 1e-5])
    metrics = pdhg_tools.compute_optimality_metrics(scaled, bad_x, np.zeros(1))
    assert metrics.is_optimal(0.0, pdhg_tools.finite_row_bounds_norm(scaled), 1e-8)
    monkeypatch.setattr(pdlp, "pdlp_preconditioning", Mock(return_value=scaled))
    primal_step = Mock(side_effect=[bad_x, np.ones(1)])
    monkeypatch.setattr(pdhg_tools, "primal_update", primal_step)
    monkeypatch.setattr(pdhg_tools, "dual_update", Mock(return_value=np.zeros(1)))
    monkeypatch.setattr(pdhg_tools, "OPTIMALITY_CHECK_INTERVAL", 1)
    x, _ = pdlp.PdlpSolver().solve(problem, (np.zeros(1), np.zeros(1)), 2)
    assert primal_step.call_count == 2
    np.testing.assert_array_equal(x, np.ones(1))
