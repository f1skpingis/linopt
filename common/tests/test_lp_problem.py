import numpy as np
import pytest

from common import lp_problem


@pytest.fixture()
def small_inequality_constrained_problem() -> lp_problem.LpProblem:
    # Convention is gx \geq h

    a = np.array([[0.0, 0.0, 1.0]])
    g = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, -1.0, 0.0],
        ]
    )
    b = np.array([100.0])
    h = np.array(
        [
            1.0,
            -10.0,
        ]
    )

    c = np.zeros(3)
    return lp_problem.LpProblem(
        constraint_matrix=a,
        rhs=b,
        objective=c,
        inequality_constraint_matrix=g,
        inequality_rhs=h,
        lower_bounds=np.array([0.0, -np.inf, -np.inf]),
        upper_bounds=np.array([np.inf, 10.0, np.inf]),
    )


def test_lp_problem_feasibility(
    small_inequality_constrained_problem: lp_problem.LpProblem,
) -> None:
    primal_feasible_solution = np.array(
        [
            1.0,
            9.0,
            100.0,
        ]
    )

    status = small_inequality_constrained_problem.primal_feasibility_status(
        primal_feasible_solution, tolerance=1e-6
    )
    assert status.feasible()
    assert status.lower_bounds == 0.0
    assert status.upper_bounds == 0.0
    assert status.equalities == 0.0
    assert status.inequalities == 0.0


@pytest.mark.parametrize(
    ("x", "field", "expected"),
    [
        ([-1.0, 9.0, 100.0], "lower_bounds", 1.0),
        ([1.0, 11.0, 100.0], "upper_bounds", 1.0),
        ([1.0, 9.0, 98.0], "equalities", 2.0),
        ([0.5, 9.0, 100.0], "inequalities", 0.5),
    ],
)
def test_lp_problem_violation_magnitudes(
    small_inequality_constrained_problem: lp_problem.LpProblem,
    x: list[float],
    field: str,
    expected: float,
) -> None:
    status = small_inequality_constrained_problem.primal_feasibility_status(
        np.array(x), tolerance=1e-6
    )
    assert not status.feasible()
    assert getattr(status, field) == pytest.approx(expected)


def test_primal_feasibility_without_inequalities() -> None:
    problem = lp_problem.LpProblem(
        constraint_matrix=np.empty((0, 1)),
        rhs=np.empty(0),
        objective=np.zeros(1),
    )
    status = problem.primal_feasibility_status(np.array([2.0]), tolerance=0.0)
    assert status.feasible()
    assert status.inequalities == 0.0


def test_primal_feasibility_uses_tolerance(
    small_inequality_constrained_problem: lp_problem.LpProblem,
) -> None:
    x = np.array([0.5, 9.0, 100.0])
    within_tolerance = small_inequality_constrained_problem.primal_feasibility_status(
        x, tolerance=0.5
    )
    outside_tolerance = small_inequality_constrained_problem.primal_feasibility_status(
        x, tolerance=0.49
    )

    assert within_tolerance.inequalities == outside_tolerance.inequalities == 0.5
    assert within_tolerance.feasible()
    assert not outside_tolerance.feasible()


def test_dual_feasibility_with_supplied_reduced_costs() -> None:
    problem = lp_problem.LpProblem(
        constraint_matrix=np.array([[1.0, 0.0, 0.0]]),
        rhs=np.array([0.0]),
        objective=np.array([2.0, 2.0, -1.0]),
        inequality_constraint_matrix=np.array([[0.0, 1.0, 0.0]]),
        inequality_rhs=np.array([0.0]),
        lower_bounds=np.array([-np.inf, 0.0, -np.inf]),
        upper_bounds=np.array([np.inf, np.inf, 1.0]),
    )
    status = problem.dual_feasibility_status(
        equality_multipliers=np.array([2.0]),
        inequality_multipliers=np.array([1.0]),
        reduced_costs=np.array([0.0, 1.0, -1.0]),
        tolerance=0.0,
    )
    assert status.feasible()
    assert status.stationarity == 0.0


def test_dual_feasibility_projects_reduced_costs() -> None:
    problem = lp_problem.LpProblem(
        constraint_matrix=np.empty((0, 1)),
        rhs=np.empty(0),
        objective=np.array([-1.0]),
        lower_bounds=np.array([0.0]),
    )
    status = problem.dual_feasibility_status(
        equality_multipliers=np.empty(0),
        inequality_multipliers=np.empty(0),
        tolerance=0.0,
    )
    assert status.stationarity == 1.0
    assert status.lower_bound_reduced_costs == 0.0
    assert not status.feasible()


@pytest.mark.parametrize(
    ("lower_bound", "upper_bound", "reduced_cost", "field"),
    [
        (0.0, np.inf, -1.0, "lower_bound_reduced_costs"),
        (-np.inf, 1.0, 1.0, "upper_bound_reduced_costs"),
        (-np.inf, np.inf, 1.0, "free_variable_reduced_costs"),
    ],
)
def test_dual_reduced_cost_violations(
    lower_bound: float, upper_bound: float, reduced_cost: float, field: str
) -> None:
    problem = lp_problem.LpProblem(
        constraint_matrix=np.empty((0, 1)),
        rhs=np.empty(0),
        objective=np.array([reduced_cost]),
        lower_bounds=np.array([lower_bound]),
        upper_bounds=np.array([upper_bound]),
    )
    status = problem.dual_feasibility_status(
        equality_multipliers=np.empty(0),
        inequality_multipliers=np.empty(0),
        reduced_costs=np.array([reduced_cost]),
        tolerance=0.0,
    )
    assert getattr(status, field) == 1.0
    assert not status.feasible()


def test_negative_inequality_multiplier_is_dually_infeasible() -> None:
    problem = lp_problem.LpProblem(
        constraint_matrix=np.empty((0, 1)),
        rhs=np.empty(0),
        objective=np.zeros(1),
        inequality_constraint_matrix=np.zeros((1, 1)),
        inequality_rhs=np.zeros(1),
    )
    status = problem.dual_feasibility_status(
        equality_multipliers=np.empty(0),
        inequality_multipliers=np.array([-0.5]),
        tolerance=0.0,
    )
    assert status.inequality_multipliers == 0.5
    assert status.stationarity == 0.0
    assert not status.feasible()
