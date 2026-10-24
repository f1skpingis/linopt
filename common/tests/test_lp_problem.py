from typing import Sequence

import numpy as np
import pytest

from common.lp_problem import BoundedFormLP, StandardFormLP


def mixed_problem() -> BoundedFormLP:
    return BoundedFormLP(
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [2.0, -1.0, 0.0]]),
        np.zeros(3),
        np.array([100.0, 1.0, -10.0]),
        np.array([100.0, np.inf, np.inf]),
        np.array([0.0, -np.inf, -np.inf]),
        np.array([np.inf, 10.0, np.inf]),
    )


def test_feasibility() -> None:
    status = mixed_problem().primal_feasibility_status(np.array([1.0, 9.0, 100.0]), 0.0)
    assert status.feasible()
    assert status.row_bounds == status.variable_bounds == 0.0


@pytest.mark.parametrize(
    "x,field,expected",
    [
        ([-1.0, 9.0, 100.0], "variable_bounds", 1.0),
        ([1.0, 11.0, 100.0], "variable_bounds", 1.0),
        ([1.0, 9.0, 98.0], "row_bounds", 2.0),
        ([0.5, 9.0, 100.0], "row_bounds", 0.5),
    ],
)
def test_violation_magnitudes(x: Sequence[float], field: str, expected: float) -> None:
    status = mixed_problem().primal_feasibility_status(np.array(x), 1e-6)
    assert not status.feasible()
    assert getattr(status, field) == pytest.approx(expected)


def test_tolerance() -> None:
    lp = mixed_problem()
    x = np.array([0.5, 9.0, 100.0])
    assert lp.primal_feasibility_status(x, 0.5).feasible()
    assert not lp.primal_feasibility_status(x, 0.49).feasible()
    assert not lp.primal_feasibility_status(np.full(3, np.nan), 0.5).feasible()


def test_standard_form_embedding() -> None:
    lp = StandardFormLP(np.array([[1.0, 1.0]]), np.array([1.0]), np.array([2.0, 3.0]))
    bounded = lp.to_bounded_form()
    assert bounded.primal_feasibility_status(np.array([1.0, 0.0]), 0.0).feasible()
    assert not bounded.primal_feasibility_status(np.array([2.0, -1.0]), 0.0).feasible()
    np.testing.assert_array_equal(bounded.row_lower, lp.rhs)
    np.testing.assert_array_equal(bounded.row_upper, lp.rhs)


@pytest.mark.parametrize(
    "lower,upper", [(np.nan, 1.0), (2.0, 1.0), (np.inf, np.inf), (-np.inf, -np.inf)]
)
def test_invalid_bounds(lower: float, upper: float) -> None:
    with pytest.raises(ValueError):
        BoundedFormLP(
            np.eye(1),
            np.zeros(1),
            np.array([lower]),
            np.array([upper]),
            np.zeros(1),
            np.ones(1),
        )


def test_rhs_must_be_vector() -> None:
    with pytest.raises(ValueError):
        StandardFormLP(np.eye(2), np.ones((2, 1)), np.zeros(2))


def test_dual_feasibility_domains_and_stationarity() -> None:
    lp = BoundedFormLP(
        np.eye(4),
        np.array([2.0, -2.0, 3.0, 0.0]),
        np.array([0.0, -np.inf, 1.0, -np.inf]),
        np.array([np.inf, 0.0, 1.0, np.inf]),
        np.full(4, -np.inf),
        np.full(4, np.inf),
    )
    assert lp.dual_feasibility_status(lp.objective.copy(), 0.0).feasible()
    status = lp.dual_feasibility_status(-lp.objective, 0.0)
    assert status.row_multipliers == 2.0
    assert status.stationarity == 6.0
    assert not status.feasible()
    status = lp.dual_feasibility_status(lp.objective, 0.0, np.ones(4))
    assert status.reduced_costs == 1.0
    assert not status.feasible()
