import logging

import numpy as np
from common import lp_problem

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
            pdhg_tools.DenseLinearOperator(k)
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

    stacked_op = pdhg_tools.vstack(
        pdhg_tools.DenseLinearOperator(a1), pdhg_tools.DenseLinearOperator(a2)
    )
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
    problem = lp_problem.LpProblem(
        constraint_matrix=a,
        rhs=b,
        inequality_constraint_matrix=g,
        inequality_rhs=h,
        lower_bounds=lower_bounds,
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


def test_lambda_projection() -> None:
    # This sets it up so variable one is free
    # Variable 2 is lower bounded
    # Variable 3 is upper bounded
    # Varaible 4 is both
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
    lam = pdhg_tools.lambda_projection(
        lower_bounds,
        upper_bounds,
        s,
    )
    assert np.allclose(lam, lam_expected)
