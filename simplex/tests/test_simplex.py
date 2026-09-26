import time
from typing import TYPE_CHECKING

import numpy as np
import pytest
from common import lp_problem
from common.numpy_type_aliases import ArrayI
from pytest_unordered import unordered
from scipy import sparse

from simplex import (
    dual_simplex,
    linear_algebra,
    pivoting_strategy,
    primal_simplex,
)
from simplex_util import InfeasibleLpError, SimplexCyclingError, UnboundedLpError

if TYPE_CHECKING:
    from common.numpy_type_aliases import ArrayF


class TestPrimalPivoting:
    @pytest.mark.parametrize(
        "piv_strat, expected",
        [
            (pivoting_strategy.BlandsRule, 5),
            (pivoting_strategy.DantzigsRule, 6),
        ],
    )
    def test_entering_selection(
        self, piv_strat: type[pivoting_strategy.PrimalPivotingStrategy], expected: int
    ) -> None:
        reduced_costs = np.array([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        non_basic_vars = np.array([0, 4, 5, 6, 3, 1, 2])

        result = piv_strat().pick_entering_index(reduced_costs, non_basic_vars)
        assert result == expected

    @pytest.mark.parametrize(
        "piv_strat",
        [
            pivoting_strategy.BlandsRule,
            pivoting_strategy.DantzigsRule,
        ],
    )
    def test_exiting_selection(
        self, piv_strat: type[pivoting_strategy.PrimalPivotingStrategy]
    ) -> None:
        x_basis = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        d = np.array(
            [1.0, -4.0, 1.0, 8.0, 1.0, 12.0],
        )

        basic_vars = np.array([10, 2, 4, 20, 7, 6])
        exiting_index = piv_strat().pick_exiting_index(
            basic_vars,
            x_basis,
            d,
            linear_algebra.ProductFormFactorization(np.eye(len(basic_vars))),
        )
        exiting_variable = basic_vars[exiting_index]

        assert exiting_index == 5
        assert exiting_variable == 6


class TestSteepestEdgeRule:
    def test_entering_selection_uses_scaled_reduced_costs(self) -> None:
        a = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        b = np.array([4.0, 2.0])
        c = np.array([-4.0, -5.0, 0.0, 0.0])
        problem = lp_problem.LpProblem(a, b, c)
        basis = np.array([2, 3])

        rule = pivoting_strategy.SteepestEdgeRule(problem, basis)

        assert (
            rule.pick_entering_index(
                reduced_costs=np.array([-4.0, -5.0]),
                non_basic_vars=np.array([0, 1]),
            )
            == 0
        )

    def test_eta_update_matches_recomputed_norms(self) -> None:
        a = np.array([[1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
        b = np.array([4.0, 2.0])
        c = np.array([-4.0, -5.0, 0.0, 0.0])
        problem = lp_problem.LpProblem(a, b, c)
        basis = np.array([2, 3])
        basis_factorization = linear_algebra.ProductFormFactorization(
            problem.constraint_matrix[:, basis]
        )
        x_basis = basis_factorization.ftran(problem.rhs)

        rule = pivoting_strategy.SteepestEdgeRule(problem, basis)
        entering_variable = rule.pick_entering_index(
            reduced_costs=np.array([-4.0, -5.0]),
            non_basic_vars=np.array([0, 1]),
        )
        basic_direction = basis_factorization.ftran(
            problem.constraint_matrix[:, entering_variable]
        )
        exiting_index = rule.pick_exiting_index(
            basis, x_basis, basic_direction, basis_factorization
        )

        basis[exiting_index] = entering_variable
        expected_non_basic_vars = np.array([2, 1])
        expected_inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])
        expected_basic_directions = (
            expected_inv_basis_matrix
            @ problem.constraint_matrix[:, expected_non_basic_vars]
        )
        expected_norms = 1.0 + np.sum(
            expected_basic_directions * expected_basic_directions, axis=0
        )

        assert np.array_equal(rule.non_basic_vars, expected_non_basic_vars)
        assert rule.norm_eta_squared == pytest.approx(expected_norms)

    def test_solve_with_steepest_edge_and_initial_basis(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        solve_result = primal_simplex.PrimalSimplex(
            pivot_strategy=pivoting_strategy.SteepestEdgeRule()
        ).solve(example_problem_35, initial_basis=np.array([3, 4, 5]))

        assert sorted(solve_result.basis) == [0, 1, 2]
        assert solve_result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert solve_result.objective_value == pytest.approx(-136)

    def test_solve_with_steepest_edge_without_initial_basis(
        self, example_problem_131: tuple[lp_problem.LpProblem, ArrayI]
    ) -> None:
        solve_result = primal_simplex.PrimalSimplex(
            pivot_strategy=pivoting_strategy.SteepestEdgeRule()
        ).solve(example_problem_131[0])

        assert solve_result.objective_value == pytest.approx(-52 / 3)


class TestDualPivoting:
    @pytest.mark.parametrize(
        "piv_strat, expected",
        [
            (pivoting_strategy.DualBlandsRule, 5),
            (pivoting_strategy.DualDantzigsRule, 6),
        ],
    )
    def test_exiting_selection(
        self, piv_strat: type[pivoting_strategy.DualPivotingStrategy], expected: int
    ) -> None:
        primal_vars = np.array([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
        basic_vars = np.array([0, 4, 5, 6, 3, 1, 2])

        result = piv_strat().pick_exiting_index(
            primal_vars,
            basic_vars,
            linear_algebra.ProductFormFactorization(np.eye(len(basic_vars))),
        )
        assert result == expected

    def test_steepest_edge_exiting_selection_uses_scaled_primal_vars(self) -> None:
        primal_vars = np.array([-3.0, -4.0])
        basic_vars = np.array([10, 11])
        basis_matrix = np.array([[1.0, 0.0], [0.0, 0.1]])
        basis_factorization = linear_algebra.ProductFormFactorization(basis_matrix)

        result = pivoting_strategy.DualSteepestEdgeRule().pick_exiting_index(
            primal_vars, basic_vars, basis_factorization
        )

        assert result == 0

    @pytest.mark.parametrize(
        "piv_strat",
        [
            pivoting_strategy.DualBlandsRule,
            pivoting_strategy.DualDantzigsRule,
            pivoting_strategy.DualSteepestEdgeRule,
        ],
    )
    def test_entering_selection(
        self, piv_strat: type[pivoting_strategy.DualPivotingStrategy]
    ) -> None:
        s = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )
        d = np.array(
            [1.0, -4.0, 1.0, 8.0, 1.0, 12.0],
        )

        non_basic_vars = np.array([10, 2, 4, 20, 7, 6])
        exiting_index = piv_strat().pick_entering_index(non_basic_vars, s, d)
        exiting_variable = non_basic_vars[exiting_index]

        assert exiting_index == 5
        assert exiting_variable == 6


class TestInverseComputation:
    def test_update_inverse(self) -> None:
        a = np.array(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        )
        basis = np.array([1, 2], dtype=int)
        basis_matrix = a[:, basis]
        assert basis_matrix == pytest.approx(
            np.array(
                [
                    [2.0, 3.0],
                    [3.0, 2.0],
                ]
            )
        )
        b_inv = np.linalg.inv(basis_matrix)

        # Switch out variable 2 for variable 3
        entering_variable = 3
        exiting_index = 1
        assert basis[exiting_index] == 2
        basis[exiting_index] = entering_variable

        b_inv = linear_algebra.update_inverse(
            a, b_inv, entering_variable, exiting_index
        )
        b_inv_expected = np.linalg.inv(a[:, basis])

        assert b_inv == pytest.approx(b_inv_expected)

    def test_update_speed(self) -> None:
        rng = np.random.default_rng(1337)
        random_column = rng.uniform(low=0, high=1, size=(500, 1))

        constraint_matrix = np.concatenate((np.eye(500), random_column), axis=1)
        basis = list(range(500))
        entering_variable = 500
        exiting_index = 499
        new_basis = [*basis[:-1], 500]

        inv_basis_matrix: ArrayF = np.eye(500)

        start_time = time.perf_counter()
        inv_basis_matrix = linear_algebra.update_inverse(
            constraint_matrix,
            inv_basis_matrix,
            entering_variable,
            exiting_index,
        )
        end_time = time.perf_counter()

        time_in_ms = (end_time - start_time) * 1000

        assert inv_basis_matrix @ constraint_matrix[:, new_basis] == pytest.approx(
            np.eye(500)
        )

        time_limit_ms = 10
        assert time_in_ms < time_limit_ms, (
            f"Should take less than {time_limit_ms} ms, took {time_in_ms}"
        )


@pytest.fixture()
def example_problem_35() -> lp_problem.LpProblem:
    """Example 3.5 in "Introduction to Linear Programming", page 101."""
    a = np.array(
        [
            [1, 2, 2, 1, 0, 0],
            [2, 1, 2, 0, 1, 0],
            [2, 2, 1, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    b = np.array([20, 20, 20], dtype=np.float64).T

    c = np.array([-10, -12, -12, 0, 0, 0], dtype=np.float64)

    return lp_problem.LpProblem(a, b, c)


@pytest.fixture()
def example_problem_131() -> tuple[lp_problem.LpProblem, ArrayI]:
    """Example 13.1 in "Introduction to Linear Programming", page 371."""
    a = np.array(
        [
            [1.0, 1.0, 1.0, 0.0],
            [2.0, 0.5, 0.0, 1.0],
        ],
    )
    b = np.array([5.0, 8.0])
    c = np.array([-4.0, -2.0, 0.0, 0.0])

    initial_basis = np.array([2, 3])

    return lp_problem.LpProblem(a, b, c), initial_basis


@pytest.fixture()
def problem_with_auxvars_left_in_basis() -> lp_problem.LpProblem:
    a = np.array(
        [  # These three at the end are aux
            [1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    b = np.array([0.0, 0.0, 0.0])
    c = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    return lp_problem.LpProblem(a, b, c)


class TestFindingInitialBasis:
    def test_pivot_out_auxiliary(
        self, problem_with_auxvars_left_in_basis: lp_problem.LpProblem
    ) -> None:
        basis = primal_simplex.purge_aux_vars(
            problem_with_auxvars_left_in_basis, np.array([3, 4, 5]), 3
        )
        assert list(basis) == unordered([0, 1, 2])


class TestSolver:
    def test_problem_with_start_solution(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        # Starting basis.
        basis = np.array([3, 4, 5])

        inv_basis_matrix = np.linalg.inv(example_problem_35.constraint_matrix[:, basis])
        x = inv_basis_matrix @ example_problem_35.rhs

        # Assert feasibility of starting point.
        assert example_problem_35.constraint_matrix[:, basis] @ x == pytest.approx(
            example_problem_35.rhs
        )
        assert (x >= 0).all()

        solve_result = primal_simplex.PrimalSimplex().solve(
            example_problem_35, initial_basis=basis
        )

        assert sorted(solve_result.basis) == [0, 1, 2]
        assert solve_result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert solve_result.objective_value == pytest.approx(-136)

    def test_find_initial_basis(self, example_problem_35: lp_problem.LpProblem) -> None:
        assert sorted(
            primal_simplex.PrimalSimplex().find_initial_basis(example_problem_35)
        ) == [
            0,
            1,
            2,
        ]

    def test_problem_without_start_solution(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        solve_result = primal_simplex.PrimalSimplex().solve(example_problem_35)

        assert sorted(solve_result.basis) == [0, 1, 2]
        assert solve_result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert solve_result.objective_value == pytest.approx(-136)

    def test_chvatals_example_for_cycling(self) -> None:
        # See for ref: https://www.matem.unam.mx/~omar/math340/degenerate.html or
        # https://sites.math.washington.edu/~vinzant/teaching/407/Chvatal.pdf, page 31.
        # This is an example of an LP which cycles under some pivot rules.
        a = np.array(
            [
                [-0.5, 5.5, 2.5, -9.0, -1.0, 0.0, 0.0],
                [-0.5, 1.5, 0.5, -1.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        b = np.array([0, 0, 1], dtype=float).T
        c = np.array([-10, 57, 9, 24, 0, 0, 0], dtype=float)
        initial_basis = np.array([4, 5, 6])

        with pytest.raises(
            SimplexCyclingError, match="Basis cycle of length 6 detected"
        ):
            primal_simplex.PrimalSimplex(pivoting_strategy.DantzigsRule()).solve(
                lp_problem.LpProblem(a, b, c), initial_basis=initial_basis
            )

        assert primal_simplex.PrimalSimplex(pivoting_strategy.BlandsRule()).solve(
            lp_problem.LpProblem(a, b, c), initial_basis=initial_basis
        ).objective_value == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        "solver",
        [primal_simplex.PrimalSimplex, dual_simplex.DualSimplex],
    )
    def test_infeasible_problem(
        self, solver: type[primal_simplex.PrimalSimplex | dual_simplex.DualSimplex]
    ) -> None:
        # min -x1 s.t. x1 <= 1, x1 >= 2
        # Standard form: x1 + s1 = 1, x1 - s2 = 2
        a = np.array([[1, 1, 0], [1, 0, -1]], dtype=float)
        b = np.array([1, 2], dtype=float)
        c = np.array([-1, 0, 0], dtype=float)

        test_problem = lp_problem.LpProblem(a, b, c)
        with pytest.raises(InfeasibleLpError):
            solver().solve(test_problem)

    @pytest.mark.parametrize(
        "solver",
        [primal_simplex.PrimalSimplex, dual_simplex.DualSimplex],
    )
    def test_unbounded_problem(
        self, solver: type[primal_simplex.PrimalSimplex | dual_simplex.DualSimplex]
    ) -> None:
        # min -2*x1 - x2 s.t. x1 - x2 <= 10, 2*x1 <= 40
        # Standard form: x1 - x2 + s1 = 10, 2*x1 + s2 = 40
        a = np.array([[1, -1, 1, 0], [2, 0, 0, 1]], dtype=float)
        b = np.array([10, 40], dtype=float)
        c = np.array([-2, -1, 0, 0], dtype=float)

        test_problem = lp_problem.LpProblem(a, b, c)
        with pytest.raises(UnboundedLpError):
            solver().solve(test_problem)

    def test_example_13_1_from_book(
        self, example_problem_131: tuple[lp_problem.LpProblem, ArrayI]
    ) -> None:
        # Lukas spotted typo in book, test to verify!
        simplex_solver = primal_simplex.PrimalSimplex()
        solve_result = simplex_solver.solve(
            problem=example_problem_131[0], initial_basis=example_problem_131[1]
        )
        assert solve_result.objective_value == pytest.approx(-52 / 3)
        assert simplex_solver.history.basis_history == [(2, 3), (0, 2), (0, 1)]
        assert simplex_solver.history.objective_history == pytest.approx(
            [0.0, -16.0, -52 / 3]
        )

    @pytest.mark.parametrize(
        "strategy",
        [pivoting_strategy.BlandsRule, pivoting_strategy.DantzigsRule],
    )
    def test_example_13_1_from_book_without_initial_basis(
        self,
        example_problem_131: tuple[lp_problem.LpProblem, ArrayI],
        strategy: type[pivoting_strategy.PrimalPivotingStrategy],
    ) -> None:
        solve_result = primal_simplex.PrimalSimplex(pivot_strategy=strategy()).solve(
            example_problem_131[0]
        )
        assert solve_result.objective_value == pytest.approx(-52 / 3)


class TestDualBlandsRule:
    def test_exiting_index(self) -> None:

        # Out of all negative primal vars, -10.4 has the smallest basic index which is 3
        primal_vars = np.array([1.2, 3.4, -1.02, 3.2, -10.4, -11.46, -0.4, 12.2])
        basis = np.array([20, 1, 6, 100, 3, 17, 9, 12])
        dual_blands_rule = pivoting_strategy.DualBlandsRule()

        assert (
            basis[
                dual_blands_rule.pick_exiting_index(
                    primal_vars,
                    basis,
                    linear_algebra.ProductFormFactorization(np.eye(len(basis))),
                )
            ]
            == 3
        )

    def test_entering_index(self) -> None:
        s = np.array([1.2, 3.4, 0.4, 10.2])
        non_basic_vars = np.array([3, 2, 1, 10])
        change_in_s_due_to_entering_index = np.array([-0.01, 1.2, 0.001, -1.0])

        dual_blands_rule = pivoting_strategy.DualBlandsRule()
        assert (
            dual_blands_rule.pick_entering_index(
                non_basic_vars, s, change_in_s_due_to_entering_index
            )
            == 1
        )


class TestDualSolve:
    def test_problem_without_start_solution_with_steepest_edge(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        solve_result = dual_simplex.DualSimplex(
            pivot_strategy=pivoting_strategy.DualSteepestEdgeRule()
        ).solve(example_problem_35)

        assert sorted(solve_result.basis) == [0, 1, 2]
        assert solve_result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert solve_result.objective_value == pytest.approx(-136)

    def test_problem_without_start_solution(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        # The optimal basis for problem 35 is 0, 1, 2.
        # This corresponds to solution 4, 4, 4, 0, 0, 0
        # By adding a constraint that x_0 + 3*x_1 + 3*x_2 < 4
        # we have a dual feasible starting point which is primal infeasible.
        # The new optimum should then be 4, 0, 0, 0, 0, 0
        # We can find an optimal basis for the dual by adding a slack
        # for this constraint and having that + the previous optimal
        # basis in the starting basis for the dual simplex

        # So we need to add a slack variable with 0 in the objective
        # And a row with 1, 3, 3, 0, 0, 0, 1
        # Also, we need to pad the other constraint rows with zeros for the new slack var.
        a = example_problem_35.constraint_matrix
        b = example_problem_35.rhs
        c = example_problem_35.objective
        a = np.pad(a, ((0, 1), (0, 1)))

        a[-1, :] = np.array([1, 3, 3, 0, 0, 0, 1])
        c = np.pad(c, (0, 1))
        b = np.pad(b, (0, 1), constant_values=4.0)
        amended_problem = lp_problem.LpProblem(constraint_matrix=a, rhs=b, objective=c)

        solution_by_primal = primal_simplex.PrimalSimplex(
            pivot_strategy=pivoting_strategy.BlandsRule()
        ).solve(amended_problem)
        assert solution_by_primal.objective_value == pytest.approx(-40)

        dual_simplex_solver = dual_simplex.DualSimplex(
            pivot_strategy=pivoting_strategy.DualBlandsRule()
        )

        initial_basis = np.array([0, 1, 2, 6])
        solution_by_dual = dual_simplex_solver.solve(
            amended_problem, 100, initial_basis
        )
        assert solution_by_dual.objective_value == pytest.approx(-40)
        assert np.allclose(solution_by_dual.solution, solution_by_primal.solution)


class TestDualPhaseOne:
    def test_artificial_problem_setup(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        # Create a problem that is not dual-feasible with the QR-chosen basis [0, 1, 2].
        # By negating the objective, we get negative reduced costs for the slack variables.
        # The problem failed previously since the basis obtained in _setup_artificial_problem
        # was already dual feasible so nothing happened.
        non_dual_feasible_problem = lp_problem.LpProblem(
            constraint_matrix=example_problem_35.constraint_matrix,
            rhs=example_problem_35.rhs,
            objective=-example_problem_35.objective,
        )

        solver = dual_simplex.DualSimplex()
        aug_prob, basis = solver._setup_artificial_problem(non_dual_feasible_problem)

        assert non_dual_feasible_problem.constraint_matrix.shape == (3, 6)
        assert aug_prob.constraint_matrix.shape == (4, 7)
        assert len(basis) == 4

        inv_basis_matrix = np.linalg.inv(aug_prob.constraint_matrix[:, basis])
        lam = inv_basis_matrix.T @ aug_prob.objective[basis]
        non_basic_vars = np.array([v for v in range(7) if v not in basis])
        s_non_basic = aug_prob.objective[non_basic_vars] - (
            aug_prob.constraint_matrix[:, non_basic_vars].T @ lam
        )
        assert np.all(s_non_basic >= -1e-6)

    def test_dual_solve_without_initial_basis(
        self, example_problem_35: lp_problem.LpProblem
    ) -> None:
        solver = dual_simplex.DualSimplex()
        result = solver.solve(example_problem_35)

        assert result.objective_value == pytest.approx(-136)
        assert result.solution == pytest.approx(np.array([4, 4, 4, 0, 0, 0]))
        assert sorted(result.basis) == [0, 1, 2]

    def test_dual_solve_unbounded(self) -> None:
        a = np.array([[1.0, -1.0, 1.0]], dtype=float)
        b = np.array([10.0], dtype=float)
        c = np.array([-1.0, -1.0, 0.0], dtype=float)

        problem = lp_problem.LpProblem(a, b, c)
        solver = dual_simplex.DualSimplex()

        with pytest.raises(UnboundedLpError):
            solver.solve(problem)


class TestProductFormFactorization:
    """Exercise the PF backend against independently constructed basis matrices."""

    @staticmethod
    def _example_constraint_matrix_and_factorization() -> tuple[
        np.ndarray, np.ndarray, linear_algebra.ProductFormFactorization
    ]:
        l_factor = np.array(
            [
                [1.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [-1.0, 3.0, 1.0],
            ]
        )
        u_factor = np.array(
            [
                [2.0, -1.0, 4.0],
                [0.0, 3.0, 5.0],
                [0.0, 0.0, -2.0],
            ]
        )
        basis = np.array([0, 2, 4])

        # Left zeros in the basis elements so they can be filled in
        constraint_matrix = np.array(
            [
                [0.0, 7.0, 0.0, 6.0, 0.0, 3.0],
                [0.0, 8.0, 0.0, 5.0, 0.0, 11.0],
                [0.0, 2.0, 0.0, 4.0, 0.0, -1.0],
            ]
        )
        constraint_matrix[:, basis] = l_factor @ u_factor
        factorization = linear_algebra.ProductFormFactorization.from_factors(
            l_factor, u_factor
        )

        return constraint_matrix, basis, factorization

    def test_product_form_factorization_solves_initial_basis(self) -> None:
        (
            constraint_matrix,
            basis,
            factorization,
        ) = self._example_constraint_matrix_and_factorization()
        basis_matrix = constraint_matrix[:, basis]
        rhs = np.array([1.0, -2.0, 3.0])

        assert sparse.issparse(factorization.l_factor)
        assert sparse.issparse(factorization.u_factor)
        assert factorization.to_matrix() == pytest.approx(basis_matrix)
        assert factorization.ftran(rhs) == pytest.approx(
            np.linalg.solve(basis_matrix, rhs)
        )
        assert factorization.btran(rhs) == pytest.approx(
            np.linalg.solve(basis_matrix.T, rhs)
        )

    def test_product_form_factorization_solves_after_one_update(self) -> None:
        (
            constraint_matrix,
            basis,
            factorization,
        ) = self._example_constraint_matrix_and_factorization()
        entering_variable = 5
        exiting_index = 1

        factorization.update(
            constraint_matrix[:, entering_variable],
            exiting_index,
        )
        basis[exiting_index] = entering_variable
        updated_basis_matrix = constraint_matrix[:, basis]
        rhs = np.array([1.0, -2.0, 3.0])

        assert factorization.to_matrix() == pytest.approx(updated_basis_matrix)
        assert factorization.ftran(rhs) == pytest.approx(
            np.linalg.solve(updated_basis_matrix, rhs)
        )
        assert factorization.btran(rhs) == pytest.approx(
            np.linalg.solve(updated_basis_matrix.T, rhs)
        )

    def test_product_form_factorization_solves_after_multiple_updates(self) -> None:
        (
            constraint_matrix,
            basis,
            factorization,
        ) = self._example_constraint_matrix_and_factorization()

        factorization.update(constraint_matrix[:, 5], exiting_index=1)
        basis[1] = 5

        # The second update must transform its entering column through the existing
        # basis eta stack before appending the next eta column. Both simplex
        # solvers rely on this after the first pivot.
        factorization.update(constraint_matrix[:, 1], exiting_index=2)
        basis[2] = 1

        updated_basis_matrix = constraint_matrix[:, basis]
        rhs = np.array([1.0, -2.0, 3.0])

        assert factorization.to_matrix() == pytest.approx(updated_basis_matrix)
        assert factorization.ftran(rhs) == pytest.approx(
            np.linalg.solve(updated_basis_matrix, rhs)
        )
        assert factorization.btran(rhs) == pytest.approx(
            np.linalg.solve(updated_basis_matrix.T, rhs)
        )
