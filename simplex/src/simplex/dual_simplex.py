import logging
import time

import jaxtyping
import numpy as np
import scipy.linalg
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI

import simplex_util
from simplex import linear_algebra, pivoting_strategy
from simplex_util import (
    INVERSE_RECOMPUTE_INTERVAL,
    NON_NEGATIVITY_TOLERANCE,
    SolveHistory,
    get_non_basic_vars,
)

logger = logging.getLogger(__name__)
LOG_FIRST_ITERATIONS = 10
LOG_INTERVAL = 100


# Solve with primal simplex
# N^T \lambda + s = c
# Add q
# N^T \lambda + s + Eq = c
# with E_ii = -1 if c<0 else 1
# Then solve min q subject to that constraint
# with initial basis all of the q's and q = |c|.


class DualSimplex:
    pivoting_strategy_: pivoting_strategy.DualPivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.DualPivotingStrategy | None = None,
    ) -> None:
        if pivot_strategy is not None:
            self.pivoting_strategy_ = pivot_strategy
        else:
            self.pivoting_strategy_ = pivoting_strategy.DualBlandsRule()

        self.solve_history_ = SolveHistory()

    @property
    def history(self) -> SolveHistory:
        return self.solve_history_

    def _setup_artificial_problem(
        self, problem: lp_problem.LpProblem, big_m: float = 1e6
    ) -> tuple[
        lp_problem.LpProblem,
        jaxtyping.Int[ArrayI, " m"],
    ]:
        """
        Sets up the problem for a Phase 1 Dual Simplex by adding a single artificial constraint.

        This method handles problems where an initial basis is not dual-feasible by adding one
        artificial constraint and one artificial variable. This allows us to perform a "magic pivot"
        that immediately yields a dual-feasible basis.

        Mathematical Formulation:
        1. Start with an arbitrary basis B and compute its reduced costs s.
        2. If some s_k < 0, the basis is not dual-feasible.
        3. Augment the primal problem with a new constraint:
           \\sum_{j \\notin B} x_j + x_{art} = M
           where x_{art} is an artificial slack variable with cost 0, and M is a large scalar.
        4. The initial basis for this augmented problem is B U {x_{art}}.
        5. We force x_{art} to exit the basis and x_k (where k = argmin(s)) to enter. This is the "magic pivot".
           Because the artificial row has a coefficient of 1 for all non-basic variables, pivoting
           on it subtracts s_k from all reduced costs: s_j' = s_j - s_k.
           Since s_k is the minimum reduced cost, s_j' >= 0 for all j, making the new basis
           B U {x_k} strictly dual-feasible!

        For reference, this is commonly known as the "Single Artificial Constraint" method
        for the dual simplex method.
        """
        m, n = problem.constraint_matrix.shape

        # Use rank-revealing QR factorization to find m linearly independent columns.
        # The permutation array `p` pushes independent columns to the front.
        _, _, p = scipy.linalg.qr(problem.constraint_matrix, pivoting=True)
        basis = p[:m]

        inv_basis_matrix = np.linalg.inv(problem.constraint_matrix[:, basis])
        non_basic_vars = get_non_basic_vars(problem.num_variables, basis)

        lam = inv_basis_matrix.T @ problem.objective[basis]
        all_dual_constraint_values = np.asarray(
            problem.sparse_constraint_matrix.T @ lam
        )
        s_non_basic = (
            problem.objective[non_basic_vars]
            - all_dual_constraint_values[non_basic_vars]
        )

        if np.all(s_non_basic >= -NON_NEGATIVITY_TOLERANCE):
            return problem, basis

        entering_index_in_non_basic = int(np.argmin(s_non_basic))
        entering_variable = non_basic_vars[entering_index_in_non_basic]

        # Augment the problem
        new_row = np.zeros(n + 1)
        new_row[non_basic_vars] = 1.0
        new_row[-1] = 1.0  # x_art

        new_a = np.zeros((m + 1, n + 1))
        new_a[:m, :n] = problem.constraint_matrix
        new_a[m, :] = new_row

        new_b = np.append(problem.rhs, big_m)
        new_c = np.append(problem.objective, 0.0)

        augmented_problem = lp_problem.LpProblem(
            constraint_matrix=new_a, rhs=new_b, objective=new_c
        )

        # The new basis is the original basis B U {entering_variable}
        new_basis = np.append(basis, entering_variable)
        return augmented_problem, new_basis

    def _finalize_result(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
        is_augmented: bool = False,
        original_num_variables: int = 0,
    ) -> simplex_util.SolveResult:
        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        final_basis = basis

        if is_augmented:
            x_art_val = solution[-1]
            if x_art_val <= NON_NEGATIVITY_TOLERANCE:
                raise simplex_util.UnboundedLpError(
                    "Artificial bound reached, original Primal problem is unbounded (Dual is infeasible)."
                )

            art_index = problem.num_variables - 1
            final_basis = np.array([b for b in basis if b != art_index])
            solution = solution[:original_num_variables]

        return simplex_util.SolveResult(
            basis=final_basis,
            solution=solution,
            objective_value=self.solve_history_.objective_history[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> simplex_util.SolveResult:

        original_num_variables = problem.num_variables
        is_augmented = False

        if initial_basis is None:
            problem, initial_basis = self._setup_artificial_problem(problem)
            is_augmented = problem.num_variables > original_num_variables

        basis = np.array(initial_basis)
        non_basic_vars = get_non_basic_vars(problem.num_variables, basis)
        basis_factorization = linear_algebra.ProductFormFactorization(
            problem.constraint_matrix[:, basis]
        )
        x_basis = basis_factorization.ftran(problem.rhs)
        self.pivoting_strategy_.initialize(problem, basis, basis_factorization)

        logger.info("Starting Dual Simplex algorithm...")
        self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
        logger.info(
            f"Initial objective value {self.solve_history_.objective_history[-1]}"
        )
        logger.info("Iter     Objective      Primal Inf.    Dual Inf.    Time")
        start = time.time()

        for iteration in range(1, max_iterations):
            if np.all(x_basis >= -NON_NEGATIVITY_TOLERANCE):
                logger.info(
                    f"Simplex algorithm found optimal objective {self.solve_history_.objective_history[-1]} after {iteration - 1} iterations."
                )
                return self._finalize_result(
                    problem, basis, x_basis, is_augmented, original_num_variables
                )

            exiting_index = self.pivoting_strategy_.pick_exiting_index(
                x_basis, basis, basis_factorization
            )
            lam = basis_factorization.btran(problem.objective[basis])
            all_dual_constraint_values = np.asarray(
                problem.sparse_constraint_matrix.T @ lam
            )
            s_non_basic = (
                problem.objective[non_basic_vars]
                - all_dual_constraint_values[non_basic_vars]
            )
            unit_exiting = np.zeros(len(basis))
            unit_exiting[exiting_index] = 1.0
            v = -basis_factorization.btran(unit_exiting)
            all_direction_values = np.asarray(problem.sparse_constraint_matrix.T @ v)
            non_basic_direction = all_direction_values[non_basic_vars]
            if np.max(non_basic_direction) <= pivoting_strategy.PIVOTING_TOLERANCE:
                raise simplex_util.InfeasibleLpError(
                    "Dual simplex detected primal infeasibility."
                )

            entering_index = self.pivoting_strategy_.pick_entering_index(
                non_basic_vars, s_non_basic, non_basic_direction
            )
            entering_variable = non_basic_vars[entering_index]
            basic_direction = basis_factorization.ftran(
                problem.constraint_matrix[:, entering_variable]
            )
            gamma = x_basis[exiting_index] / basic_direction[exiting_index]
            x_basis -= gamma * basic_direction
            x_basis[exiting_index] = gamma

            non_basic_vars[entering_index] = basis[exiting_index]
            basis[exiting_index] = entering_variable
            if iteration % INVERSE_RECOMPUTE_INTERVAL == 0:
                basis_factorization = linear_algebra.ProductFormFactorization(
                    problem.constraint_matrix[:, basis]
                )
                self.pivoting_strategy_.initialize(problem, basis, basis_factorization)
            else:
                basis_factorization.update(
                    problem.constraint_matrix[:, entering_variable],
                    exiting_index,
                )

            self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
            if (iteration < LOG_FIRST_ITERATIONS) or (iteration % LOG_INTERVAL == 0):
                primal_inf = (
                    np.sum(
                        np.abs(
                            problem.constraint_matrix[:, basis] @ x_basis - problem.rhs
                        )
                    )
                    - np.minimum(x_basis, 0.0).sum()
                )
                dual_inf = abs(min(float(np.min(s_non_basic)), 0.0))
                logger.info(
                    f"{iteration:4d}    {problem.objective[basis].T @ x_basis:10.3e}     "
                    f"{primal_inf:10.3e}     {dual_inf:10.3e}"
                    f"    {time.time() - start:.4}s"
                )

        logger.info(
            f"Simplex algorithm terminated due to {max_iterations} iteration limit"
        )
        raise simplex_util.IterationLimitError
