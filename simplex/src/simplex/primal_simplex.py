import logging
import time

import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF, ArrayI

from simplex import linear_algebra, pivoting_strategy
from simplex_util import (
    INVERSE_RECOMPUTE_INTERVAL,
    NON_NEGATIVITY_TOLERANCE,
    OPTIMALITY_TOL,
    InfeasibleLpError,
    IterationLimitError,
    SolveFailedError,
    SolveHistory,
    SolveResult,
    UnboundedLpError,
    get_non_basic_vars,
)

logger = logging.getLogger(__name__)
LOG_FIRST_ITERATIONS = 10
LOG_INTERVAL = 100


def is_linearly_independent(
    problem: lp_problem.LpProblem,
    basis_factorization: linear_algebra.ProductFormFactorization,
    entering_var: int,
    exiting_index: int,
) -> bool:
    basic_direction = basis_factorization.ftran(
        problem.constraint_matrix[:, entering_var]
    )
    pivot_column_entry = basic_direction[exiting_index]
    tolerance = 1e-9
    return bool(abs(pivot_column_entry) > tolerance)


def purge_aux_vars(
    problem: lp_problem.LpProblem,
    basis: jaxtyping.Int[ArrayI, " m"],
    num_variables: int,
) -> jaxtyping.Int[ArrayI, " m"]:
    aux_vars_still_in_basis = [b for b in basis if not b < num_variables]
    if aux_vars_still_in_basis:
        basis_factorization = linear_algebra.ProductFormFactorization(
            problem.constraint_matrix[:, basis]
        )

        while aux_vars_still_in_basis:
            exiting_variable = aux_vars_still_in_basis.pop()
            exiting_index = next(
                i for i, x in enumerate(basis) if x == exiting_variable
            )

            # Generator of vars not in basis:
            non_basic_vars = (b for b in range(num_variables) if b not in basis)
            while not is_linearly_independent(
                problem,
                basis_factorization,
                (entering_var := next(non_basic_vars)),
                exiting_index=exiting_index,
            ):
                pass

            basis_factorization.update(
                problem.constraint_matrix[:, entering_var],
                exiting_index=exiting_index,
            )
            basis[exiting_index] = entering_var
    return basis


class PrimalSimplex:
    pivoting_strategy_: pivoting_strategy.PrimalPivotingStrategy
    solve_history_: SolveHistory

    def __init__(
        self,
        pivot_strategy: pivoting_strategy.PrimalPivotingStrategy | None = None,
    ) -> None:
        if pivot_strategy is not None:
            self.pivoting_strategy_ = pivot_strategy
        else:
            self.pivoting_strategy_ = pivoting_strategy.BlandsRule()

        self.solve_history_ = SolveHistory()

    @property
    def history(self) -> SolveHistory:
        return self.solve_history_

    def find_initial_basis(
        self, problem: lp_problem.LpProblem, max_iterations: int = 100
    ) -> jaxtyping.Int[ArrayI, " {problem.constraint_matrix.shape[0]}"]:
        """
        Finds a basic feasible solution by solving an auxiliary LP. Throws a SolveFailedError if it fails.

        Need to use the symbolic expression " {problem.constraint_matrix.shape[0]}" instead of " m" in the
        jaxtyping annotation. The constraint_matrix field in the LpProblem dataclass is annotated with "m n",
        but the runtime type checker can't see the annotations inside the dataclass, see
        https://github.com/patrick-kidger/jaxtyping/issues/342.
        """

        logger.info("Solving auxiliary phase one LP to find a starting basis...")

        e_matrix = np.diag(np.array([1.0 if b >= 0 else -1.0 for b in problem.rhs]))
        phase_one_problem = lp_problem.LpProblem(
            constraint_matrix=np.concatenate(
                [problem.constraint_matrix, e_matrix], axis=1
            ),
            rhs=np.array(problem.rhs),
            objective=np.concatenate(
                [np.zeros(problem.num_variables), np.ones(len(problem.rhs))]
            ),
        )
        # Use the smallest subscript rule to hopefully basis containing the original variables
        phase_one_solver = PrimalSimplex(pivot_strategy=self.pivoting_strategy_)
        phase_one_result = phase_one_solver.solve(
            phase_one_problem,
            initial_basis=np.array(
                range(problem.num_variables, phase_one_problem.num_variables)
            ),
            max_iterations=max_iterations,
        )
        if phase_one_result.objective_value > OPTIMALITY_TOL:
            raise SolveFailedError(
                f"Phase one objective value {phase_one_result.objective_value} is positive: Original problem is infeasible"
            )

        max_print_size = 10
        logger.info(
            f"Found starting basis {phase_one_result.basis if len(phase_one_result.basis) < max_print_size else ''}"
        )

        # Pivot out any auxiliary variables that may be in the basis
        basis = purge_aux_vars(
            phase_one_problem, phase_one_result.basis, problem.num_variables
        )

        aux_vars_still_in_basis = [b for b in basis if not b < problem.num_variables]
        if aux_vars_still_in_basis:
            raise SolveFailedError("Auxiliary variables still present in the basis.")

        return basis

    def _compute_reduced_costs(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        non_basic_vars: jaxtyping.Int[ArrayI, " num_nonbasic"],
        basis_factorization: linear_algebra.ProductFormFactorization,
    ) -> jaxtyping.Float[ArrayF, " num_nonbasic"]:
        lambda_t = basis_factorization.btran(problem.objective[basis])
        all_dual_constraint_values = np.asarray(
            problem.sparse_constraint_matrix.T @ lambda_t
        )
        reduced_costs: ArrayF = (
            problem.objective[non_basic_vars]
            - all_dual_constraint_values[non_basic_vars]
        )
        return reduced_costs

    def _finalize_result(
        self,
        problem: lp_problem.LpProblem,
        basis: jaxtyping.Int[ArrayI, " m"],
        x_basis: jaxtyping.Float[ArrayF, " m"],
    ) -> SolveResult:
        solution = np.zeros(problem.num_variables)
        solution[basis] = x_basis

        return SolveResult(
            basis=basis,
            solution=solution,
            objective_value=self.solve_history_.objective_history[-1],
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        max_iterations: int = 100,
        initial_basis: jaxtyping.Int[ArrayI, " m"] | None = None,
    ) -> SolveResult:
        self.solve_history_ = SolveHistory()

        if initial_basis is not None:
            basis = np.array(initial_basis)
        else:
            try:
                basis = self.find_initial_basis(problem, max_iterations=max_iterations)
            except SolveFailedError as e:
                raise InfeasibleLpError(
                    f"Failed to find an initial simplex basis: {e}"
                ) from e
        non_basic_vars = get_non_basic_vars(problem.num_variables, basis)

        basis_factorization = linear_algebra.ProductFormFactorization(
            problem.constraint_matrix[:, basis]
        )
        self.pivoting_strategy_.initialize(problem, basis, basis_factorization)
        x_basis = basis_factorization.ftran(problem.rhs)

        logger.info("Starting simplex algorithm...")
        self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
        logger.info(
            f"Initial objective value {self.solve_history_.objective_history[-1]}"
        )

        logger.info("Iter     Objective      Primal Inf.    Dual Inf.    Time")
        start = time.time()
        for iteration in range(1, max_iterations):
            reduced_costs = self._compute_reduced_costs(
                problem, basis, non_basic_vars, basis_factorization
            )
            if np.all(reduced_costs >= -NON_NEGATIVITY_TOLERANCE):
                logger.info(
                    f"Simplex algorithm found optimal objective {self.solve_history_.objective_history[-1]} after {iteration - 1} iterations."
                )
                return self._finalize_result(problem, basis, x_basis)

            entering_index = self.pivoting_strategy_.pick_entering_index(
                reduced_costs, non_basic_vars
            )
            entering_variable = non_basic_vars[entering_index]
            d = basis_factorization.ftran(
                problem.constraint_matrix[:, entering_variable]
            )
            if np.all(d <= pivoting_strategy.PIVOTING_TOLERANCE):
                raise UnboundedLpError

            basic_exiting_index = self.pivoting_strategy_.pick_exiting_index(
                basis, x_basis, d, basis_factorization
            )
            x_entering = x_basis[basic_exiting_index] / d[basic_exiting_index]

            non_basic_vars[entering_index] = basis[basic_exiting_index]
            basis[basic_exiting_index] = entering_variable

            if iteration % INVERSE_RECOMPUTE_INTERVAL == 0:
                non_basic_vars = get_non_basic_vars(problem.num_variables, basis)
                basis_factorization = linear_algebra.ProductFormFactorization(
                    problem.constraint_matrix[:, basis]
                )
                self.pivoting_strategy_.initialize(problem, basis, basis_factorization)
            else:
                self.pivoting_strategy_.sync_non_basic_variables(
                    non_basic_vars
                )  # TODO(martin): Remove
                basis_factorization.update(
                    problem.constraint_matrix[:, entering_variable],
                    basic_exiting_index,
                )

            x_basis -= x_entering * d
            x_basis[basic_exiting_index] = x_entering

            self.solve_history_.update(basis, float(problem.objective[basis] @ x_basis))
            if (iteration < LOG_FIRST_ITERATIONS) or (iteration % LOG_INTERVAL == 0):
                lambda_t = basis_factorization.btran(problem.objective[basis])
                all_dual_constraint_values = np.asarray(
                    problem.sparse_constraint_matrix.T @ lambda_t
                )
                logger.info(
                    f"{iteration:4d}    {problem.objective[basis].T @ x_basis:10.3e}     "
                    f"{np.sum(np.abs(problem.constraint_matrix[:, basis] @ x_basis - problem.rhs)):10.3e}     {max(0.0, np.sum(all_dual_constraint_values - problem.objective)):10.3e}"
                    f"    {time.time() - start:.4}s"
                )

        logger.info(
            f"Simplex algorithm terminated due to {max_iterations} iteration limit"
        )
        raise IterationLimitError
