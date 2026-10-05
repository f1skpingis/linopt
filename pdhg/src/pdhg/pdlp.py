import logging
import time

import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF

from pdhg import pdhg_tools

logger = logging.getLogger(__name__)


class PdlpSolver:
    def __init__(self) -> None:
        pass

    @staticmethod
    def log_metrics(
        metrics: pdhg_tools.PdhgOptimalityMetrics, iteration: int, t: float
    ) -> None:
        logger.info(
            f"{iteration:4d}   {metrics.primal_objective:10.3e}  "
            f"{metrics.dual_objective:10.3e}  "
            f"{metrics.primal_residual:10.3e} "
            f"{metrics.dual_residual:10.3e}  "
            f"{metrics.relative_gap:8.3e}    "
            f"{t:5.2}s"
        )

    def solve(
        self,
        problem: lp_problem.LpProblem,
        start_solution: tuple[ArrayF, ArrayF],
        max_iterations: int,
    ) -> tuple[ArrayF, ArrayF]:

        k = pdhg_tools.vstack(
            pdhg_tools.DenseLinearOperator(
                problem.effective_inequality_constraint_matrix
            ),
            pdhg_tools.DenseLinearOperator(problem.constraint_matrix),
        )
        primal_weight = 1e0
        power_iteration_result = pdhg_tools.iterative_spectral_norm(k)
        if power_iteration_result.status != pdhg_tools.PowerIterationStatus.CONVERGED:
            raise RuntimeError("Could not compute the spectral norm of K")
        step_size = 0.9 / power_iteration_result.spectral_norm
        step_size_parameters = pdhg_tools.PdhgStepSizeParameters(
            primal_weight=primal_weight, step_size=step_size
        )

        x, y = start_solution

        q = np.concatenate(
            (
                problem.effective_inequality_rhs,
                problem.rhs,
            )
        )

        q_norm = np.linalg.norm(q)
        c_norm = np.linalg.norm(problem.objective)

        num_inequality_constraints = problem.num_inequality_constraints

        # TODO(martin): Centralize this type of logging format. It is the same for Simplex
        logger.info("Starting PDHG algorithm...")

        logger.info("                Objective              Residual")
        logger.info(
            "Iter       Primal       Dual       Primal      Dual     Gap        Time"
        )
        start = time.time()
        for iteration in range(max_iterations):
            metrics: pdhg_tools.PdhgOptimalityMetrics | None = None
            old_x = x.copy()

            # TODO(martins): Put these operations in their own functions
            # and document the mathematics of the projections in the docstrings
            # to keep code scientific and informative.
            x = np.clip(
                x
                - step_size_parameters.primal_step_size
                * (problem.objective - k.rmatvec(y)),
                problem.effective_lower_bounds,
                problem.effective_upper_bounds,
            )

            y = y + step_size_parameters.dual_step_size * (q - k.matvec(2 * x - old_x))

            y[0:num_inequality_constraints] = np.maximum(
                0.0, y[0:num_inequality_constraints]
            )

            if iteration % pdhg_tools.OPTIMALITY_CHECK_INTERVAL == 0:
                metrics = pdhg_tools.compute_optimality_metrics(problem, k, x, y, q)
                if metrics.is_optimal(
                    c_norm=c_norm,
                    q_norm=q_norm,
                    tolerance=pdhg_tools.ACCURATE_OPTIMALITY_TOLERANCE,
                ):
                    return (x, y)

            if (iteration < pdhg_tools.LOG_FIRST_ITERATIONS) or (
                iteration % pdhg_tools.LOG_INTERVAL == 0
            ):
                if metrics is None:
                    metrics = pdhg_tools.compute_optimality_metrics(problem, k, x, y, q)

                self.log_metrics(metrics, iteration, time.time() - start)

        raise RuntimeError("Failed to solve LP")
