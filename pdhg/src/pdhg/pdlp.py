import logging
import time

import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF

from pdhg import pdhg_tools

logger = logging.getLogger(__name__)


class PdlpSolver:
    """Baseline PDHG using the newer PDLP paper, Section 2, Eq. (3).

    TODO(martins): Add diag preconditioning (see section three of the paper),
    add step size (see page 7 of the paper) and primal weight (Algorithm 3 in the paper)
    """

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

    @staticmethod
    def validate_start_solution(
        problem: lp_problem.BoundedFormLP, x: ArrayF, y: ArrayF
    ) -> None:
        if x.shape != (problem.num_variables,) or y.shape != (
            problem.constraint_matrix.shape[0],
        ):
            raise ValueError("Starting point must have shapes (n,) and (m,).")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("Starting point must be finite.")

    def solve(
        self,
        problem: lp_problem.BoundedFormLP,
        start_solution: tuple[ArrayF, ArrayF],
        max_iterations: int,
    ) -> tuple[ArrayF, ArrayF]:

        if not isinstance(problem, lp_problem.BoundedFormLP):
            raise TypeError(
                "PDHG requires BoundedFormLP; use to_bounded_form() for standard form."
            )
        a = pdhg_tools.DenseLinearOperator(problem.constraint_matrix)
        primal_weight = 1e0
        power_iteration_result = pdhg_tools.iterative_spectral_norm(a)
        if power_iteration_result.status != pdhg_tools.PowerIterationStatus.CONVERGED:
            raise RuntimeError("Could not compute the spectral norm of A")
        step_size = (
            0.9 / power_iteration_result.spectral_norm
            if power_iteration_result.spectral_norm > 0
            else 1.0
        )
        step_size_parameters = pdhg_tools.PdhgStepSizeParameters(
            primal_weight=primal_weight, step_size=step_size
        )

        x, y = (np.array(v, dtype=float, copy=True) for v in start_solution)

        self.validate_start_solution(problem, x, y)

        bounds_norm = pdhg_tools.finite_row_bounds_norm(problem)
        c_norm = float(np.linalg.norm(problem.objective))

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

            x = pdhg_tools.primal_update(
                problem, a, x, y, step_size_parameters.primal_step_size
            )
            y = pdhg_tools.dual_update(
                problem, a, 2 * x - old_x, y, step_size_parameters.dual_step_size
            )

            if iteration % pdhg_tools.OPTIMALITY_CHECK_INTERVAL == 0:
                metrics = pdhg_tools.compute_optimality_metrics(problem, a, x, y)
                if metrics.is_optimal(
                    c_norm=c_norm,
                    bounds_norm=bounds_norm,
                    tolerance=pdhg_tools.ACCURATE_OPTIMALITY_TOLERANCE,
                ):
                    return (x, y)

            if (iteration < pdhg_tools.LOG_FIRST_ITERATIONS) or (
                iteration % pdhg_tools.LOG_INTERVAL == 0
            ):
                if metrics is None:
                    metrics = pdhg_tools.compute_optimality_metrics(problem, a, x, y)

                self.log_metrics(metrics, iteration, time.time() - start)

        raise RuntimeError("Failed to solve LP")
