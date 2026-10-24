import logging
import time
from dataclasses import replace

import numpy as np
from common import lp_problem
from common.diagonal_scaling import DiagonalScaling, PreconditionTransformation
from common.numpy_type_aliases import ArrayF
from common.preconditioning import pdlp_preconditioning

from pdhg import pdhg_tools

logger = logging.getLogger(__name__)


class PdlpSolver:
    """Baseline PDHG using the newer PDLP paper, Section 2, Eq. (3).

    TODO(martins): Add adaptive step size (see page 7 of the paper) and primal weight (Algorithm 3 in the paper)
    """

    def __init__(self, apply_preconditioning: bool = True) -> None:
        self.apply_preconditioning = apply_preconditioning

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
        """Solve with starts, results, and optimality checks in input coordinates.

        Internal diagonal scaling changes only the iteration coordinates.
        Existing recovery metadata on the input remains available to the caller.
        """
        if not isinstance(problem, lp_problem.BoundedFormLP):
            raise TypeError(
                "PDHG requires BoundedFormLP; use to_bounded_form() for standard form."
            )

        x, y = (np.array(v, dtype=float, copy=True) for v in start_solution)
        self.validate_start_solution(problem, x, y)

        # Track only scaling introduced by this solve. The input can already
        # carry a map to an earlier problem, but starts, results, and stopping
        # criteria must all use the coordinates of the supplied problem.
        preconditioned_problem = replace(problem, diagonal_preconditioning=None)
        if self.apply_preconditioning:
            preconditioned_problem = pdlp_preconditioning(preconditioned_problem)

        scaling = preconditioned_problem.diagonal_preconditioning
        if scaling is None:
            scaling = DiagonalScaling(
                np.ones(problem.constraint_matrix.shape[0]),
                np.ones(problem.num_variables),
            )

        x = scaling.transform_primal(x, direction=PreconditionTransformation.TO)
        y = scaling.transform_dual(y, direction=PreconditionTransformation.TO)

        bounds_norm = pdhg_tools.finite_row_bounds_norm(problem)
        c_norm = float(np.linalg.norm(problem.objective))

        primal_weight = pdhg_tools.initialize_primal_weight(
            float(np.linalg.norm(preconditioned_problem.objective)),
            pdhg_tools.finite_row_bounds_norm(preconditioned_problem),
        )

        step_size = (
            1.0 / np.max(np.abs(preconditioned_problem.constraint_matrix))
            if preconditioned_problem.constraint_matrix.size != 0
            else 1.0
        )
        step_size_parameters = pdhg_tools.PdhgStepSizeParameters(
            primal_weight=primal_weight, step_size=step_size
        )
        step_size_hat = step_size_parameters

        # TODO(martin): Centralize this type of logging format. It is the same for Simplex
        logger.info("Starting PDHG algorithm...")

        logger.info("                Objective              Residual")
        logger.info(
            "Iter       Primal       Dual       Primal      Dual     Gap        Time"
        )
        start = time.time()
        for iteration in range(max_iterations):
            metrics: pdhg_tools.PdhgOptimalityMetrics | None = None

            (x, y), _, step_size_hat = pdhg_tools.adaptive_pdhg_step(
                # The paper says having iteration start at 0, but that would lead to weird step sizes on the first iteration
                preconditioned_problem,
                step_size_hat,
                x,
                y,
                iteration + 1,
            )

            if iteration % pdhg_tools.OPTIMALITY_CHECK_INTERVAL == 0:
                metrics = pdhg_tools.compute_optimality_metrics(
                    problem,
                    scaling.transform_primal(
                        x, direction=PreconditionTransformation.FROM
                    ),
                    scaling.transform_dual(
                        y, direction=PreconditionTransformation.FROM
                    ),
                )
                if metrics.is_optimal(
                    c_norm=c_norm,
                    bounds_norm=bounds_norm,
                    tolerance=pdhg_tools.ACCURATE_OPTIMALITY_TOLERANCE,
                ):
                    return (
                        scaling.transform_primal(
                            x, direction=PreconditionTransformation.FROM
                        ),
                        scaling.transform_dual(
                            y, direction=PreconditionTransformation.FROM
                        ),
                    )

            if (iteration < pdhg_tools.LOG_FIRST_ITERATIONS) or (
                iteration % pdhg_tools.LOG_INTERVAL == 0
            ):
                if metrics is None:
                    metrics = pdhg_tools.compute_optimality_metrics(
                        problem,
                        scaling.transform_primal(
                            x, direction=PreconditionTransformation.FROM
                        ),
                        scaling.transform_dual(
                            y, direction=PreconditionTransformation.FROM
                        ),
                    )

                self.log_metrics(metrics, iteration, time.time() - start)

        raise RuntimeError("Failed to solve LP")
