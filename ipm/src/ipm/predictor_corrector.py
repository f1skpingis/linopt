import logging
import time

import numpy as np
from common import lp_problem

from ipm import ipm_tools

logger = logging.getLogger(__name__)


def calculate_starting_point(
    problem: lp_problem.StandardFormLP,
) -> ipm_tools.PrimalDualTuple:
    """Calculates a suitable starting point for the predictor corrector algorithm
    according to the recipe on p. 410 in Nocedal & Wright"""
    aat_inv = np.linalg.inv(problem.constraint_matrix @ problem.constraint_matrix.T)

    x_tilde = problem.constraint_matrix.T @ aat_inv @ problem.rhs
    lam_tilde = aat_inv @ problem.constraint_matrix @ problem.objective
    s_tilde = problem.objective - problem.constraint_matrix.T @ lam_tilde

    delta_x = max(0.0, -1.5 * float(np.min(x_tilde)))
    delta_s = max(0.0, -1.5 * float(np.min(s_tilde)))

    x_hat = x_tilde + delta_x * np.ones(x_tilde.shape)
    s_hat = s_tilde + delta_s * np.ones(s_tilde.shape)

    delta_x_hat = 0.5 * (x_hat.T @ s_hat) / (sum(s_hat))
    delta_s_hat = 0.5 * (x_hat.T @ s_hat) / (sum(x_hat))

    x_0 = x_hat + delta_x_hat * np.ones(x_hat.shape)
    s_0 = s_hat + delta_s_hat * np.ones(s_hat.shape)
    lam_0 = lam_tilde
    return ipm_tools.PrimalDualTuple(x=x_0, lam=lam_0, s=s_0)


def update_point(
    point: ipm_tools.PrimalDualTuple,
    step: ipm_tools.PrimalDualTuple,
    primal_step_size: float,
    dual_step_size: float,
) -> ipm_tools.PrimalDualTuple:
    """Updates the current solution using a primal dual step and
    corresponding step sizes. Corresponds to the last two steps in the
    for-loop of algorithm 14.3 on p. 411 in the book."""
    return ipm_tools.PrimalDualTuple(
        x=point.x + primal_step_size * step.x,
        lam=point.lam + dual_step_size * step.lam,
        s=point.s + dual_step_size * step.s,
    )


class PredictorCorrector:
    def __init__(self, max_iterations: int, optimality_tolerance: float) -> None:
        self.max_iterations = max_iterations
        self.optimality_tolerance = optimality_tolerance

    def solve(self, problem: lp_problem.StandardFormLP) -> ipm_tools.PrimalDualTuple:
        """Solves the LP `problem` using algorithm 14.3 on p. 411 in Nocedal & Wright."""

        if not isinstance(problem, lp_problem.StandardFormLP):
            raise TypeError("This solver requires StandardFormLP (Ax = b, x >= 0).")
        point = calculate_starting_point(problem)

        logger.info("                Objective              Residual")
        logger.info(
            "Iter       Primal       Dual       Primal      Dual     Compl       Time"
        )

        iteration = 0
        start = time.time()
        while (
            ipm_tools.calculate_duality_measure(point.x, point.s)
            > self.optimality_tolerance
            and iteration < self.max_iterations
        ):
            direction = ipm_tools.solve_predictor_corrector_direction(problem, point)
            eta = 0.995 ** (1.0 / (iteration + 1))
            primal_step_size = min(
                ipm_tools.calculate_max_step_size(point.x, direction.x) * eta, 1.0
            )
            dual_step_size = min(
                ipm_tools.calculate_max_step_size(point.s, direction.s) * eta, 1.0
            )

            point = update_point(point, direction, primal_step_size, dual_step_size)
            logger.info(
                f"{iteration:4d}   {problem.objective @ point.x:10.3e}  "
                f"{problem.rhs @ point.lam:10.3e}  "
                f"{np.linalg.norm(problem.constraint_matrix @ point.x - problem.rhs):10.3e} "
                f"{np.linalg.norm(problem.constraint_matrix.T @ point.lam + point.s - problem.objective):10.3e}  "
                f"{ipm_tools.calculate_duality_measure(point.x, point.s):8.3e}    "
                f"{time.time() - start:5.2}s"
            )
            iteration += 1

        return point
