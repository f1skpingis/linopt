from dataclasses import dataclass
from enum import StrEnum
import logging

import numpy as np
from common import lp_problem
from common.linear_algebra import LinearOperator
from common.lp_problem import dual_domain_projection
from common.numpy_type_aliases import ArrayF

LOG_INTERVAL = 100
LOG_FIRST_ITERATIONS = 10

OPTIMALITY_CHECK_INTERVAL = 20
CRUDE_OPTIMALITY_TOLERANCE = 1e-4
MODERATE_OPTIMALITY_TOLERANCE = 1e-6
ACCURATE_OPTIMALITY_TOLERANCE = 1e-8


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdhgStepSizeParameters:
    primal_weight: float
    step_size: float

    def __post_init__(self) -> None:
        if not self.primal_weight > 0:
            raise ValueError("Primal weight must be positive.")
        if not self.step_size > 0:
            raise ValueError("Step size must be positive.")

    @property
    def primal_step_size(self) -> float:
        return self.step_size / self.primal_weight

    @property
    def dual_step_size(self) -> float:
        return self.step_size * self.primal_weight


class PowerIterationStatus(StrEnum):
    CONVERGED = "CONVERGED"
    MAX_ITER = "MAX_ITER"


@dataclass
class PowerIterationResult:
    spectral_norm: float
    iterations: int
    status: PowerIterationStatus


def iterative_spectral_norm(
    operator: LinearOperator, tol: float = 1e-6, max_iter: int = 100
) -> PowerIterationResult:
    size = operator.shape[1]

    if size == 0:
        return PowerIterationResult(0.0, 0, PowerIterationStatus.CONVERGED)

    random_vec: ArrayF = np.random.default_rng().random(size)
    random_vec /= np.linalg.norm(random_vec)
    old_norm = np.linalg.norm(operator.matvec(random_vec))

    iters_left = max_iter
    while iters_left > 0:
        iteration = max_iter - iters_left

        random_vec = operator.rmatvec(operator.matvec(random_vec))
        norm = np.linalg.norm(random_vec)
        if norm == 0.0:
            return PowerIterationResult(
                0.0, iteration + 1, PowerIterationStatus.CONVERGED
            )
        random_vec /= norm

        iters_left -= 1

        if iteration % 10 == 0:
            norm = np.linalg.norm(operator.matvec(random_vec))
            if np.isclose(norm, old_norm, rtol=tol, atol=0.0):
                return PowerIterationResult(
                    spectral_norm=float(norm),
                    iterations=max_iter - iters_left,
                    status=PowerIterationStatus.CONVERGED,
                )
            old_norm = norm

    spectral_norm = np.linalg.norm(operator.matvec(random_vec))
    return PowerIterationResult(
        spectral_norm=float(spectral_norm),
        iterations=max_iter,
        status=PowerIterationStatus.MAX_ITER,
    )


def bound_support(values: ArrayF, lower: ArrayF, upper: ArrayF) -> float:
    """p(v; lower, upper) = upper.T v+ - lower.T v-.

    Both v+ = max(v, 0) and v- = -min(v, 0) are nonnegative in
    the newer paper. Mask zero entries to avoid undefined 0 * infinity.
    """
    positive = values > 0
    negative = values < 0
    return float(
        upper[positive] @ values[positive] + lower[negative] @ values[negative]
    )


def primal_update(
    problem: lp_problem.BoundedFormLP,
    a: LinearOperator,
    x: ArrayF,
    y: ArrayF,
    tau: float,
) -> ArrayF:
    """Equation (3a): projection onto the variable bounds X."""
    return np.clip(
        x - tau * (problem.objective - a.rmatvec(y)),
        problem.variable_lower,
        problem.variable_upper,
    )


def dual_update(
    problem: lp_problem.BoundedFormLP,
    a: LinearOperator,
    extrapolated_x: ArrayF,
    y: ArrayF,
    sigma: float,
) -> ArrayF:
    """Equation (3b), with y_hat = y - sigma A (2 x_new - x_old).

    y_new = y_hat - sigma proj_{[-row_upper, -row_lower]}(y_hat / sigma).
    Equality rows reduce to y_new = y + sigma (b - A x_bar).
    """
    y_hat = y - sigma * a.matvec(extrapolated_x)
    return y_hat - sigma * np.clip(
        y_hat / sigma, -problem.row_upper, -problem.row_lower
    )


def compute_dual_objective(
    problem: lp_problem.BoundedFormLP, y: ArrayF, reduced_costs: ArrayF
) -> float:
    """Equation (1): -p(-y; row bounds) - p(-r; variable bounds)."""
    return -bound_support(-y, problem.row_lower, problem.row_upper) - bound_support(
        -reduced_costs, problem.variable_lower, problem.variable_upper
    )


def compute_relative_duality_gap(
    primal_objective: float, dual_objective: float
) -> float:
    return abs(dual_objective - primal_objective) / (
        abs(dual_objective) + abs(primal_objective) + 1e-9
    )


@dataclass(frozen=True)
class PdhgOptimalityMetrics:
    primal_objective: float
    dual_objective: float
    primal_residual: float
    dual_residual: float
    relative_gap: float

    def is_optimal(self, c_norm: float, bounds_norm: float, tolerance: float) -> bool:
        # Retain the baseline's relative L2 stopping policy. The newer paper's
        # Section 7 experimental polishing tolerances are intentionally not used.
        return (
            abs(self.dual_objective - self.primal_objective)
            <= tolerance * (1 + abs(self.dual_objective) + abs(self.primal_objective))
            and self.primal_residual <= tolerance * (1 + bounds_norm)
            and self.dual_residual <= tolerance * (1 + c_norm)
        )


def finite_row_bounds_norm(problem: lp_problem.BoundedFormLP) -> float:
    """L2 scale of the largest finite bound magnitude per row."""
    lower = np.where(np.isfinite(problem.row_lower), np.abs(problem.row_lower), 0.0)
    upper = np.where(np.isfinite(problem.row_upper), np.abs(problem.row_upper), 0.0)
    return float(np.linalg.norm(np.maximum(lower, upper)))


def initialize_primal_weight(
    objective_norm: float, combine_bounds_norm: float
) -> float:
    if objective_norm > 0.0 and combine_bounds_norm > 0.0:
        return objective_norm / combine_bounds_norm
    return 1.0


def compute_optimality_metrics(
    problem: lp_problem.BoundedFormLP, x: ArrayF, y: ArrayF
) -> PdhgOptimalityMetrics:
    raw_reduced_costs = problem.objective - problem.lin_op_a.rmatvec(y)
    r = dual_domain_projection(
        problem.variable_lower, problem.variable_upper, raw_reduced_costs
    )
    projected_y = dual_domain_projection(problem.row_lower, problem.row_upper, y)
    ax = problem.lin_op_a.matvec(x)
    row_violation = ax - np.clip(ax, problem.row_lower, problem.row_upper)
    variable_violation = x - np.clip(x, problem.variable_lower, problem.variable_upper)
    primal = float(problem.objective @ x)
    dual = compute_dual_objective(problem, y, r)
    return PdhgOptimalityMetrics(
        primal,
        dual,
        float(np.linalg.norm(np.concatenate((row_violation, variable_violation)))),
        float(np.linalg.norm(np.concatenate((raw_reduced_costs - r, y - projected_y)))),
        compute_relative_duality_gap(primal, dual),
    )


def omega_norm(x: ArrayF, y: ArrayF, primal_weight: float) -> float:
    return np.sqrt(
        primal_weight * (np.linalg.norm(x, 2) ** 2)
        + (np.linalg.norm(y, 2) ** 2) / primal_weight
    )


def adaptive_pdhg_step(
    problem: lp_problem.BoundedFormLP,
    step_size_parameters: PdhgStepSizeParameters,
    x: ArrayF,
    y: ArrayF,
    iteration: int,
    logging: bool = False,
) -> tuple[tuple[ArrayF, ArrayF], PdhgStepSizeParameters, PdhgStepSizeParameters]:
    """Adaptive rule to control the PDLP step size

    The rule heuristically attempts to stay close to a condition from the convergence
    analysis of PDHG with which convergence can be proven.

    The condition is equation (5) of 2501.07018v2
      `eta <= ||z^{k+1} - z^k||^2_omega / (2 * (y^{k+1} - y^k)^T A (x^{k+1) - x^k))`

    If the step calculated with the current step size violates that condition,
    then a smaller step size is tried until the condition is satisfied.

    """
    max_iterations = 100

    if logging:
        logger.info("Iteration    eta    eta_prime    done")

    adaptive_algo_it = 0
    while adaptive_algo_it < max_iterations:
        x_prime = primal_update(
            problem, problem.lin_op_a, x, y, step_size_parameters.primal_step_size
        )
        y_prime = dual_update(
            problem,
            problem.lin_op_a,
            2 * x_prime - x,
            y,
            step_size_parameters.dual_step_size,
        )

        y_diff = y_prime - y
        x_diff = x_prime - x

        a_inner_product_between_diffs = -y_diff.T @ problem.lin_op_a.matvec(x_diff)
        if a_inner_product_between_diffs > 0:
            step_size_bar = omega_norm(
                x_diff, y_diff, step_size_parameters.primal_weight
            ) ** 2 / (2 * a_inner_product_between_diffs)
        else:
            step_size_bar = np.inf

        # TODO(martins): Give better names to these constaints and also
        # more intitive names to step_size_prime etc. Write a nice docstring.
        step_size_prime = min(
            (1 - ((iteration + 1) ** (-0.3))) * step_size_bar,
            (1 + ((iteration + 1) ** (-0.6))) * step_size_parameters.step_size,
        )

        if logging:
            logger.info(
                f"{adaptive_algo_it:<8}     {step_size_parameters.step_size:<6.3} {step_size_prime:<6.3}       {bool(step_size_parameters.step_size <= step_size_bar)}"
            )

        if step_size_parameters.step_size <= step_size_bar:
            return (
                (x_prime, y_prime),
                step_size_parameters,
                PdhgStepSizeParameters(
                    primal_weight=step_size_parameters.primal_weight,
                    step_size=step_size_prime,
                ),
            )
        step_size_parameters = PdhgStepSizeParameters(
            primal_weight=step_size_parameters.primal_weight, step_size=step_size_prime
        )
        max_iterations -= 1

    raise RuntimeError("Adaptive step hit max iteration limit.")
