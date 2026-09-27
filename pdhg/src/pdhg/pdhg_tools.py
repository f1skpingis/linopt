from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from common import lp_problem
import numpy as np
from common.numpy_type_aliases import ArrayF
from numpy.typing import NDArray


LOG_INTERVAL = 100
LOG_FIRST_ITERATIONS = 10

OPTIMALITY_CHECK_INTERVAL = 20
CRUDE_OPTIMALITY_TOLERANCE = 1e-4
MODERATE_OPTIMALITY_TOLERANCE = 1e-6
ACCURATE_OPTIMALITY_TOLERANCE = 1e-8



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


class LinearOperator(Protocol):
    @property
    def shape(self) -> tuple[int, int]: ...

    def matvec(self, x: ArrayF) -> ArrayF: ...

    def rmatvec(self, y: ArrayF) -> ArrayF: ...


@dataclass(frozen=True)
class VStackOperator:
    operators: tuple[LinearOperator, ...]

    @property
    def shape(self) -> tuple[int, int]:
        n = self.operators[0].shape[1]

        if any(op.shape[1] != n for op in self.operators):
            raise ValueError("All operators must have the same input dimension")

        return (
            sum(op.shape[0] for op in self.operators),
            n,
        )

    def matvec(self, x: ArrayF) -> ArrayF:
        return np.concatenate([op.matvec(x) for op in self.operators])

    def rmatvec(self, y: ArrayF) -> ArrayF:
        offset = 0
        result = np.zeros(self.shape[1], dtype=y.dtype)

        for op in self.operators:
            m = op.shape[0]
            result += op.rmatvec(y[offset : offset + m])
            offset += m

        return result


def vstack(*operators: LinearOperator) -> LinearOperator:
    return VStackOperator(tuple(operators))


class DenseLinearOperator:
    def __init__(self, a: NDArray[np.float64]) -> None:
        self.a = a

    @property
    def shape(self) -> tuple[int, int]:
        return self.a.shape

    def matvec(self, x: ArrayF) -> ArrayF:
        return self.a @ x

    def rmatvec(self, y: ArrayF) -> ArrayF:
        return self.a.T @ y


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

    random_vec = np.random.rand(size)
    random_vec /= np.linalg.norm(random_vec)

    old_norm = np.linalg.norm(operator.matvec(random_vec))

    iters_left = max_iter
    while iters_left > 0:
        iteration = max_iter - iters_left

        random_vec = operator.rmatvec(operator.matvec(random_vec))
        norm = np.linalg.norm(random_vec)
        random_vec /= norm

        iters_left -= 1

        if iteration % 10 == 0:
            norm = np.linalg.norm(operator.matvec(random_vec))
            if np.isclose(norm, old_norm, rtol=tol, atol=0.0):
                return PowerIterationResult(
                    spectral_norm=norm,
                    iterations=max_iter - iters_left,
                    status=PowerIterationStatus.CONVERGED,
                )
            old_norm = norm

    spectral_norm = np.linalg.norm(operator.matvec(random_vec))
    return PowerIterationResult(
        spectral_norm=spectral_norm,
        iterations=max_iter,
        status=PowerIterationStatus.MAX_ITER,
    )


# TODO(martins): add proper docstring, take s as input instead
def lambda_projection(lower_bounds: ArrayF, upper_bounds: ArrayF, s: ArrayF) -> ArrayF:

    free = np.isneginf(lower_bounds) & np.isposinf(upper_bounds)
    lower_only = np.isfinite(lower_bounds) & np.isposinf(upper_bounds)
    upper_only = np.isneginf(lower_bounds) & np.isfinite(upper_bounds)

    lam = s.copy()
    lam[free] = 0
    lam[lower_only] = np.maximum(0.0, lam[lower_only])
    lam[upper_only] = np.minimum(0.0, lam[upper_only])

    return lam


def compute_primal_objective(c: ArrayF, x: ArrayF) -> float:
    return float(c @ x)


# TODO(martin): Consider setting q as a cached property on LpProblem.
# TODO(martin): Formalizing bounds into their own object could enable
# the construction of the "finite" bounds below to be a @property.
def compute_dual_objective(
    y: ArrayF, lam: ArrayF, q: ArrayF, lower_bounds: ArrayF, upper_bounds: ArrayF
) -> float:
    finite_lower_bounds = lower_bounds > -np.inf
    finite_upper_bounds = upper_bounds < np.inf
    return float(
        q.T @ y
        + lower_bounds[finite_lower_bounds].T
        @ np.maximum(0.0, lam[finite_lower_bounds])
        - upper_bounds[finite_upper_bounds] @ np.minimum(0.0, lam[finite_upper_bounds])
    )


def compute_relative_duality_gap(
    primal_objective: float, dual_objective: float
) -> float:
    div_by_zero_protection = 1e-9
    return abs(dual_objective - primal_objective) / (
        abs(dual_objective) + abs(primal_objective) + div_by_zero_protection
    )


def compute_primal_residual(
    k: LinearOperator, q: ArrayF, x: ArrayF, num_inequality_constraints: int
) -> float:
    r = k.matvec(x) - q
    r[0:num_inequality_constraints] = np.maximum(0.0, -r[0:num_inequality_constraints])
    return np.linalg.norm(r)


# TODO(martins): Think about having a protocol for vectors so these can be used with both sparse and dense
def compute_dual_residual(
    c: ArrayF, k: LinearOperator, y: ArrayF, lam: ArrayF
) -> float:
    return np.linalg.norm(c - k.rmatvec(y) - lam)


# These are very similar and could be the same, but since they correspond to different
# optimality conditions I think it is instructive to have two different names
def primal_acceptable(primal_residual: float, q_norm: float, tolerance: float) -> bool:
    return primal_residual <= tolerance * (1 + q_norm)


def dual_acceptable(dual_residual: float, c_norm: float, tolerance: float) -> bool:
    return dual_residual <= tolerance * (1 + c_norm)


def gap_acceptable(
    primal_objective: float, dual_objective: float, tolerance: float
) -> bool:
    return abs(dual_objective - primal_objective) <= tolerance * (
        1 + abs(dual_objective) + abs(primal_objective)
    )


@dataclass(frozen=True)
class PdhgOptimalityMetrics:
    primal_objective: float
    dual_objective: float
    primal_residual: float
    dual_residual: float
    relative_gap: float

    def is_optimal(self, c_norm: float, q_norm: float, tolerance: float) -> bool:
        return (
            gap_acceptable(self.primal_objective, self.dual_objective, tolerance)
            and primal_acceptable(self.primal_residual, q_norm, tolerance)
            and dual_acceptable(self.dual_residual, c_norm, tolerance)
        )


def compute_optimality_metrics(
    problem: lp_problem.LpProblem,
    k: LinearOperator,
    x: ArrayF,
    y: ArrayF,
    q: ArrayF,
) -> PdhgOptimalityMetrics:
    s = problem.objective - k.rmatvec(y)
    lam = lambda_projection(
        problem.effective_lower_bounds, problem.effective_upper_bounds, s
    )

    primal_objective = compute_primal_objective(problem.objective, x)
    dual_objective = compute_dual_objective(
        y,
        lam,
        q,
        problem.effective_lower_bounds,
        problem.effective_upper_bounds,
    )

    primal_residual = compute_primal_residual(
        k, q, x, problem.num_inequality_constraints
    )
    dual_residual = compute_dual_residual(problem.objective, k, y, lam)

    gap = compute_relative_duality_gap(primal_objective, dual_objective)

    return PdhgOptimalityMetrics(
        primal_objective=primal_objective,
        dual_objective=dual_objective,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        relative_gap=gap,
    )
