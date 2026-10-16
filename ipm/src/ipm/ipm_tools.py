from dataclasses import dataclass

import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF
from scipy import sparse


@dataclass(frozen=True)
class PrimalDualTuple:
    x: jaxtyping.Float[ArrayF, " n"]
    lam: jaxtyping.Float[ArrayF, " m"]
    s: jaxtyping.Float[ArrayF, " n"]


def calculate_max_step_size(
    x: jaxtyping.Float[ArrayF, " n"], dx: jaxtyping.Float[ArrayF, " n"]
) -> float:
    """Calculates the maximum possible step size in the direction `dx`
    such that no variable in `x` becomes negative. Used for both primal and
    dual variables. For details, see p. 409, (14.36) in Nocedal & Wright."""
    mask = (
        dx < -1e-6
    )  # TODO(martins): Introduce constant for this tolerance. Helps a lot with stability.
    if not np.any(mask):
        return 0.0
    distances = np.divide(-x[mask], dx[mask])
    return float(np.min(distances))


def calculate_affine_step_size(
    x: jaxtyping.Float[ArrayF, " n"], dx: jaxtyping.Float[ArrayF, " n"]
) -> float:
    """Calculates the maximum allowable step length for the affine scaling algorithm. Used for both primal and
    dual variables. For details, see p. 408, (14.32) in Nocedal & Wright."""
    return min(1.0, calculate_max_step_size(x, dx))


def calculate_duality_measure(
    primal_vars: jaxtyping.Float[ArrayF, " n"], dual_vars: jaxtyping.Float[ArrayF, " n"]
) -> float:
    r"""Calculate the duality measure, `mu`, which is the average complementary slackness."""
    return float(np.mean(primal_vars * dual_vars))


def calculate_mu_after_step(
    point: PrimalDualTuple,
    step: PrimalDualTuple,
) -> float:
    """Calculates the value of the duality measure if one steps the
    primal and dual variables according to the input data.
    See p. 408, (14.33) in Nocedal & Wright for details."""
    new_primal = point.x + step.x
    new_dual = point.s + step.s
    return calculate_duality_measure(new_primal, new_dual)


def solve_ipm_system(
    a: sparse.csr_array,
    point: PrimalDualTuple,
    d2: jaxtyping.Float[ArrayF, " n"],
    r_c: jaxtyping.Float[ArrayF, " n"],
    r_b: jaxtyping.Float[ArrayF, " m"],
    r_xs: jaxtyping.Float[ArrayF, " n"],
) -> PrimalDualTuple:
    """Solves the system (14.41) in the book.
    Input args:
        a: The LP constraint matrix
        point: The primal and dual variables, `(x, lambda, s)`
        d2: The primal variables divided by the dual, X * S^-1.
            Passed to not compute twice during predictor corrector solve.
        r_c: The dual feasibility residual, see p. 398 (14.7)
        r_b: The primal feasibility residual, see p. 398 (14.7)
        r_xs: Algorithm dependent duality residual, see for example (14.8), (14.9), (14.35)

    Output:
        Solution of system (14.41)
    """

    ad2a = (a * d2) @ a.T
    # Much faster than using dense
    dlam = sparse.linalg.spsolve(
        ad2a.tocsc(),
        -r_b - (a * d2) @ r_c + a @ (r_xs / point.s),
        use_umfpack=True,  # Umfpack gives big increase...
    )
    ds = -r_c - a.T @ dlam
    dx = -(r_xs / point.s) - (d2 * ds)
    return PrimalDualTuple(x=dx, lam=dlam, s=ds)


def calculate_centering_parameter(
    point: PrimalDualTuple, step: PrimalDualTuple
) -> float:
    """Calculates the centering parameter used to set up the system solved
    for the predictor-corrector step. See p. 408, (14.34).
    """
    return (
        float(
            calculate_mu_after_step(point, step)
            / calculate_duality_measure(point.x, point.s)
        )
        ** 3
    )


def solve_newton_direction(
    lp_problem: lp_problem.StandardFormLP,
    point: PrimalDualTuple,
) -> PrimalDualTuple:
    """Solves the system (14.35) in Nocedal & Wright for the Newton direction"""

    a = lp_problem.sparse_constraint_matrix
    b = lp_problem.rhs
    c = lp_problem.objective

    r_c = a.T @ point.lam + point.s - c
    r_b = a @ point.x - b

    r_xs = point.x * point.s

    return solve_ipm_system(a, point, point.x / point.s, r_c, r_b, r_xs)


def calculate_affine_scaling_step(
    point: PrimalDualTuple,
    newton_direction: PrimalDualTuple,
) -> PrimalDualTuple:
    """Scales the Newton direction with the affine scaling step sizes. See (14.33) on p. 408 in the book."""
    primal_affine_step_size = calculate_affine_step_size(
        x=point.x, dx=newton_direction.x
    )
    dual_affine_step_size = calculate_affine_step_size(x=point.s, dx=newton_direction.s)
    return PrimalDualTuple(
        x=newton_direction.x * primal_affine_step_size,
        lam=newton_direction.lam * dual_affine_step_size,
        s=newton_direction.s * dual_affine_step_size,
    )


def solve_predictor_corrector_direction(
    lp_problem: lp_problem.StandardFormLP,
    point: PrimalDualTuple,
) -> PrimalDualTuple:
    """Solves the system (14.35) in Nocedal & Wright for the predictor corrector direction"""

    a = lp_problem.sparse_constraint_matrix
    b = lp_problem.rhs
    c = lp_problem.objective

    r_c = a.T @ point.lam + point.s - c
    r_b = a @ point.x - b

    xs = point.x * point.s
    d2 = point.x / point.s

    newton_direction = solve_ipm_system(a, point, d2, r_c, r_b, xs)
    affine_step = calculate_affine_scaling_step(point, newton_direction)

    dxds = newton_direction.x * newton_direction.s
    e = np.ones(point.x.shape)
    mu = calculate_duality_measure(point.x, point.s)
    r_xs = xs + dxds - calculate_centering_parameter(point, affine_step) * mu * e

    return solve_ipm_system(a, point, d2, r_c, r_b, r_xs)
