import numpy as np
from pdhg import pdlp
import pytest

from common import lp_problem
from common.netlib import load_netlib_problems
from ipm import predictor_corrector
from simplex import dual_simplex, primal_simplex
from simplex import pivoting_strategy
from scipy.optimize import linprog  # For benchmarking against

# Database of known optimum values and recommended parameters
# Note: MPS files do not contain the optimum objective value.
NETLIB_SOLUTIONS = {
    "afiro": {
        "optimum": -464.75,
        "simplex_iters": 100,
    },
    "adlittle": {
        "optimum": 225494.963160,
        "simplex_iters": 1000,
    },
    "bandm": {
        "optimum": -158.62801845,
        "simplex_iters": 100000,
    },
    "scsd1": {
        "optimum": 8.6666666,
        "simplex_iters": 100000,
    },  # IPM won't quite reach all the way on this one. OK for simplex
    "scsd6": {
        "optimum": 5.050000e1,
        "simplex_iters": 100000,
    },  # Takes forever in simplex
    "scsd8": {
        "optimum": 9.049999e2,
        "simplex_iters": 100000,
    },  # Takes forever in simplex also
    "stocfor2": {
        "optimum": -3.90244085e04,
        "simplex_iters": 100000,
    },
    "woodw": {
        "optimum": 1.3044763331e0,
        "simplex_iters": 100000,
    },
    "fit2d": {
        "optimum": -6.8464293294e04,
        "simplex_iters": 100000,
    },
    "pilot": {
        "optimum": -5.5740430007e02,
        "simplex_iters": 100000,
    },
    "pilotnov": {
        "optimum": -4.4972761882e03,
        "simplex_iters": 100000,
    },
    "d2q06c": {
        "optimum": 1.2278423615e05,
        "simplex_iters": 100000,
    },
    "maros-r7": {
        "optimum": 1.4971851665e06,
        "simplex_iters": 100000,
    },
    "dfl001": {
        "optimum": 1.12664e07,
        "simplex_iters": 100000,
    },  # Really hard. Think presolve will be necessary
}

# Add problem names to this list to test them (if they are in NETLIB_SOLUTIONS)
PROBLEMS_TO_TEST = [
    "afiro",
    "adlittle",
    "bandm",
    "scsd1",
    "scsd6",
    "scsd8",
    "stocfor2",
    "woodw",
    "fit2d",
    "pilot",
    "pilotnov",
    "d2q06c",
    "maros-r7",
    "dfl001",
]


@pytest.fixture(scope="module")
def cached_lp_problems() -> dict[str, lp_problem.StandardFormLP]:
    """Cache for downloaded and converted LP problems to avoid multiple downloads."""
    return {}


def get_problem(
    name: str,
    cache: dict[str, lp_problem.StandardFormLP],
) -> lp_problem.StandardFormLP:
    if name not in cache:
        res = load_netlib_problems.download_and_parse_mps(name)
        assert res is not None, f"Failed to download/parse {name}"

        a, b, c, row_types, _, _ = res

        a_std, b_std, c_std = load_netlib_problems.convert_to_standard_form(
            a, b, c, row_types
        )

        cache[name] = lp_problem.StandardFormLP(
            constraint_matrix=a_std,
            rhs=b_std,
            objective=c_std,
        )

    return cache[name]


@pytest.mark.parametrize("name", PROBLEMS_TO_TEST)
def test_netlib_ipm(
    name: str, cached_lp_problems: dict[str, lp_problem.StandardFormLP]
) -> None:
    """Test the IPM solver on a Netlib problem."""
    lp = get_problem(name, cached_lp_problems)
    optimum = NETLIB_SOLUTIONS[name]["optimum"]

    ipm_solver = predictor_corrector.PredictorCorrector(10000, 1e-7)
    solution = ipm_solver.solve(lp)

    obtained_optimum = float(lp.objective.T @ solution.x)
    assert np.isclose(obtained_optimum, optimum, rtol=1e-3)


@pytest.mark.parametrize("name", PROBLEMS_TO_TEST)
def test_netlib_primal_simplex(
    name: str, cached_lp_problems: dict[str, lp_problem.StandardFormLP]
) -> None:
    """Test the Primal Simplex solver on a Netlib problem."""
    lp = get_problem(name, cached_lp_problems)
    optimum = NETLIB_SOLUTIONS[name]["optimum"]
    iters = int(NETLIB_SOLUTIONS[name].get("simplex_iters", 1000))

    primal_simplex_solver = primal_simplex.PrimalSimplex(
        pivot_strategy=pivoting_strategy.DantzigsRule()
    )
    solution = primal_simplex_solver.solve(lp, max_iterations=iters)

    obtained_optimum = float(lp.objective.T @ solution.solution)
    assert np.isclose(obtained_optimum, optimum)


@pytest.mark.parametrize("name", PROBLEMS_TO_TEST)
def test_netlib_dual_simplex(
    name: str, cached_lp_problems: dict[str, lp_problem.StandardFormLP]
) -> None:
    """Test the Dual Simplex solver on a Netlib problem."""
    lp = get_problem(name, cached_lp_problems)
    optimum = NETLIB_SOLUTIONS[name]["optimum"]
    iters = int(NETLIB_SOLUTIONS[name].get("simplex_iters", 1000))

    dual_simplex_solver = dual_simplex.DualSimplex(
        pivot_strategy=pivoting_strategy.DualSteepestEdgeRule()
    )
    solution = dual_simplex_solver.solve(lp, max_iterations=iters)

    obtained_optimum = float(lp.objective.T @ solution.solution)
    assert np.isclose(obtained_optimum, optimum)


# AFIRO — vanilla PDHG
# tol=1e-8   23,800 iterations   0.27 s
# Diagonal preconditioning
# tol=1e-8   5,600             0.055 s
# Adaptive step size
# tol=1e-8  900                0.016 s
# SCSD1 - vanilla PDHG
# tol=1e-8   35,300 iterations.   0.51 s
# Diagonal preconditioning
# tol=1e-8   30,800               0.43 s
# Adaptive step size
# -||-      3,200      0.075s 
# Adlittle - vanilla PDGH
# tol=1e-8.  fail
# Digaonal preconditioning
# tol=1e-8. 212,200 iterations 2.2s
# Adaptive step size
# -||-    110,400.  iterations 1.6s 
# Bandm - vanilla PDHG
# tol=1e-8 fail
# Diagonal preconditioning
# tol=1e-8 fail
# Adaptive step size 
# -||-    1659400 39s
# SCSD6 - vanilla PDHG
# tol=1e-8 Didn't even start solve, spectral norm power iteration failed.
# tol=1e-8 170700 3.5s
# Diagonal preconditioning
# tol=1e-8 231800 4.8s
# Adaptive step size
# -||-  31,400  0.92s
@pytest.mark.parametrize("name", PROBLEMS_TO_TEST)
def test_netlib_pdhg(
    name: str, cached_lp_problems: dict[str, lp_problem.StandardFormLP]
) -> None:
    """Test the Dual Simplex solver on a Netlib problem."""
    lp = get_problem(name, cached_lp_problems)
    optimum = NETLIB_SOLUTIONS[name]["optimum"]
    iters = int(NETLIB_SOLUTIONS[name].get("simplex_iters", 1000))

    pdhg_solver = pdlp.PdlpSolver()
    y_size, x_size = lp.constraint_matrix.shape
    x = np.zeros(x_size)
    y = np.zeros(y_size)
    solution = pdhg_solver.solve(
        lp.to_bounded_form(), (x, y), max_iterations=iters * 1000
    )

    obtained_optimum = float(lp.objective.T @ solution[0])
    assert np.isclose(obtained_optimum, optimum)


# For comparison with SciPy.
@pytest.mark.parametrize("name", PROBLEMS_TO_TEST)
def test_netlib_scipy(
    name: str, cached_lp_problems: dict[str, lp_problem.StandardFormLP]
) -> None:
    """Test the Dual Simplex solver on a Netlib problem."""
    lp = get_problem(name, cached_lp_problems)
    optimum = NETLIB_SOLUTIONS[name]["optimum"]

    solution = linprog(
        lp.objective, A_eq=lp.constraint_matrix, b_eq=lp.rhs, options={"disp": True}
    )

    obtained_optimum = float(solution.fun)
    assert np.isclose(obtained_optimum, optimum)
