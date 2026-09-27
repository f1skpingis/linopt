import jaxtyping
import numpy as np
from common import lp_problem
from common.numpy_type_aliases import ArrayF

from ipm import ipm_tools, predictor_corrector


class TestStepSizeCalculation:
    def test_calculate_maximum_stepsize(self) -> None:
        """Test calculation  of the step sizes, `\alpha_k^{pri}` and `\alpha_k^{dual}`, in the
        predictor corrector algorithm is correct."""
        x = np.array([4.0, 6.0, 2.0, 8.0, 9.0, 3.0, 1.0])
        dx = np.array([0.0, -1.0, 3.0, -1000.0, 10.0, 1000.0, 1.0])

        # Should be -8 / -1000 = 0.008
        assert ipm_tools.calculate_max_step_size(x, dx) == 0.008

    def test_affine_scaling_step(self) -> None:
        """Test calculation of the affine scaling step calculation"""
        x = np.array([4.0, 6.0, 2.0, 8.0, 9.0, 3.0, 1.0])
        dx = np.array([0.0, -1.0, 3.0, -1000.0, 10.0, 1000.0, 1.0])

        # Should be -8 / -1000 = 0.008
        assert ipm_tools.calculate_affine_step_size(x, dx) == 0.008

        dx_long = np.array([0.0, -0.1, 0.3, -5.0, 1.0, 100.0, 0.5])

        # All fractions above are larger than one, but tightest is 8 / -5 = -1.6
        assert ipm_tools.calculate_max_step_size(x, dx_long) == 1.6
        assert ipm_tools.calculate_affine_step_size(x, dx_long) == 1.0


class TestDualityMeasureCalculation:
    def test_calculate_duality_measure(self) -> None:
        """Tests calculation of the duality measure which is the average
        of the pairwise products `x_i * s_i`"""
        primal_vars = np.array([1.0, 2.0, 3.0])
        dual_vars = np.array([0.0, 2.0, 0.0])
        assert ipm_tools.calculate_duality_measure(primal_vars, dual_vars) == 4 / 3

    def test_calculate_mu_after_step(self) -> None:
        """Tests calculation of the duality measure obtained after making a
        step in the primal and dual variables"""
        primal_step_size = 1.0
        dual_step_size = 2.0
        point = ipm_tools.PrimalDualTuple(
            x=np.array([1.0, 1.0, 2.0]),
            lam=np.zeros(3),
            s=np.array([2.0, 1.0, 3.0]),
        )
        step = ipm_tools.PrimalDualTuple(
            x=primal_step_size * np.array([0, -1, -1]),
            lam=np.zeros(3),
            s=dual_step_size * np.array([1, 0, -1]),
        )

        # Resulting duality measure should be
        # mu = ((1 + 0) * (2 + 2 * 1) + (1 - 1) * (1 + 0) + (2 - 1) * (3 - 2)) / 3
        #    = (4 + 1) / 3 = 5 / 3
        assert ipm_tools.calculate_mu_after_step(point, step) == 5 / 3

    def test_calculate_centering_parameter_for_predictor_corrector(self) -> None:
        """Test the centering parameter heuristic calculation which determines
        the centering parameter as `(mu_aff / mu)^3`"""

        primal_step_size = 1.0
        dual_step_size = 2.0
        point = ipm_tools.PrimalDualTuple(
            x=np.array([1.0, 1.0, 2.0]),
            lam=np.zeros(3),
            s=np.array([2.0, 1.0, 3.0]),
        )
        step = ipm_tools.PrimalDualTuple(
            x=np.array([0.0, -1.0, -1.0]) * primal_step_size,
            lam=np.zeros(3),
            s=np.array([1.0, 0.0, -1.0]) * dual_step_size,
        )

        mu = ipm_tools.calculate_duality_measure(point.x, point.s)
        mu_aff = ipm_tools.calculate_mu_after_step(point, step)

        assert mu == 3
        assert mu_aff == 5 / 3
        assert (
            ipm_tools.calculate_centering_parameter(point, step) == (mu_aff / mu) ** 3
        )


class TestPredictorCorrectorSolve:
    # LP with unit box as feasible sol.
    a = np.array(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
        ]
    )
    b = np.array(
        [
            1.0,
            1.0,
        ]
    )
    c = np.array(
        [
            1.0,
            1.0,
            0.0,
            0.0,
        ]
    )  # 0, 0, 1, 1, is the optimal solution.
    lp = lp_problem.LpProblem(constraint_matrix=a, rhs=b, objective=c)

    def test_solve_for_newton_direction(self) -> None:
        """Test computation of the newton direction, see (14.8) in the book."""

        # Primal variables
        x = 0.5 * np.ones(4)

        # Starting dual variables
        lam = -0.25 * np.ones(2)
        s = np.array([1.25, 1.25, 0.25, 0.25])  # Must be in the interior

        # Now let us solve for the Newton direction dx, dlam, ds
        #
        #
        #   | 0  A^T  I | |  dx  |   |  0   |
        #   |           | |      |   |      |
        #   | A   0   0 | | dlam | = |  0   |
        #   |           | |      |   |      |
        #   | S   0   X | |  ds  |   | -XSe |
        #

        # The point (x, lam, s) is on the central path. Thus, the direction
        # we obtain should be along it

        # dx = (-1, 1) / sqrt(2). Towards the optimum.
        # (dlam, ds) = 1, 1, -1, -1 to stay dual feasible
        expected_direction_x = -np.array([1.0, 1.0, -1.0, -1.0])
        expected_direction_lam = np.array([1.0, 1.0])
        expected_direction_s = np.array([-1.0, -1.0, -1.0, -1.0])

        newton_direction = ipm_tools.solve_newton_direction(
            self.lp,
            ipm_tools.PrimalDualTuple(x=x, lam=lam, s=s),
        )

        def normalize(
            v: jaxtyping.Float[ArrayF, " d"],
        ) -> jaxtyping.Float[ArrayF, " d"]:
            return v / np.linalg.norm(v)

        assert np.allclose(
            normalize(expected_direction_x), normalize(newton_direction.x)
        )
        assert np.allclose(
            normalize(expected_direction_lam), normalize(newton_direction.lam)
        )
        assert np.allclose(
            normalize(expected_direction_s), normalize(newton_direction.s)
        )

    def test_solve_for_predictor_corrector_direction(self) -> None:
        """Tests the computation of finding the next direction in the
        practical 'Predictor-Corrector' IPM due to Mehrotra."""

        # Primal variables
        x = 0.5 * np.ones(4)

        # Starting dual variables
        lam = -0.25 * np.ones(2)
        s = np.array([1.25, 1.25, 0.25, 0.25])  # Must be in the interior

        # Assert both sets are primal and dual feasible
        assert np.allclose(self.a @ x, self.b)
        assert np.allclose(self.a.T @ lam + s, self.c)
        assert np.all(x > 0)
        assert np.all(s > 0)

        # Now let us solve for the predictor corrector direction dx, dlam, ds
        #
        #
        #   | 0  A^T  I | |  dx  |   |              0              |
        #   |           | |      |   |                             |
        #   | A   0   0 | | dlam | = |              0              |
        #   |           | |      |   |                             |
        #   | S   0   X | |  ds  |   | -XSe - dXdSe + \sigma \mu e |
        #

        point = ipm_tools.PrimalDualTuple(x=x, lam=lam, s=s)
        newton_direction = ipm_tools.solve_newton_direction(self.lp, point)
        step = ipm_tools.calculate_affine_scaling_step(point, newton_direction)

        mu = ipm_tools.calculate_duality_measure(point.x, point.s)
        centering_param = ipm_tools.calculate_centering_parameter(point, step)

        predictor_corrector_direction = ipm_tools.solve_predictor_corrector_direction(
            self.lp, point
        )

        assert np.allclose(
            self.a.T @ predictor_corrector_direction.lam
            + predictor_corrector_direction.s,
            0,
        )
        assert np.allclose(self.a @ predictor_corrector_direction.x, 0)
        r_xs = (
            point.x * point.s
            + newton_direction.x * newton_direction.s
            - centering_param * mu * np.ones(point.x.shape)
        )
        assert np.allclose(
            point.s * predictor_corrector_direction.x
            + point.x * predictor_corrector_direction.s,
            -r_xs,
        )

    def test_start_solution(self) -> None:
        starting_point = predictor_corrector.calculate_starting_point(self.lp)

        assert np.all(starting_point.x > 0)
        assert np.all(starting_point.s > 0)

    def test_solve_lp_using_corrector_predictor_algo(self) -> None:

        # Test solving the lp
        solver = predictor_corrector.PredictorCorrector(100, 1e-12)
        solution = solver.solve(self.lp)

        expected_solution = [0.0, 0.0, 1.0, 1.0]
        assert np.allclose(solution.x, expected_solution, atol=1e-6)
        assert np.allclose(self.a @ solution.x, self.b, atol=1e-6)


class TestLpSolving:
    def test_chvatals_example(self) -> None:
        # See for ref: https://www.matem.unam.mx/~omar/math340/degenerate.html or
        # https://sites.math.washington.edu/~vinzant/teaching/407/Chvatal.pdf, page 31.
        # This is an example of an LP which cycles under some pivot rules.
        #
        # For IPM, cycling is of course not of any concearn.

        a = np.array(
            [
                [-0.5, 5.5, 2.5, -9.0, -1.0, 0.0, 0.0],
                [-0.5, 1.5, 0.5, -1.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        b = np.array([0.0, 0.0, 1.0], dtype=float).T
        c = np.array([-10.0, 57.0, 9.0, 24.0, 0.0, 0.0, 0.0], dtype=float)
        lp = lp_problem.LpProblem(constraint_matrix=a, rhs=b, objective=c)

        # Test solving the lp
        solver = predictor_corrector.PredictorCorrector(100, 1e-10)
        solution = solver.solve(lp)

        assert np.isclose(c.T @ solution.x, -1.0)
        assert np.allclose(a @ solution.x, b)
