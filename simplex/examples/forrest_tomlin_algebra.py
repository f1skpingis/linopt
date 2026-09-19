"""Worked dense Forrest-Tomlin algebra from Huangfu and Hall, equations 11-13.

This educational example does not exercise the product-form solver backend and
is not an implementation test. Run from the repository root with:
    .venv/bin/python simplex/examples/forrest_tomlin_algebra.py
"""

import numpy as np


def demonstrate_forrest_tomlin_update() -> None:
    # Dense matrices expose the paper algebra; this does not call a solver backend.
    l_factor = np.array(
        [
            [1.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [-1.0, 3.0, 1.0],
        ]
    )
    u_factor = np.array(
        [
            [2.0, -1.0, 4.0],
            [0.0, 3.0, 5.0],
            [0.0, 0.0, -2.0],
        ]
    )

    basis = np.array([0, 2, 4])
    entering_variable = 5
    exiting_index = 1

    # The current basis is B = L U.
    basis_matrix = l_factor @ u_factor
    constraint_matrix = np.array(
        [
            [0.0, 7.0, 0.0, 6.0, 0.0, 3.0],
            [0.0, 8.0, 0.0, 5.0, 0.0, 11.0],
            [0.0, 2.0, 0.0, 4.0, 0.0, -1.0],
        ]
    )
    constraint_matrix[:, basis] = basis_matrix

    np.testing.assert_allclose(
        l_factor @ u_factor,
        constraint_matrix[:, basis],
        rtol=1e-6,
        atol=1e-12,
    )

    # Pivot: replace basis column p by the entering column a_q.
    updated_basis = basis.copy()
    updated_basis[exiting_index] = entering_variable

    # Equation 11:
    #   L^-1 B_bar = U + (L^-1 a_q - U e_p) e_p^T = U0.
    #
    # So the basis pivot is expressed in U-space by replacing column p of U
    # with the partial FTRAN result a_tilde_q = L^-1 a_q.
    a_tilde_q = np.linalg.solve(l_factor, constraint_matrix[:, entering_variable])
    np.testing.assert_allclose(
        a_tilde_q,
        np.array([3.0, 5.0, -13.0]),
        rtol=1e-6,
        atol=1e-12,
    )

    spiked_u = u_factor.copy()
    spiked_u[:, exiting_index] = a_tilde_q
    expected_spiked_u = np.array(
        [
            [2.0, 3.0, 4.0],
            [0.0, 5.0, 5.0],
            [0.0, -13.0, -2.0],
        ]
    )
    np.testing.assert_allclose(
        spiked_u,
        expected_spiked_u,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        l_factor @ spiked_u,
        constraint_matrix[:, updated_basis],
        rtol=1e-6,
        atol=1e-12,
    )

    # Equation 12:
    #   r^T = e_p^T - u_pp * e_tilde_p^T,
    #   where e_tilde_p^T = e_p^T U^-1.
    #
    # The row eta R = I + e_p r^T has pivot 1, so R^-1 = I - e_p r^T.
    # Applying R^-1 is a row operation: it changes only row p.
    unit_p = np.eye(3)[exiting_index]
    partial_btran = unit_p @ np.linalg.inv(u_factor)
    pivot_in_u = u_factor[exiting_index, exiting_index]

    row_eta_vector = unit_p - pivot_in_u * partial_btran
    row_eta_vector[exiting_index] = 0.0
    np.testing.assert_allclose(
        row_eta_vector,
        np.array([0.0, 0.0, -2.5]),
        rtol=1e-6,
        atol=1e-12,
    )

    row_eta = np.eye(3) + np.outer(unit_p, row_eta_vector)
    row_eta_inverse = np.eye(3) - np.outer(unit_p, row_eta_vector)
    np.testing.assert_allclose(
        row_eta @ row_eta_inverse,
        np.eye(3),
        rtol=1e-6,
        atol=1e-12,
    )

    # R^-1 U eliminates the off-diagonal entries in row p of the old U.
    np.testing.assert_allclose(
        row_eta_inverse @ u_factor,
        np.array([[2.0, -1.0, 4.0], [0.0, 3.0, 0.0], [0.0, 0.0, -2.0]]),
        rtol=1e-6,
        atol=1e-12,
    )

    # Applying the same row operation to U0 gives U_bar. In the paper this is
    # "permuted triangular": the off-diagonal entries of row p are gone, while
    # the spike lives in column p below the diagonal.
    u_bar = row_eta_inverse @ spiked_u
    np.testing.assert_allclose(
        u_bar,
        np.array([[2.0, 3.0, 4.0], [0.0, -27.5, 0.0], [0.0, -13.0, -2.0]]),
        rtol=1e-6,
        atol=1e-12,
    )

    # Equation 13 for one update:
    #   B_bar = L R U_bar
    #   B_bar^-1 = U_bar^-1 R^-1 L^-1.
    #
    # Since U_bar = R^-1 U0, this stores the same updated basis matrix
    # without refactorizing B_bar from scratch.
    updated_basis_matrix = constraint_matrix[:, updated_basis]
    np.testing.assert_allclose(
        l_factor @ row_eta @ u_bar,
        updated_basis_matrix,
        rtol=1e-6,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.linalg.inv(u_bar) @ row_eta_inverse @ np.linalg.inv(l_factor),
        np.linalg.inv(updated_basis_matrix),
        rtol=1e-6,
        atol=1e-12,
    )

    # This is the same final basis matrix we would get from directly replacing
    # column p of B by a_q. The value above is the factorized representation.
    direct_basis_matrix = basis_matrix.copy()
    direct_basis_matrix[:, exiting_index] = constraint_matrix[:, entering_variable]
    np.testing.assert_allclose(
        direct_basis_matrix,
        constraint_matrix[:, updated_basis],
        rtol=1e-6,
        atol=1e-12,
    )


if __name__ == "__main__":
    demonstrate_forrest_tomlin_update()
    print("Forrest-Tomlin algebra checks passed (educational example only).")
