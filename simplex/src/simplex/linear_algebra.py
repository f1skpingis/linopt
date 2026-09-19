from dataclasses import dataclass

import jaxtyping
import numpy as np
import scipy.linalg
from common.numpy_type_aliases import ArrayF, ArrayI
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular


@dataclass
class _ColumnEta:
    index: int
    indices: jaxtyping.Int[ArrayI, " nnz"]
    values: jaxtyping.Float[ArrayF, " nnz"]
    pivot: float

    @classmethod
    def from_column(
        cls,
        index: int,
        column: jaxtyping.Float[ArrayF, " m"],
    ) -> "_ColumnEta":
        nonzero_indices = np.flatnonzero(column)
        pivot = float(column[index])
        if np.isclose(pivot, 0.0):
            raise np.linalg.LinAlgError(
                "ProductFormFactorization update produced a zero eta pivot."
            )
        return cls(
            index=index,
            indices=nonzero_indices,
            values=np.asarray(column[nonzero_indices], dtype=np.float64),
            pivot=pivot,
        )

    def apply_inverse(
        self, vector: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve E x = vector for a column eta matrix E."""
        result = vector.copy()
        pivot_component = result[self.index] / self.pivot
        result[self.indices] -= self.values * pivot_component
        result[self.index] = pivot_component
        return result

    def apply_inverse_transpose(
        self, vector: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve E.T x = vector for a column eta matrix E."""
        result = vector.copy()
        dot_without_pivot = float(self.values @ vector[self.indices]) - (
            self.pivot * vector[self.index]
        )
        result[self.index] = (vector[self.index] - dot_without_pivot) / self.pivot
        return result

    def matrix(self, matrix_size: int) -> sparse.csr_array:
        eta_matrix = sparse.eye(matrix_size, format="lil")
        eta_matrix[:, self.index] = sparse.csc_array(
            (self.values, (self.indices, np.zeros_like(self.indices))),
            shape=(matrix_size, 1),
        )
        return eta_matrix.tocsr()


class ProductFormFactorization:
    """
    LU-backed product-form (PF) basis factorization with column eta updates.

    The representation after k updates is
        B = P L U E_1 E_2 ... E_k,
    so solving with the basis uses
        B^-1 = E_k^-1 ... E_2^-1 E_1^-1 U^-1 L^-1 P^T.

    Initial LU is computed densely and stored as sparse triangular factors.
    Those factors stay fixed while basis replacements append column eta records.
    FTRAN applies eta inverses oldest first; BTRAN applies inverse transposes
    newest first. This is PF updating, not Forrest-Tomlin updating.
    """

    def __init__(self, basis_matrix: jaxtyping.Float[ArrayF, "m m"]) -> None:
        permutation, l_factor, u_factor = scipy.linalg.lu(
            np.asarray(basis_matrix, dtype=np.float64)
        )
        self.permutation = sparse.csr_array(permutation)
        self.l_factor = sparse.csc_array(l_factor)
        self.u_factor = sparse.csc_array(u_factor)
        self.basis_etas: list[_ColumnEta] = []

    @classmethod
    def from_factors(
        cls,
        l_factor: jaxtyping.Float[ArrayF, "m m"],
        u_factor: jaxtyping.Float[ArrayF, "m m"],
    ) -> "ProductFormFactorization":
        factorization = cls.__new__(cls)
        factorization.permutation = sparse.eye(l_factor.shape[0], format="csr")
        factorization.l_factor = sparse.csc_array(l_factor)
        factorization.u_factor = sparse.csc_array(u_factor)
        factorization.basis_etas = []
        return factorization

    def ftran(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve B x = rhs, returning x = B^-1 rhs."""
        partial = self._left_factor_inverse_times(rhs)
        return self._u_ftran(partial)

    def btran(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        """Solve B.T x = rhs, returning x = B^-T rhs."""
        result = self._u_btran(rhs)
        result = np.asarray(
            spsolve_triangular(
                self.l_factor.T,
                result,
                lower=False,
                unit_diagonal=True,
            ),
            dtype=np.float64,
        )
        return np.asarray(self.permutation @ result, dtype=np.float64)

    def update(
        self,
        entering_column: jaxtyping.Float[ArrayF, " m"],
        exiting_index: int,
    ) -> None:
        """
        Replace basis position `exiting_index` by `entering_column`.

        Compute d = B^-1 entering_column and append the column eta
        E = I + (d - e_p) e_p.T, where p = exiting_index. Then B_new = B E.
        The base L and U factors are unchanged.
        """
        partial_column = self._left_factor_inverse_times(entering_column)
        eta_column = self._u_ftran(partial_column)
        self.basis_etas.append(_ColumnEta.from_column(exiting_index, eta_column))

    def to_matrix(self) -> jaxtyping.Float[ArrayF, "m m"]:
        """Reconstruct the represented basis matrix. Intended for tests/debugging."""
        result = self.permutation @ self.l_factor
        result = result @ self.u_factor
        for eta in self.basis_etas:
            result = result @ eta.matrix(result.shape[0])
        return np.asarray(result.toarray(), dtype=np.float64)

    def _left_factor_inverse_times(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        result: ArrayF = np.asarray(
            spsolve_triangular(
                self.l_factor,
                self.permutation.T @ rhs,
                lower=True,
                unit_diagonal=True,
            ),
            dtype=np.float64,
        )
        return result

    def _u_ftran(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        result: ArrayF = np.asarray(
            spsolve_triangular(self.u_factor, rhs, lower=False),
            dtype=np.float64,
        )
        for eta in self.basis_etas:
            result = eta.apply_inverse(result)
        return result

    def _u_btran(
        self, rhs: jaxtyping.Float[ArrayF, " m"]
    ) -> jaxtyping.Float[ArrayF, " m"]:
        result: ArrayF = np.asarray(rhs, dtype=np.float64)
        for eta in reversed(self.basis_etas):
            result = eta.apply_inverse_transpose(result)
        return np.asarray(
            spsolve_triangular(self.u_factor.T, result, lower=True),
            dtype=np.float64,
        )


def update_inverse(
    a: jaxtyping.Float[ArrayF, "m n"],
    b_inv: jaxtyping.Float[ArrayF, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[ArrayF, "m m"]:
    """
    Computes `B_new^-1` where `B_new` is formed by replacing column `exiting_index`
    in the matrix `B` with the column `A[:, entering_variable]`.

    Uses the Sherman-Morrison formula,
    ```
    (B + u * v^T)^-1 = B^-1 - (B^-1 * u * v^T * B^-1)/(1 + v^T * B^-1 * u),
    ```
    with
    ```
    u = -B[:, exiting_index] + A[:, entering_variable],
    ```
    and `v` a vector with zeros everywhere except a 1 at `exiting_index`.

    The product of the old inverse with the column vector u becomes,
    ```
    B^-1 * u = -v + B^-1 * A[:, entering_variable] = -v + d,
    ```
    where `d` is the basic direction vector.

    Approximate time consumption on Optdev's Dell machines: 3.5 ms

    Args:
        a: constraint matrix.
        b_inv: inverse of the current basis matrix.
        entering_variable: variable index for the variable entering the basis.
        exiting_index: index of the column in the basis matrix that should be replaced.

    Returns:
        inverse of the updated basis matrix, `B_new^-1`.
    """

    # O(m^2)
    d = b_inv @ a[:, [entering_variable]]
    v = np.zeros((a.shape[0], 1))
    v[exiting_index] = 1

    # O(m^2)
    return b_inv - ((-v + d) @ b_inv[[exiting_index], :]) / np.float64(d[exiting_index])


def update_inverse_gaussian(
    a: jaxtyping.Float[ArrayF, "m n"],
    b_inv: jaxtyping.Float[ArrayF, "m m"],
    entering_variable: int,
    exiting_index: int,
) -> jaxtyping.Float[ArrayF, "m m"]:
    """
    Computes `B_new^-1` where `B_new` is formed by replacing column `exiting_index`
    in the matrix `B` with the column `A[:, entering_variable]`.

    Follows the method on page 97 of "Introduction to Linear Optimization",
    The new basis matrix is
    ```
    B_new = [A_B(1), A_B(2), ... , A_l, ...  A_B(m).
    ```
    where `l` is the entering index.
    To find the inverse of this matrix, we compute the product between this matrix
    and the old inverse
    ```
    B^-1 * B_new = [e_1, ...  d_l, e_m]
    ```
    If we now perform a set of row operations, `Q`, to turn the `l`th
    into the `l`th unit vector, we obtain the identity matrix,
    ```
    Q * B^-1 * B_new = I,
    ```
    thus, `Q * B^-1 = B_new^-1`.

    For efficiency, it suffices to just compute `d_l` (basic direction of entering variable)
    to determine the necessary row operations.

    Approximate time consumption on Optdev's Dell machines:  4.0 ms

    Args:
        a: constraint matrix.
        b_inv: inverse of the current basis matrix.
        entering_variable: variable index for the variable entering the basis.
        exiting_index: index of the column in the basis matrix that should be replaced.

    Returns:
        inverse of the updated basis matrix, `B_new^-1`.

    """
    # O(m^2)
    d = b_inv @ a[:, [entering_variable]]
    dl = d[exiting_index]
    rowl = b_inv[exiting_index]

    # O(m^2)
    return np.array(
        [
            row - (rowl * d[i] / dl) if i != exiting_index else row / dl
            for i, row in enumerate(b_inv)
        ]
    )
