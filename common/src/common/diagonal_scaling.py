"""Coordinate maps for A_scaled = D1 A D2."""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from common.numpy_type_aliases import ArrayF


class PreconditionTransformation(Enum):
    """TO preconditioned coordinates divides; FROM multiplies."""

    TO = auto()
    FROM = auto()


@dataclass(frozen=True)
class DiagonalScaling:
    """Positive row (D1) and column (D2) factors defining a coordinate map."""

    row_factors: ArrayF
    column_factors: ArrayF

    def __post_init__(self) -> None:
        for factors in (self.row_factors, self.column_factors):
            if (
                factors.ndim != 1
                or not np.all(np.isfinite(factors))
                or np.any(factors <= 0)
            ):
                raise ValueError("Scaling factors must be finite, positive vectors.")

    def transform_primal(
        self, x: ArrayF, *, direction: PreconditionTransformation
    ) -> ArrayF:
        """TO: x_scaled = D2^-1 x; FROM: x = D2 x_scaled."""
        return self._transform(x, self.column_factors, direction)

    def transform_dual(
        self, y: ArrayF, *, direction: PreconditionTransformation
    ) -> ArrayF:
        """TO: y_scaled = D1^-1 y; FROM: y = D1 y_scaled."""
        return self._transform(y, self.row_factors, direction)

    @staticmethod
    def _transform(
        values: ArrayF, factors: ArrayF, direction: PreconditionTransformation
    ) -> ArrayF:
        if not isinstance(direction, PreconditionTransformation):
            raise ValueError("Expected a PreconditionTransformation direction.")
        if values.shape != factors.shape:
            raise ValueError("Coordinate vector must match the scaling factors.")
        return (
            values / factors
            if direction is PreconditionTransformation.TO
            else factors * values
        )
