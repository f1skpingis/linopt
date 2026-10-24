from dataclasses import dataclass
from typing import Protocol

from common.numpy_type_aliases import ArrayF
import numpy as np


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
    def __init__(self, a: ArrayF) -> None:
        self.a = a

    @property
    def shape(self) -> tuple[int, int]:
        return self.a.shape

    def matvec(self, x: ArrayF) -> ArrayF:
        return self.a @ x

    def rmatvec(self, y: ArrayF) -> ArrayF:
        return self.a.T @ y
