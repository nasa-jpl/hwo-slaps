"""Bounded exact refinement of profiled residuals for difficult float inputs.

Fractions represent the original binary floats exactly. Normal matrices here
contain exact rational numbers, so they neither square a floating point
condition error nor subtract scalar signal information. Information and cross
products come from data residuals plus the exact Gaussian-prior penalty.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np


class ExactProfileRefinement:
    """Small, rank-consistent exact least squares; constructed only on demand."""

    MAX_WORK = 32768  # n_data * max(1, n_nuisance)**2
    MAX_NUISANCE = 16

    @classmethod
    def affordable(cls, n_data: int, n_nuisance: int) -> bool:
        return n_nuisance <= cls.MAX_NUISANCE and n_data * max(1, n_nuisance)**2 <= cls.MAX_WORK

    def __init__(self, design: np.ndarray, precision: np.ndarray, rank: int) -> None:
        n, p = design.shape
        columns = [[Fraction(float(value)) for value in design[:, j]] for j in range(p)]
        prior = [Fraction(float(value)) for value in precision]
        normal = [[sum((a*b for a, b in zip(columns[i], columns[j], strict=True)), Fraction())
                   + (prior[i] if i == j else 0) for j in range(p)] for i in range(p)]
        reduced = [row.copy() for row in normal]
        pivots = []
        for column in range(p):
            row = len(pivots)
            pivot = next((i for i in range(row, p) if reduced[i][column]), None)
            if pivot is None:
                continue
            reduced[row], reduced[pivot] = reduced[pivot], reduced[row]
            scale = reduced[row][column]
            reduced[row] = [value / scale for value in reduced[row]]
            for i in range(row + 1, p):
                scale = reduced[i][column]
                reduced[i] = [x - scale*y for x, y in zip(reduced[i], reduced[row], strict=True)]
            pivots.append(column)
        # A deliberately truncated nonzero direction is a numerical-rank policy,
        # not an exact dependency. Do not silently replace it by the full span.
        self.supported = len(pivots) == rank
        self.columns = [columns[i] for i in pivots]
        self.prior = [prior[i] for i in pivots]
        self.n_data = n
        self.inverse = []
        if not self.supported:
            return
        r = len(pivots)
        inverse = [[normal[i][j] for j in pivots] + [Fraction(k == row) for k in range(r)]
                   for row, i in enumerate(pivots)]
        for j in range(r):
            scale = inverse[j][j]
            inverse[j] = [value / scale for value in inverse[j]]
            for i in range(r):
                if i != j:
                    scale = inverse[i][j]
                    inverse[i] = [x - scale*y for x, y in zip(inverse[i], inverse[j], strict=True)]
        self.inverse = [row[r:] for row in inverse]

    def residual(self, values: np.ndarray):
        data = [Fraction(float(value)) for value in values]
        cross = [sum((a*b for a, b in zip(column, data, strict=True)), Fraction()) for column in self.columns]
        coefficients = [sum((a*b for a, b in zip(row, cross, strict=True)), Fraction()) for row in self.inverse]
        residual = [value - sum((column[i]*coefficient for column, coefficient in
                                zip(self.columns, coefficients, strict=True)), Fraction())
                    for i, value in enumerate(data)]
        return residual, coefficients

    def inner(self, first, second) -> Fraction:
        a, ac = first
        b, bc = second
        return sum((x*y for x, y in zip(a, b, strict=True)), Fraction()) + sum(
            (p*x*y for p, x, y in zip(self.prior, ac, bc, strict=True)), Fraction())
