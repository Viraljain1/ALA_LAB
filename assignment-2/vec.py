"""
Vec: the tuple-based vector class written for Assignment 1.

Pure Python only (math module); NumPy is deliberately not used here.
"""

import math


class Vec:
    __slots__ = ("elements",)

    def __init__(self, values):
        elements = tuple(values)
        if not elements:
            raise ValueError("Vec cannot be empty")
        for x in elements:
            if not isinstance(x, (int, float)):
                raise TypeError(f"Components must be int or float, got {type(x).__name__}")
        self.elements = elements

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, i):
        return self.elements[i]

    def __repr__(self):
        return f"Vec{self.elements}"

    def _same_dimension(self, other):
        if not isinstance(other, Vec):
            raise TypeError("Operand must be a Vec")
        if len(self) != len(other):
            raise ValueError("Vectors must have the same dimension")

    def add(self, other):
        self._same_dimension(other)
        return Vec(tuple(map(lambda p: p[0] + p[1], zip(self.elements, other.elements))))

    def scale(self, k):
        return Vec(tuple(k * x for x in self.elements))

    def dot(self, other):
        self._same_dimension(other)
        # math.fsum gives a correctly rounded sum of the products
        return math.fsum(x * y for x, y in zip(self.elements, other.elements))

    def norm(self):
        return math.sqrt(self.dot(self))

    def cosine_similarity(self, other):
        denominator = self.norm() * other.norm()
        if denominator == 0:
            raise ZeroDivisionError("cosine similarity needs two nonzero vectors")
        return self.dot(other) / denominator

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, k):
        return self.scale(k)

    def __rmul__(self, k):
        return self.scale(k)
