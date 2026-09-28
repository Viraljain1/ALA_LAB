import unittest

from vector import Vector


class TestVector(unittest.TestCase):

    # Tests for mean()

    def test_mean(self):
        vector = Vector([1, 2, 3, 4, 5])

        self.assertEqual(vector.mean(), 3)

    def test_mean_property(self):
        vector = Vector([10, 20, 30])

        # Mean should be between minimum and maximum
        self.assertGreaterEqual(
            vector.mean(),
            min(vector.values)
        )

        self.assertLessEqual(
            vector.mean(),
            max(vector.values)
        )

    # Tests for demean()

    def test_demean(self):
        vector = Vector([1, 2, 3, 4, 5])

        demeaned = vector.demean()

        expected = Vector([-2, -1, 0, 1, 2])

        self.assertEqual(
            demeaned.values,
            expected.values
        )

    def test_demean_mean_is_zero(self):
        vector = Vector([2, 4, 6, 8])

        demeaned = vector.demean()

        # Mean of de-meaned vector should be zero
        self.assertAlmostEqual(
            demeaned.mean(),
            0.0
        )

    def test_demean_returns_new_vector(self):
        vector = Vector([1, 2, 3])

        demeaned = vector.demean()

        # It should return a Vector
        self.assertIsInstance(
            demeaned,
            Vector
        )

        # It should be a different object
        self.assertIsNot(
            demeaned,
            vector
        )

        # Original vector should not change
        self.assertEqual(
            vector.values,
            [1, 2, 3]
        )

    # -------------------------
    # Tests for std()
    # -------------------------

    def test_std(self):
        vector = Vector([1, 2, 3, 4, 5])

        # Standard deviation = sqrt(2)
        self.assertAlmostEqual(
            vector.std(),
            2 ** 0.5
        )

    def test_std_of_constant_vector(self):
        vector = Vector([5, 5, 5, 5])

        # All values are the same,
        # so standard deviation is zero.
        self.assertEqual(
            vector.std(),
            0
        )

    def test_std_non_negative(self):
        vector = Vector([2, 5, 8, 10])

        # Standard deviation cannot be negative
        self.assertGreaterEqual(
            vector.std(),
            0
        )


if __name__ == "__main__":
    unittest.main(verbosity = 2)