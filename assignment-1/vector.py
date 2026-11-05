class Vector:
    def __init__(self, values):
        self.values = list(values)

    def __str__(self):
        return str(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def mean(self):
        """Return the mean of the vector entries."""

        if len(self.values) == 0:
            raise ValueError("Mean is not defined for an empty vector.")

        return sum(self.values) / len(self.values)

    def demean(self):
        """Return a new vector with the mean subtracted."""

        mean_value = self.mean()

        return Vector([x - mean_value for x in self.values])

    def std(self):
        """Return the population standard deviation."""

        if len(self.values) == 0:
            raise ValueError(
                "Standard deviation is not defined for an empty vector."
            )

        # Get the de-meaned vector
        demeaned = self.demean()

        # Square every deviation
        squared_deviations = [
            x * x for x in demeaned.values
        ]

        # Calculate average of squared deviations
        variance = (
            sum(squared_deviations)
            / len(squared_deviations)
        )

        # Square root of variance
        return variance ** 0.5
    
if __name__ == "__main__":

    vector = Vector([10, 22, 27, 8, 14])

    print("Original Vector:")
    print(vector.values)

    # Mean
    print("\nMean:")
    print(vector.mean())

    # Demean
    print("\nDemeaned Vector:")
    print(vector.demean().values)

    # Standard deviation
    print("\nStandard Deviation:")
    print(vector.std())