import numpy as np

# The generator for non-singular matrices.
class MatrixGenerator:
    def __init__(self):

        # Keep the lower triangular values small.
        self.lower_triangular_distribution_params = {
            'low': 0.0,
            'high': 0.99,
        }

        # Make sure the values are comfortably away from zero, and not too large.
        self.upper_triangular_distribution_params = {
            'low': 0.2,
            'high': 5.0,
        }
        self.round_to_int = False

    def generate_non_singular_lower_triangular(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1):
                matrix[i, j] = np.random.uniform(
                    self.lower_triangular_distribution_params['low'],
                    self.lower_triangular_distribution_params['high']
                )

        # Ensure non-singularity by making sure all diagonal elements are not close to zero.
        for i in range(size):
            if abs(matrix[i, i]) < 1e-10:  
                matrix[i, i] = 0.99
        
        if self.round_to_int:
            matrix = np.round(matrix).astype(int)
            
        return matrix

    def generate_non_singular_upper_triangular(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                matrix[i, j] = np.random.uniform(
                    self.upper_triangular_distribution_params['low'],
                    self.upper_triangular_distribution_params['high']
                )

        # Ensure non-singularity by making sure all diagonal elements are not close to zero.
        for i in range(size):
            if abs(matrix[i, i]) < 1e-10:
                matrix[i, i] = 4.0
        
        if self.round_to_int:
            matrix = np.round(matrix).astype(int)
            
        return matrix

    def generate_non_singular_square(self, size):

        # Generate a non-singular matrix by multiplying a lower triangular matrix by an 
        # upper triangular matrix.
        lower = self.generate_non_singular_lower_triangular(size)
        upper = self.generate_non_singular_upper_triangular(size)
        matrix = np.matmul(lower, upper)
        return matrix
