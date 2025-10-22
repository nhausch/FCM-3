import numpy as np

# The generator for non-singular matrices.
class MatrixGenerator:
    def __init__(self):
        self.vector_distribution_params = {
            'low': 0.0,
            'high': 0.99,
        }
        self.distribution_params = {
            'low': -0.5,
            'high': 0.5,
        }
        self.round_to_int = False

    def generate_non_singular_lower_triangular(self, size, well_conditioned: bool = True):
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1):
                if well_conditioned:
                    matrix[i, j] = np.random.uniform(
                        self.distribution_params['low'],
                        self.distribution_params['high']
                    )
                else:
                    matrix[i, j] = np.random.uniform(
                        self.distribution_params['low'] + 0.5,
                        self.distribution_params['high'] + 0.5
                    )

        # Ensure non-singularity by making sure all diagonal elements are not close to zero.
        for i in range(size):
            if abs(matrix[i, i]) < 1e-10:  
                matrix[i, i] = 0.5

        if self.round_to_int:
            matrix = np.round(matrix).astype(int)
        return matrix

    def generate_non_singular_upper_triangular(self, size, well_conditioned: bool = True):
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i, size):
                if well_conditioned:
                    matrix[i, j] = np.random.uniform(
                        self.distribution_params['low'],
                        self.distribution_params['high']
                    )
                else:
                    matrix[i, j] = np.random.uniform(
                        self.distribution_params['low'] + 0.5,
                        self.distribution_params['high'] + 0.5
                    )

        # Ensure non-singularity by making sure all diagonal elements are not close to zero.
        for i in range(size):
            if abs(matrix[i, i]) < 1e-10:
                matrix[i, i] = 0.5
        
        if self.round_to_int:
            matrix = np.round(matrix).astype(int)
            
        return matrix

    def generate_non_singular_square(self, size, well_conditioned: bool = True):

        # Generate a non-singular matrix by multiplying a lower triangular matrix by an 
        # upper triangular matrix.
        lower = self.generate_non_singular_lower_triangular(size, well_conditioned)
        upper = self.generate_non_singular_upper_triangular(size, well_conditioned)
        matrix = np.matmul(lower, upper)

        # Add diagonal dominance for well-conditioned matrices.
        if well_conditioned:
            for i in range(size):
                matrix[i, i] += size
        return matrix

    def generate_vector(self, size):
        vector = np.random.uniform(
            self.vector_distribution_params['low'],
            self.vector_distribution_params['high'],
            size
        )
        if self.round_to_int:
            vector = np.round(vector).astype(int)   
        return vector
