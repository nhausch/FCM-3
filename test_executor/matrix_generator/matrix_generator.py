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
        self.positive_distribution_params = {
            'low': 0.1,
            'high': 5.0,
        }
        self.round_to_int = False
        self.use_single_precision = True

    def generate_non_singular_unit_lower_triangular(self, size, well_conditioned: bool = True):
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
                        self.positive_distribution_params['low'],
                        self.distribution_params['high']
                    )

        # Ensure non-singularity by making sure all diagonal elements are not close to zero.
        for i in range(size):
            matrix[i, i] = 1.0

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
                        self.positive_distribution_params['low'],
                        self.positive_distribution_params['high']
                    )

        # Ensure non-singularity by making sure all diagonal elements are not close to zero.
        for i in range(size):
            if abs(matrix[i, i]) < 1e-10:
                matrix[i, i] = np.random.uniform(
                    self.positive_distribution_params['low'] + 1.0,
                    self.positive_distribution_params['high']
                )
        
        if self.round_to_int:
            matrix = np.round(matrix).astype(int)
            
        return matrix

    def generate_non_singular_square(self, size, well_conditioned: bool = True):

        # Generate a non-singular matrix by multiplying a lower triangular matrix by an 
        # upper triangular matrix.
        lower = self.generate_non_singular_unit_lower_triangular(size, well_conditioned)
        upper = self.generate_non_singular_upper_triangular(size, well_conditioned)
        matrix = np.matmul(lower, upper)

        # Add diagonal dominance for well-conditioned matrices.
        if well_conditioned:
            for i in range(size):
                matrix[i, i] += 0.1

        if self.use_single_precision:
            matrix = matrix.astype(np.float32)
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

    def generate_diagonal_increasing_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, i] = i + 1
        return matrix

    def generate_diagonal_decreasing_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, i] = size - 1
        return matrix
    
    def generate_anti_diagonal_increasing_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, size - i - 1] = size - i
        return matrix

    def generate_anti_diagonal_decreasing_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, size - i - 1] = i + 1
        return matrix

    def generate_x_pattern_diagonally_dominant_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, i] = i + 1
            matrix[i, size - i - 1] = 0.5 * i + 1
        return matrix

    def generate_x_pattern_anti_diagonally_dominant_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, i] = size - i
            matrix[i, size - i - 1] = 4 * (size - i)
        return matrix

    def generate_unit_lower_triangular_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1):
                matrix[i, j] = np.random.uniform(
                    self.distribution_params['low'] + 0.5,
                    self.distribution_params['high'] + 0.5
                )

        # Ensure non-singularity by making sure all diagonal elements are not close to zero.
        for i in range(size):
            matrix[i, i] = 1.0

        return matrix

    def generate_positive_lower_triangular_matrix(self, size, diagonal_min=1.1, diagonal_max=2.0, off_diagonal_min=1.1, off_diagonal_max=10.0):
        matrix = np.zeros((size, size))
        
        for i in range(size):
            for j in range(i + 1):
                if i == j:
                    matrix[i, j] = np.random.uniform(diagonal_min, diagonal_max)
                else:
                    matrix[i, j] = np.random.uniform(off_diagonal_min, off_diagonal_max)
        
        if self.round_to_int:
            matrix = np.round(matrix).astype(int)
            
        return matrix

    def generate_tridiagonal_matrix(self, size, diagonally_dominant: bool = False):
        matrix = np.zeros((size, size))
        for i in range(size):
            matrix[i, i] = np.random.uniform(
                self.distribution_params['low'],
                self.distribution_params['high']
            )
            if diagonally_dominant:
                matrix[i, i] += (self.distribution_params['high'] - self.distribution_params['low'])
            if i > 0:
                matrix[i, i - 1] = np.random.uniform(
                    self.distribution_params['low'],
                    self.distribution_params['high']
                )
            if i < size - 1:
                matrix[i, i + 1] = np.random.uniform(
                    self.distribution_params['low'],
                    self.distribution_params['high']
                )
        
        if self.round_to_int:
            matrix = np.round(matrix).astype(int)
            
        return matrix

    def generate_special_ones_case_matrix(self, size):
        matrix = np.zeros((size, size))
        
        # Fill in the values in the following structure.
        # [[ 1,  0,  0,  1],
        #  [-1,  1,  0,  1], 
        #  [-1, -1,  1,  1],
        #  [-1, -1, -1,  1]]
        for i in range(size):
            for j in range(size):
                if i == j:
                    matrix[i, j] = 1
                elif j == size - 1:
                    matrix[i, j] = 1
                elif i > j:
                    matrix[i, j] = -1
        
        return matrix

    def generate_lower_lower_product_matrix(self, size):
        lower_triangular = self.generate_lower_triangular_matrix(size)
        lower_triangular_product = np.matmul(lower_triangular, lower_triangular.T)
        return lower_triangular_product

    def generate_lower_triangular_matrix(self, size):
        matrix = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1):
                if i == j:
                    matrix[i, j] = np.random.uniform(
                        self.positive_distribution_params['low'],
                        self.positive_distribution_params['high']
                    )
                else:
                    matrix[i, j] = np.random.uniform(
                        self.distribution_params['low'],
                        self.distribution_params['high']
                    )
        return matrix

