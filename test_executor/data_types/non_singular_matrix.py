from enum import Enum
import numpy as np


class FactorizationType(Enum):
    NO_PIVOTING = "no_pivoting"
    PARTIAL_PIVOTING = "partial_pivoting"
    FULL_PIVOTING = "full_pivoting"


class NonSingularMatrix:
    def __init__(self, np_matrix, factorization_type: FactorizationType):
        self.np_matrix = np_matrix
        self.size = np_matrix.shape[0]
        self.factorization_type = factorization_type
        self.is_factored = False

    def factorize(self):
        self.np_matrix_copy = self.np_matrix.copy()
        if self.factorization_type == FactorizationType.NO_PIVOTING:
            self.perform_lu_factorization()
        if self.factorization_type == FactorizationType.PARTIAL_PIVOTING:
            self.perform_lu_factorization_with_partial_pivoting()
        if self.factorization_type == FactorizationType.FULL_PIVOTING:
            self.perform_lu_factorization_with_full_pivoting()
        self.is_factored = True

    def solve(self, b):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")
        if self.factorization_type == FactorizationType.NO_PIVOTING:
            return self.solve_without_pivoting(b)
        if self.factorization_type == FactorizationType.PARTIAL_PIVOTING:
            return self.solve_with_partial_pivoting(b)
        if self.factorization_type == FactorizationType.FULL_PIVOTING:
            return self.solve_with_full_pivoting(b)

    def solve_without_pivoting(self, b):

        # Solve Ly = b using forward substitution.
        y = np.zeros(self.size)
        for i in range(self.size):
            y[i] = b[i]

            # Subtract the contributions from previous rows.
            for j in range(i):
                y[i] -= self.np_matrix[i, j] * y[j]
        
        # Solve Ux = y using back substitution.
        x = np.zeros(self.size)
        for i in range(self.size - 1, -1, -1):
            x[i] = y[i]

            # Subtract the contributions from subsequent rows.
            for j in range(i + 1, self.size):
                x[i] -= self.np_matrix[i, j] * x[j]

            # Divide by the diagonal element.
            x[i] /= self.np_matrix[i, i]

        return x

    def solve_with_partial_pivoting(self, b):

        # Permute the right-hand side vector b using the row permutation vector.
        permuted_b = np.zeros(self.size)
        for i in range(self.size):
            permuted_b[i] = b[self.row_permutation_vector[i]]

        return self.solve_without_pivoting(permuted_b)
    
    def solve_with_full_pivoting(self, b):

        # Permute the right-hand side vector b using the row permutation vector.
        permuted_b = np.zeros(self.size)
        for i in range(self.size):
            permuted_b[i] = b[self.row_permutation_vector[i]]
        
        # Solve LUy = permuted_b to get y = Qx.
        y = self.solve_without_pivoting(permuted_b)
        
        # Permute y using the column permutation vector to get x.
        # The column permutation vector tells us which original column is in each position
        # To get x from y = Qx, we need x = Q^T y.
        x = np.zeros(self.size)
        for i in range(self.size):
            x[self.column_permutation_vector[i]] = y[i]

        return x

    def perform_lu_factorization(self):

        # Perform Gaussian elimination column by column.
        for k in range(self.size - 1):
            self._perform_gaussian_elimination_step(k)

    def perform_lu_factorization_with_partial_pivoting(self):

        # Initialize the row permutation vector as [0, 1, 2, ..., n-1].
        self.row_permutation_vector = np.arange(self.size)
        
        # Perform Gaussian elimination with partial pivoting column by column.
        for k in range(self.size - 1):
            
            # Find the largest magnitude element in column k from row k onwards.
            max_row = k
            max_value = abs(self.np_matrix[k, k])
            for i in range(k + 1, self.size):
                if abs(self.np_matrix[i, k]) > max_value:
                    max_value = abs(self.np_matrix[i, k])
                    max_row = i
            
            # Swap rows to bring the largest element to position (k, k) if needed.
            if max_row != k:
                self.np_matrix[[k, max_row], :] = self.np_matrix[[max_row, k], :]

                # Update the permutation vector by swapping entries.
                self.row_permutation_vector[k], self.row_permutation_vector[max_row] = \
                    self.row_permutation_vector[max_row], self.row_permutation_vector[k]

            # Perform the elimination step.
            self._perform_gaussian_elimination_step(k)
        
    def perform_lu_factorization_with_full_pivoting(self):

        # Initialize the permutation vectors as [0, 1, 2, ..., n-1].
        self.row_permutation_vector = np.arange(self.size)
        self.column_permutation_vector = np.arange(self.size)
        
        # Perform Gaussian elimination with full pivoting column by column.
        for k in range(self.size - 1):
            
            # Find the largest magnitude element in the active submatrix (from row k onwards, column k onwards).
            max_row = k
            max_col = k
            max_value = abs(self.np_matrix[k, k])
            for i in range(k, self.size):
                for j in range(k, self.size):
                    if abs(self.np_matrix[i, j]) > max_value:
                        max_value = abs(self.np_matrix[i, j])
                        max_row = i
                        max_col = j
            
            # Swap rows to bring the largest element to position (k, k) if needed.
            if max_row != k:
                self.np_matrix[[k, max_row], :] = self.np_matrix[[max_row, k], :]

                # Update the row permutation vector by swapping entries.
                self.row_permutation_vector[k], self.row_permutation_vector[max_row] = \
                    self.row_permutation_vector[max_row], self.row_permutation_vector[k]
            
            # Swap columns to bring the largest element to position (k, k) if needed.
            if max_col != k:
                self.np_matrix[:, [k, max_col]] = self.np_matrix[:, [max_col, k]]

                # Update the column permutation vector by swapping entries.
                self.column_permutation_vector[k], self.column_permutation_vector[max_col] = \
                    self.column_permutation_vector[max_col], self.column_permutation_vector[k]

            # Perform the elimination step.
            self._perform_gaussian_elimination_step(k)

    def _perform_gaussian_elimination_step(self, k):

        # Check for a zero pivot.
        if abs(self.np_matrix[k, k]) < 1e-15:
            raise ValueError(f"Zero pivot encountered at position ({k}, {k}). Matrix may be singular.")
        
        # Eliminate elements below the diagonal in column k.
        for i in range(k + 1, self.size):
            
            # Compute the multiplier l_ik = a_ik / a_kk.
            multiplier = self.np_matrix[i, k] / self.np_matrix[k, k]
            
            # Store the multiplier in the lower triangular part.
            self.np_matrix[i, k] = multiplier
            
            # Eliminate a_ik and update the rest of row i.
            for j in range(k + 1, self.size):
                self.np_matrix[i, j] -= multiplier * self.np_matrix[k, j]

    def print(self):
        print(f"Non-singular matrix size {self.size}:")
        print(f"  Factorization Type: {self.factorization_type}")
        print(f"  Is Factored: {self.is_factored}")
        print(f"  Values:") 
        print(self.np_matrix)
