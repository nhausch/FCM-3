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

    def perform_lu_factorization(self):

        # Perform Gaussian elimination column by column.
        for k in range(self.size - 1):

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

    def perform_lu_factorization_with_partial_pivoting(self):

        # Initialize the permutation matrix as identity.
        self.row_permutation_np_matrix = np.eye(self.size, dtype=int)
        
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
                self.row_permutation_np_matrix[[k, max_row], :] = self.row_permutation_np_matrix[[max_row, k], :]
            
            # Check for a zero pivot.
            if abs(self.np_matrix[k, k]) < 1e-15:
                raise ValueError(f"Zero pivot encountered at position ({k}, {k}) even after pivoting. Matrix may be singular.")
            
            # Eliminate elements below the diagonal in column k.
            for i in range(k + 1, self.size):

                # Compute the multiplier l_ik = a_ik / a_kk.
                multiplier = self.np_matrix[i, k] / self.np_matrix[k, k]
                
                # Store the multiplier in the lower triangular part.
                self.np_matrix[i, k] = multiplier
                
                # Eliminate a_ik and update the rest of row i.
                for j in range(k + 1, self.size):
                    self.np_matrix[i, j] -= multiplier * self.np_matrix[k, j]
        

    def perform_lu_factorization_with_full_pivoting(self):
        pass

    def print(self):
        print(f"Non-singular matrix size {self.size}:")
        print(f"  Factorization Type: {self.factorization_type}")
        print(f"  Is Factored: {self.is_factored}")
        print(f"  Values:") 
        print(self.np_matrix)
