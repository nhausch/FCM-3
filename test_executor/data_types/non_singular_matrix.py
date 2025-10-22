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
        
        # Debugging variables
        self.swap_count = 0
        self.condition_number = None
        self.swap_details = []
        self.print_swap_details = False

    def factorize(self):
        self.np_matrix_copy = self.np_matrix.copy()
        
        # Calculate condition number for debugging
        self.condition_number = np.linalg.cond(self.np_matrix)
        
        # Reset debugging variables
        self.swap_count = 0
        self.swap_details = []
        
        if self.factorization_type == FactorizationType.NO_PIVOTING:
            self.perform_lu_factorization()
        if self.factorization_type == FactorizationType.PARTIAL_PIVOTING:
            self.perform_lu_factorization_with_partial_pivoting()
        if self.factorization_type == FactorizationType.FULL_PIVOTING:
            self.perform_lu_factorization_with_full_pivoting()
        self.is_factored = True

    def remultiply(self):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")
        if self.factorization_type == FactorizationType.NO_PIVOTING:
            return self.remultiply_without_pivoting()
        if self.factorization_type == FactorizationType.PARTIAL_PIVOTING:
            return self.remultiply_with_partial_pivoting()
        if self.factorization_type == FactorizationType.FULL_PIVOTING:
            return self.remultiply_with_full_pivoting()

    def solve(self, b):
        self.b = b
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")
        if self.factorization_type == FactorizationType.NO_PIVOTING:
            self.x = self.solve_without_pivoting(b)
        if self.factorization_type == FactorizationType.PARTIAL_PIVOTING:
            self.x = self.solve_with_partial_pivoting(b)
        if self.factorization_type == FactorizationType.FULL_PIVOTING:
            self.x = self.solve_with_full_pivoting(b)
        return self.x

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
            
            # Check if swap is beneficial
            current_pivot = abs(self.np_matrix[k, k])
            potential_pivot = abs(self.np_matrix[max_row, k]) if max_row != k else current_pivot
            improvement_ratio = potential_pivot / current_pivot if current_pivot > 0 else float('inf')
            
            # Swap rows to bring the largest element to position (k, k) if needed.
            if max_row != k:
                self.swap_count += 1
                self.swap_details.append({
                    'step': k,
                    'from_row': k,
                    'to_row': max_row,
                    'current_pivot': current_pivot,
                    'new_pivot': potential_pivot,
                    'improvement_ratio': improvement_ratio
                })
                
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
            
            # Check if swaps are beneficial
            current_pivot = abs(self.np_matrix[k, k])
            potential_pivot = max_value
            improvement_ratio = potential_pivot / current_pivot if current_pivot > 0 else float('inf')
            
            # Swap rows if needed
            if max_row != k:
                self.swap_count += 1
                self.swap_details.append({
                    'step': k,
                    'type': 'row_swap',
                    'from_row': k,
                    'to_row': max_row,
                    'current_pivot': current_pivot,
                    'new_pivot': potential_pivot,
                    'improvement_ratio': improvement_ratio
                })
                
                self.np_matrix[[k, max_row], :] = self.np_matrix[[max_row, k], :]

                # Update the row permutation vector by swapping entries.
                self.row_permutation_vector[k], self.row_permutation_vector[max_row] = \
                    self.row_permutation_vector[max_row], self.row_permutation_vector[k]
            
            # Swap columns to bring the largest element to position (k, k) if needed.
            if max_col != k:
                self.swap_count += 1
                self.swap_details.append({
                    'step': k,
                    'type': 'column_swap',
                    'from_col': k,
                    'to_col': max_col,
                    'current_pivot': current_pivot,
                    'new_pivot': potential_pivot,
                    'improvement_ratio': improvement_ratio
                })
                
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

    def compute_lu(self, LU_combined, size, use_absolute_value_for_multiplication: bool = False):
        if LU_combined.ndim != 2 or LU_combined.shape[0] != LU_combined.shape[1]:
            raise ValueError("LU must be a square array.")
        if LU_combined.shape[0] != size:
            raise ValueError(f"LU size mismatch: {LU_combined.shape[0]} != {size}")
        
        # Match the output type to the input matrix.
        M = np.zeros((size, size), dtype=LU_combined.dtype)

        # Compute the result matrix.
        for i in range(size):
            for j in range(size):
                total = 0.0

                # Sum over strictly lower part of L.
                for k in range(i):
                    if k <= j:
                        if use_absolute_value_for_multiplication:
                            total += abs(LU_combined[i,k]) * abs(LU_combined[k,j])
                        else:
                            total += LU_combined[i,k] * LU_combined[k,j]

                # Add unit diagonal of L times U[i,j].
                if i <= j:
                    if use_absolute_value_for_multiplication:
                        total += abs(LU_combined[i,j])
                    else:
                        total += LU_combined[i,j]
                M[i,j] = total
        return M

    def remultiply_without_pivoting(self):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")
        self.reconstructed_np_array = self.compute_lu(self.np_matrix, self.size)

    def remultiply_with_partial_pivoting(self):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")

        LU = self.compute_lu(self.np_matrix, self.size)
        self.reconstructed_np_array = np.zeros((self.size, self.size))

        # Apply the inverse permutation to reconstruct A from PA = LU.
        for i in range(self.size):
            original_row = -1
            for k in range(self.size):
                if self.row_permutation_vector[k] == i:
                    original_row = k
                    break
            for j in range(self.size):
                self.reconstructed_np_array[i, j] = LU[original_row, j]

    def remultiply_with_full_pivoting(self):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")

        LU = self.compute_lu(self.np_matrix, self.size)
        temp_matrix = np.zeros((self.size, self.size))
        
        # Apply the inverse permutation.
        for i in range(self.size):
            original_row = -1
            for k in range(self.size):
                if self.row_permutation_vector[k] == i:
                    original_row = k
                    break
            for j in range(self.size):
                temp_matrix[i, j] = LU[original_row, j]
        
        # Now apply the inverse column permutation to get the final result.
        self.reconstructed_np_array = np.zeros((self.size, self.size))
        for i in range(self.size):
            for j in range(self.size):
                original_col = -1
                for k in range(self.size):
                    if self.column_permutation_vector[k] == j:
                        original_col = k
                        break
                self.reconstructed_np_array[i, j] = temp_matrix[i, original_col]

    def get_relative_factorization_accuracy(self):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")
        diff = self.np_matrix_copy - self.reconstructed_np_array
        return np.linalg.norm(diff) / np.linalg.norm(self.np_matrix_copy)

    def get_residual(self):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")
        diff = self.b - np.matmul(self.np_matrix_copy, self.x)
        return np.linalg.norm(diff) / np.linalg.norm(self.b)

    def get_growth_factor(self):
        if not self.is_factored:
            raise ValueError("Matrix is not factored. Please factorize the matrix first.")
        
        LU = self.compute_lu(self.np_matrix, self.size, use_absolute_value_for_multiplication=True)
        
        return np.linalg.norm(LU) / np.linalg.norm(self.np_matrix_copy)
        
        # elif self.factorization_type == FactorizationType.PARTIAL_PIVOTING:
        #     PA = np.zeros((self.size, self.size))
        #     for i in range(self.size):
        #         original_row = self.row_permutation_vector[i]
        #         PA[i, :] = self.np_matrix_copy[original_row, :]
        #     return np.linalg.norm(LU) / np.linalg.norm(PA)
        
        # elif self.factorization_type == FactorizationType.FULL_PIVOTING:
        #     # Full pivoting: compare against PAQ
        #     # row_permutation_vector[i] tells us which original row is now in position i
        #     PA = np.zeros((self.size, self.size))
        #     for i in range(self.size):
        #         original_row = self.row_permutation_vector[i]
        #         PA[i, :] = self.np_matrix_copy[original_row, :]
            
        #     # column_permutation_vector[j] tells us which original column is now in position j
        #     PAQ = np.zeros((self.size, self.size))
        #     for i in range(self.size):
        #         for j in range(self.size):
        #             original_col = self.column_permutation_vector[j]
        #             PAQ[i, j] = PA[i, original_col]
            
        #     return np.linalg.norm(LU) / np.linalg.norm(PAQ)

    def print_debug_info(self):
        """Print debugging information about swaps and condition number"""
        print(f"\n=== DEBUG INFO for {self.factorization_type.value.upper()} ===")
        print(f"Matrix size: {self.size}")
        print(f"Condition number: {self.condition_number:.2e}")
        print(f"Number of swaps: {self.swap_count}")
        
        if self.swap_details:
            # Calculate average improvement ratio
            avg_improvement = np.mean([swap['improvement_ratio'] for swap in self.swap_details])
            min_improvement = min([swap['improvement_ratio'] for swap in self.swap_details])
            max_improvement = max([swap['improvement_ratio'] for swap in self.swap_details])
            
            print(f"Average improvement ratio: {avg_improvement:.2f}")
            print(f"Min improvement ratio: {min_improvement:.2f}")
            print(f"Max improvement ratio: {max_improvement:.2f}")
            
            if self.print_swap_details:
                print("Swap details:")
                for swap in self.swap_details:
                    if 'type' in swap:  # Full pivoting
                        print(f"  Step {swap['step']}: {swap['type']} from {swap.get('from_row', swap.get('from_col'))} to {swap.get('to_row', swap.get('to_col'))}")
                        print(f"    Improvement ratio: {swap['improvement_ratio']:.2f}")
                    else:  # Partial pivoting
                        print(f"  Step {swap['step']}: Row swap from {swap['from_row']} to {swap['to_row']}")
                        print(f"    Improvement ratio: {swap['improvement_ratio']:.2f}")
        else:
            print("No swaps performed")

    def print(self):
        print(f"Non-singular matrix size {self.size}:")
        print(f"  Factorization Type: {self.factorization_type}")
        print(f"  Is Factored: {self.is_factored}")
        print(f"  Values:") 
        print(self.np_matrix)
