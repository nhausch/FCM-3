

class FactorizationTest:
    def __init__(self):
        pass

    def run(self, input_data):

        # Prepare the output.
        b = input_data["b"]
        result_data = {}
        result_data["size"] = input_data["size"]
        result_data["b"] = b
        result_data["relative_factorization_accuracy"] = []
        result_data["residual"] = []
        result_data["growth_factor"] = []
        result_data["condition_number"] = []
        result_data["swap_count"] = []

        # Process each input matrix and store the results.
        # Each matrix contains a variable determining the type of factorization to perform.
        for matrix in input_data["matrices"]:
            try:
                # Factor the matrix, solve for x, and remultiply the matrix.
                matrix.factorize()
                x = matrix.solve(b)
                matrix.remultiply()

                # Compute the relative accuracy of the factorization and the residual.
                relative_factorization_accuracy = matrix.get_relative_factorization_accuracy()
                residual = matrix.get_residual()
                growth_factor = matrix.get_growth_factor()
                condition_number = matrix.get_condition_number()
                swap_count = matrix.get_swap_count()
                result_data["relative_factorization_accuracy"].append(relative_factorization_accuracy)
                result_data["residual"].append(residual)
                result_data["growth_factor"].append(growth_factor)
                result_data["condition_number"].append(condition_number)
                result_data["swap_count"].append(swap_count)
                
            except ValueError as e:
                
                # Add NaN values to maintain consistent array lengths.
                result_data["relative_factorization_accuracy"].append(float('nan'))
                result_data["residual"].append(float('nan'))
                result_data["growth_factor"].append(float('nan'))
                result_data["condition_number"].append(float('nan'))
                result_data["swap_count"].append(0)

        self.print_result(result_data)
        return result_data

    def print_result(self, result_data):
        print("Size: ", result_data["size"])
        print("Relative factorization accuracy: ", result_data["relative_factorization_accuracy"])
        print("Residual: ", result_data["residual"])
        print("Growth factor: ", result_data["growth_factor"])
        print("Condition number: ", result_data["condition_number"])
        print("Swap count: ", result_data["swap_count"])
        print("--------------------------------")
