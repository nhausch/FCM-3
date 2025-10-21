

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

        # Process each input matrix and store the results.
        # Each matrix contains a variable determining the type of factorization to perform.
        for matrix in input_data["matrices"]:

            # Factor the matrix, solve for x, and remultiply the matrix.
            matrix.factorize()
            x = matrix.solve(b)
            matrix.remultiply()

            # Compute the relative accuracy of the factorization and the residual.
            relative_factorization_accuracy = matrix.get_relative_factorization_accuracy()
            residual = matrix.get_residual()
            growth_factor = matrix.get_growth_factor()
            result_data["relative_factorization_accuracy"].append(relative_factorization_accuracy)
            result_data["residual"].append(residual)
            result_data["growth_factor"].append(growth_factor)
            print("Size: ", input_data["size"])
            print("Growth factor: ", growth_factor)
            print("Relative factorization accuracy: ", relative_factorization_accuracy)
            print("Residual: ", residual)
            print("--------------------------------")

        return result_data
