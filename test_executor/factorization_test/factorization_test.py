

class FactorizationTest:
    def __init__(self):
        pass

    def run(self, input_data):
        result_data = {}
        result_data["size"] = input_data["size"]
        result_data["matrices"] = []
        result_data["solutions"] = []
        b = input_data["b"]
        result_data["b"] = b

        # Factor each input matrix.
        # Each matrix contains a variable determining the type of factorization to perform.
        for matrix in input_data["matrices"]:
            matrix.print()
            matrix.factorize()
            matrix.print()
            result_data["matrices"].append(matrix)
            x = matrix.solve(b)
            result_data["solutions"].append(x)

        print(result_data["solutions"])
        return result_data
