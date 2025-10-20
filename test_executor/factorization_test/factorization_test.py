

class FactorizationTest:
    def __init__(self):
        pass

    def run(self, input_data):
        result_data = {}
        result_data["size"] = input_data["size"]
        result_data["matrices"] = []

        # Factor each input matrix.
        # Each matrix contains a variable determining the type of factorization to perform.
        for matrix in input_data["matrices"]:
            matrix.print()
            matrix.factorize()
            matrix.print()
            result_data["matrices"].append(matrix)

        return result_data
