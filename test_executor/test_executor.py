from enum import Enum

from test_executor.data_types.non_singular_matrix import NonSingularMatrix, FactorizationType
from test_executor.matrix_generator.matrix_generator import MatrixGenerator
from test_executor.factorization_test.factorization_test import FactorizationTest
from test_executor.result_plotter.result_plotter import ResultPlotter


class TestType(Enum):
    NON_SINGULAR = "non_singular"
    DIAGONAL_INCREASING = "diagonal_increasing"
    DIAGONAL_DECREASING = "diagonal_decreasing"
    ANTI_DIAGONAL_INCREASING = "anti_diagonal_increasing"
    ANTI_DIAGONAL_DECREASING = "anti_diagonal_decreasing"
    X_PATTERN_DIAGONALLY_DOMINANT = "x_pattern_diagonally_dominant"
    X_PATTERN_ANTI_DIAGONALLY_DOMINANT = "x_pattern_anti_diagonally_dominant"
    UNIT_LOWER_TRIANGULAR = "unit_lower_triangular"
    LOWER_TRIANGULAR = "lower_triangular"
    TRIDIAGONAL = "tridiagonal"
    TRIDIAGONAL_DIAGONALLY_DOMINANT = "tridiagonal_diagonally_dominant"
    SPECIAL_ONES_CASE = "special_ones_case"
    LOWER_LOWER_PRODUCT = "lower_lower_product"


class TestExecutor:
    SIZE_RANGE = [10, 101]
    SIZE_STEP = 10
    TESTS_PER_SIZE = 5

    def __init__(self):
        self.matrix_generator = MatrixGenerator()
        self.factorization_test = FactorizationTest()
        self.result_plotter = ResultPlotter()

    def execute(self, test_type: TestType):
        results = []

        # Execute the test for each size in the range.
        for size in range(self.SIZE_RANGE[0], self.SIZE_RANGE[1], self.SIZE_STEP):
            input_data = self.get_input_data(test_type, size)
            result_data = self.factorization_test.run(input_data)
            results.append(result_data)

        # self.result_plotter.plot(results)

    # Generate the input data for the test.
    # Three copies of the matrix are added, one for each factorization type.
    def get_input_data(self, test_type: TestType, size: int):
        input_data = {}
        input_data["size"] = size
        input_data["matrices"] = []
        input_data["b"] = self.matrix_generator.generate_vector(size)

        # Generate the matrix depending on the test type.
        if test_type == TestType.NON_SINGULAR:       
            np_matrix = self.matrix_generator.generate_non_singular_square(size, well_conditioned=True)
        elif test_type == TestType.DIAGONAL_INCREASING:
            np_matrix = self.matrix_generator.generate_diagonal_increasing_matrix(size)
        elif test_type == TestType.DIAGONAL_DECREASING:
            np_matrix = self.matrix_generator.generate_diagonal_decreasing_matrix(size)
        elif test_type == TestType.ANTI_DIAGONAL_INCREASING:
            np_matrix = self.matrix_generator.generate_anti_diagonal_increasing_matrix(size)
        elif test_type == TestType.ANTI_DIAGONAL_DECREASING:
            np_matrix = self.matrix_generator.generate_anti_diagonal_decreasing_matrix(size)
        elif test_type == TestType.X_PATTERN_DIAGONALLY_DOMINANT:
            np_matrix = self.matrix_generator.generate_x_pattern_diagonally_dominant_matrix(size)
        elif test_type == TestType.X_PATTERN_ANTI_DIAGONALLY_DOMINANT:
            np_matrix = self.matrix_generator.generate_x_pattern_anti_diagonally_dominant_matrix(size)
        elif test_type == TestType.UNIT_LOWER_TRIANGULAR:
            np_matrix = self.matrix_generator.generate_unit_lower_triangular_matrix(size)
        elif test_type == TestType.LOWER_TRIANGULAR:
            np_matrix = self.matrix_generator.generate_positive_lower_triangular_matrix(size)
        elif test_type == TestType.TRIDIAGONAL:
            np_matrix = self.matrix_generator.generate_tridiagonal_matrix(size)
        elif test_type == TestType.TRIDIAGONAL_DIAGONALLY_DOMINANT:
            np_matrix = self.matrix_generator.generate_tridiagonal_matrix(size, diagonally_dominant=True)
        elif test_type == TestType.SPECIAL_ONES_CASE:
            np_matrix = self.matrix_generator.generate_special_ones_case_matrix(size)
        elif test_type == TestType.LOWER_LOWER_PRODUCT:
            np_matrix = self.matrix_generator.generate_lower_lower_product_matrix(size)
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        for factorization_type in FactorizationType:
            matrix = NonSingularMatrix(np_matrix.copy(), factorization_type)
            input_data["matrices"].append(matrix)

        return input_data
