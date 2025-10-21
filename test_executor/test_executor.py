from enum import Enum

from test_executor.data_types.non_singular_matrix import NonSingularMatrix, FactorizationType
from test_executor.matrix_generator.matrix_generator import MatrixGenerator
from test_executor.factorization_test.factorization_test import FactorizationTest


class TestType(Enum):
    NON_SINGULAR = "non_singular"
    DIAGONAL = "diagonal"
    ANTI_DIAGONAL = "anti_diagonal"
    X_PATTERN = "x_pattern"
    UNIT_LOWER_TRIANGULAR = "unit_lower_triangular"
    LOWER_TRIANGULAR_SMALL_DIAGONAL = "lower_triangular_small_diagonal"
    TRIDIAGONAL = "tridiagonal"
    TRIDIAGONAL_DIAGONALLY_DOMINANT = "tridiagonal_diagonally_dominant"
    SPECIAL_ONES_CASE = "special_ones_case"
    LOWER_LOWER_PRODUCT = "non_singular_square"


class TestExecutor:
    SIZE_RANGE = [10, 11]
    SIZE_STEP = 10

    def __init__(self):
        self.matrix_generator = MatrixGenerator()
        self.factorization_test = FactorizationTest()

    def execute(self, test_type: TestType, use_absolute_value_for_remultiplication: bool = False):
        for size in range(self.SIZE_RANGE[0], self.SIZE_RANGE[1], self.SIZE_STEP):
            input_data = self.get_input_data(test_type, size)
            self.factorization_test.run(input_data, use_absolute_value_for_remultiplication)

    def get_input_data(self, test_type: TestType, size: int):
        input_data = {}
        input_data["size"] = size
        input_data["matrices"] = []
        input_data["b"] = self.matrix_generator.generate_vector(size)

        if test_type == TestType.NON_SINGULAR:       
            np_matrix = self.matrix_generator.generate_non_singular_square(size)

            for factorization_type in FactorizationType:
                matrix = NonSingularMatrix(np_matrix.copy(), factorization_type)
                input_data["matrices"].append(matrix)

        return input_data
