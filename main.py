import argparse

from test_executor.test_executor import TestExecutor, TestType


def main():
    parser = argparse.ArgumentParser(description='Execute matrix factorization tests')
    parser.add_argument('-t', '--test_type', 
                    type=str, 
                    choices=[test_type.value for test_type in TestType],
                    default=TestType.NON_SINGULAR.value,
                    help='Type of test to execute (default: non_singular)')
    args = parser.parse_args()
    test_type = TestType(args.test_type)
    test_executor = TestExecutor()
    test_executor.execute(test_type)

if __name__ == "__main__":
    main()

# Test of generic square matrix not required?
# Which matrix should the norm be computed on?
# Do we have to compare to library factorizations?
# Do we need to time the factorization?
