import argparse

from test_executor.test_executor import TestExecutor, TestType


def main():
    parser = argparse.ArgumentParser(description='Execute matrix factorization tests')
    parser.add_argument('-t', '--test_type', 
                    type=str, 
                    choices=[test_type.value for test_type in TestType],
                    default=TestType.NON_SINGULAR.value,
                    help='Type of test to execute (default: non_singular)')
    
    parser.add_argument('-a', '--use_absolute_value_for_remultiplication',
                    action='store_true',
                    help='Use absolute value for remultiplication')
    args = parser.parse_args()
    test_type = TestType(args.test_type)

    test_executor = TestExecutor()
    test_executor.execute(test_type, args.use_absolute_value_for_remultiplication)

if __name__ == "__main__":
    main()

# Which matrix should the norm be computed on?
# Do we have to compare to library factorizations?
