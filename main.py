from perceptron import Perceptron
import digit_example
import and_example
import or_example

TRAINING_ITERS_COUNT = 5

def show_example(training_data, test_data, example_name):
    print('#### {} ####'.format(example_name))
    perceptron = Perceptron(training_data, TRAINING_ITERS_COUNT)
    print('weights after training: {}'.format(perceptron.weights))
    print('Test data:')
    for test_vector in test_data:
        print('{}: {}'.format(test_vector, perceptron.get_decision(test_vector)))
    print('\n')

if __name__ == '__main__':
    show_example(digit_example.TRAINING_DATA, digit_example.TEST_DATA, digit_example.EXAMPLE_NAME)
    show_example(and_example.TRAINING_DATA, and_example.TEST_DATA, and_example.EXAMPLE_NAME)
    show_example(or_example.TRAINING_DATA, or_example.TEST_DATA, or_example.EXAMPLE_NAME)
