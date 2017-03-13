from random import randrange
from collections import defaultdict
from perceptron import Perceptron
from drawer import display_result

#  Plain
LEFT = -50
RIGHT = 50
BOTTOM = -50
TOP = 50
H_STEP = 1
V_STEP = 1

CLASSES_COUNT = 4
VECTOR_SIZE = 2
TRAIN_DATA = [
    ([RIGHT, TOP], 0),
    ([LEFT, TOP], 1),
    ([LEFT, BOTTOM], 2),
    ([RIGHT, BOTTOM], 3)
]

TEST_COUNT = 40

def get_test_vectors(count):
    result = []
    for _ in range(count):
        x = randrange(LEFT, RIGHT, H_STEP)
        y = randrange(BOTTOM, TOP, V_STEP)
        result.append([x, y])
    return result


def print_weights(weights):
    print('Weights after training:')
    for function_weights in weights:
        print(*(map(int, function_weights)), sep=', ')


if __name__ == '__main__':
    perceptron = Perceptron(CLASSES_COUNT, VECTOR_SIZE)
    perceptron.train(TRAIN_DATA)
    result = defaultdict(list)
    for v in TRAIN_DATA:
        vector = v[0]
        result[perceptron.get_class(vector)].append(vector)
    test_vectors = get_test_vectors(TEST_COUNT)
    for test_vector in test_vectors:
        result[perceptron.get_class(test_vector)].append(test_vector)
    display_result(result)
    print_weights(perceptron.functions_weights)
