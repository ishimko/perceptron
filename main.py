from collections import defaultdict
from perceptron import Perceptron
from drawer import display_result

#  Plain
LEFT = -5
RIGHT = 5
BOTTOM = -5
TOP = 5

CLASSES_COUNT = 4
VECTOR_SIZE = 2
TRAIN_DATA = [
    ([RIGHT, TOP], 0),
    ([LEFT, TOP], 1),
    ([LEFT, BOTTOM], 2),
    ([RIGHT, BOTTOM], 3)
]

if __name__ == '__main__':
    perceptron = Perceptron(CLASSES_COUNT, VECTOR_SIZE)
    perceptron.train(TRAIN_DATA)
    result = defaultdict(list)
    for v in TRAIN_DATA:
        vector = v[0]
        result[perceptron.get_class(vector)].append(vector)
    display_result(result)
