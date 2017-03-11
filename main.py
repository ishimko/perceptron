from perceptron import Perceptron


CLASSES_COUNT = 3
VECTOR_SIZE = 2
TRAIN_DATA = [
    ([0, 0], 0),
    ([1, 1], 1),
    ([-1, 1], 2)
]

if __name__ == '__main__':
    perceptron = Perceptron(CLASSES_COUNT, VECTOR_SIZE)
    perceptron.train(TRAIN_DATA)
