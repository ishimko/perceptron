from perceptron import Perceptron

TRAINING_DATA = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]
TRAINING_ITERS_COUNT = 5

if __name__ == '__main__':
    perceptron = Perceptron(TRAINING_DATA, TRAINING_ITERS_COUNT)
    for x, _ in TRAINING_DATA:
        print('{} AND {} = {}'.format(x[0], x[1], perceptron.get_class(x)))
    print('weights: {}'.format(perceptron.weights))
