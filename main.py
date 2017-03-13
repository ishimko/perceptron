from perceptron import Perceptron

TRAINING_DATA = [
    (list('111101101101111'), 0),
    (list('001001001001001'), 0),
    (list('111001111100111'), 0),
    (list('111001111001111'), 0),
    (list('101101111001001'), 0),
    (list('111100111001111'), 1),
    (list('111100111101111'), 0),
    (list('111001001001001'), 0),
    (list('111101111101111'), 0),
    (list('111101111001111'), 0)
]

TEST = [
    list('111100111000111'),
    list('111100010001111'),
    list('111100011001111'),
    list('110100111001111'),
    list('110100111001011'),
    list('111100101001111')
]

TRAINING_ITERS_COUNT = 3

if __name__ == '__main__':
    TRAINING_DATA = [(list(map(int, l)), result) for l, result in TRAINING_DATA]
    perceptron = Perceptron(TRAINING_DATA, TRAINING_ITERS_COUNT)
    TEST = [list(map(int, x)) for x in TEST]
    for x in TEST:
        print('{}'.format(perceptron.get_class(x)))
    print('weights: {}'.format(perceptron.weights))
