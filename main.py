from perceptron import Perceptron

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
    result = perceptron.get_class(TRAIN_DATA[2][0])
    print(result)
