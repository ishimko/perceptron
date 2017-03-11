from perceptron import Perceptron

if __name__ == '__main__':
    perceptron = Perceptron(3, 2)
    train_data = [
        ([0, 0], 0),
        ([1, 1], 1),
        ([-1, 1], 2)
    ]
    perceptron.train(train_data)
