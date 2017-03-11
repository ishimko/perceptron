import matplotlib.pyplot as plt
from random import random


def display_result(classes):
    colors = [[random() for _ in range(3)] for _ in range(len(classes))]
    for class_index, vectors in classes.items():
        sample = vectors[0]
        xs = list(x[0] for x in vectors)
        ys = list(x[1] for x in vectors)
        for vector in vectors:
            plt.plot([vector[0], sample[0]], [vector[1], sample[1]], c=colors[class_index], lw=1)
        plt.scatter(xs, ys, c=colors[class_index], marker='.')
    plt.show()
