import numpy as np


class Perceptron:
    BIAS_VALUE = 1
    COEFFICIENT = 1

    def __init__(self, classes_count, vector_size):
        self.functions_weights = [np.matrix([0] * (vector_size + 1)).transpose()] * classes_count
        self.classes_count = classes_count
        self.vector_size = vector_size

    def train(self, sample):
        step = 0
        errors = True
        while errors:
            step += 1
            errors = False
            print('step {}'.format(step))
            for vector, expected_class in sample:
                assert len(vector) == self.vector_size
                vector = np.matrix(self._add_bias(vector))
                functions_results = self._get_separating_functions_results(vector)
                if not self._is_true_max(expected_class, functions_results):
                    print('bad boy')
                    errors = True
                    punishment = self.COEFFICIENT * vector.transpose()
                    self.functions_weights[expected_class] = self.functions_weights[expected_class] + punishment
                    for i in range(len(self.functions_weights)):
                        if i != expected_class:
                            if functions_results[i] >= functions_results[expected_class]:
                                self.functions_weights[i] = self.functions_weights[i] - punishment
                else:
                    print('good boy')
                print(self.functions_weights)
        print('Done!')
        print('Weights are: {}'.format(self.functions_weights))

    def get_class(self, vector):
        pass

    def save_weights(self):
        pass

    def load_weights(self):
        pass

    def _add_bias(self, vector):
        return vector + [self.BIAS_VALUE]

    def _get_separating_functions_results(self, vector):
        result = []
        for weight in self.functions_weights:
            result.append(int(vector * weight))
        return result

    def _is_true_max(self, expected_class, functions_results):
        expected_max = functions_results[expected_class]
        others = [x for i, x in enumerate(functions_results) if i != expected_class]
        return all([x < expected_max for x in others])
