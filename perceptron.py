import numpy as np


class Perceptron:
    BIAS_VALUE = 1
    COEFFICIENT = 1

    def __init__(self, classes_count, vector_size):
        self.functions_weights = [np.matrix([0] * (vector_size + 1)).transpose()] * classes_count
        self.classes_count = classes_count
        self.vector_size = vector_size

    def train(self, sample):
        errors = True
        while errors:
            errors = False
            for vector, expected_class in sample:
                assert len(vector) == self.vector_size
                vector = np.matrix(self._add_bias(vector))
                functions_results = self._get_separating_functions_results(vector)
                if not self._is_true_max(expected_class, functions_results):
                    errors = True
                    punishment = self.COEFFICIENT * vector.transpose()
                    new_weights = self.functions_weights[expected_class] + punishment
                    self.functions_weights[expected_class] = new_weights
                    for i in range(len(self.functions_weights)):
                        if i != expected_class:
                            if functions_results[i] >= functions_results[expected_class]:
                                self.functions_weights[i] = self.functions_weights[i] - punishment

    def get_class(self, vector):
        vector = np.matrix(self._add_bias(vector))
        results = self._get_separating_functions_results(vector)
        return max(enumerate(results), key=lambda x: x[1])[0]

    def _add_bias(self, vector):
        return vector + [self.BIAS_VALUE]

    def _get_separating_functions_results(self, vector):
        result = []
        for weight in self.functions_weights:
            result.append(int(vector * weight))
        return result

    def _is_true_max(self, expected_class, functions_results):
        expected_max = functions_results[expected_class]
        others = (x for i, x in enumerate(functions_results) if i != expected_class)
        return all((x < expected_max for x in others))
