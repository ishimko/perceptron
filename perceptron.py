import numpy as np


class Perceptron:
    BIAS_VALUE = 1
    COEFFICIENT = 1

    def __init__(self, training_data, training_iters_count):
        self.sensors_count = len(training_data[0][0])
        self.weights = [0] * (self.sensors_count + 1)
        for _ in range(training_iters_count):
            for vector, expected_result in training_data:
                assert len(vector) == self.sensors_count
                vector = np.array(self._add_bias(vector))
                actual_result = self._get_decision_internal(vector)
                punishment = (expected_result - actual_result) * vector * self.COEFFICIENT
                self.weights = self.weights + punishment

    def get_decision(self, vector):
        assert len(vector) == self.sensors_count
        vector = np.array(self._add_bias(vector))
        return bool(self._get_decision_internal(vector))

    def _add_bias(self, vector):
        return vector + [self.BIAS_VALUE]

    def _get_decision_internal(self, vector):
        return self._unit_step(np.dot(vector, self.weights))

    @staticmethod
    def _unit_step(x):
        return 0 if x < 0 else 1
