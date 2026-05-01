import numpy as np

from hw5code import DecisionTree


def test_decision_tree_predicts_simple_real_and_categorical_data():
    real_X = np.array([[0.0], [1.0], [2.0], [3.0]])
    real_y = np.array(["left", "left", "right", "right"])

    real_tree = DecisionTree(["real"])
    real_tree.fit(real_X, real_y)

    assert real_tree.predict(real_X).tolist() == real_y.tolist()

    categorical_X = np.array(
        [["red"], ["red"], ["blue"], ["blue"], ["green"]], dtype=object
    )
    categorical_y = np.array(["yes", "yes", "no", "no", "yes"])

    categorical_tree = DecisionTree(["categorical"])
    categorical_tree.fit(categorical_X, categorical_y)

    assert categorical_tree.predict(categorical_X).tolist() == categorical_y.tolist()
