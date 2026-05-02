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


def test_decision_tree_respects_stopping_parameters():
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([0, 0, 1, 1])

    shallow_tree = DecisionTree(["real"], max_depth=0)
    shallow_tree.fit(X, y)
    assert shallow_tree._tree["type"] == "terminal"

    split_tree = DecisionTree(["real"], max_depth=1)
    split_tree.fit(X, y)
    assert split_tree._tree["type"] == "nonterminal"
    assert split_tree._tree["left_child"]["type"] == "terminal"
    assert split_tree._tree["right_child"]["type"] == "terminal"

    min_split_tree = DecisionTree(["real"], min_samples_split=5)
    min_split_tree.fit(X, y)
    assert min_split_tree._tree["type"] == "terminal"

    min_leaf_tree = DecisionTree(["real"], min_samples_leaf=3)
    min_leaf_tree.fit(X, y)
    assert min_leaf_tree._tree["type"] == "terminal"
