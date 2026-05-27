import numpy as np
from collections import Counter


def find_best_split(
    feature_vector, target_vector, grad_vector=None, hess_vector=None, l2=1.0
):
    feature_vector = np.asarray(feature_vector)

    sorted_idx = np.argsort(feature_vector)
    feature_sorted = feature_vector[sorted_idx]
    split_indices = np.where(feature_sorted[1:] != feature_sorted[:-1])[0] + 1

    if len(split_indices) == 0:
        best_score = (
            -np.inf
            if grad_vector is not None and hess_vector is not None
            else float("inf")
        )
        return np.array([]), np.array([]), None, best_score

    thresholds = (
        feature_sorted[split_indices - 1] + feature_sorted[split_indices]
    ) / 2.0

    if grad_vector is not None and hess_vector is not None:
        l2 = 0.0 if l2 is None else l2
        grad_sorted = np.asarray(grad_vector)[sorted_idx]
        hess_sorted = np.asarray(hess_vector)[sorted_idx]

        grad_cumsum = np.cumsum(grad_sorted)
        hess_cumsum = np.cumsum(hess_sorted)

        G_l = grad_cumsum[split_indices - 1]
        H_l = hess_cumsum[split_indices - 1]
        G_m = grad_cumsum[-1]
        H_m = hess_cumsum[-1]
        G_r = G_m - G_l
        H_r = H_m - H_l

        gains = (
            G_l**2 / (H_l + l2)
            + G_r**2 / (H_r + l2)
            - G_m**2 / (H_m + l2)
        )

        max_gain = np.max(gains)
        best_indices = np.where(gains == max_gain)[0]
        best_idx = best_indices[np.argmin(thresholds[best_indices])]

        return thresholds, gains, thresholds[best_idx], gains[best_idx]

    n = len(feature_vector)
    target_sorted = np.asarray(target_vector)[sorted_idx]

    cumsum = np.cumsum(target_sorted)
    total_sum = cumsum[-1]

    sum_left = cumsum[split_indices - 1]
    sum_right = total_sum - sum_left
    p1_left = sum_left / split_indices
    p1_right = sum_right / (n - split_indices)

    gini_l = 2 * p1_left * (1 - p1_left)
    gini_r = 2 * p1_right * (1 - p1_right)
    ginis = (split_indices / n) * gini_l + ((n - split_indices) / n) * gini_r

    min_gini = np.min(ginis)
    best_indices = np.where(ginis == min_gini)[0]
    best_idx = best_indices[np.argmin(thresholds[best_indices])]

    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]


class DecisionTree:
    """
    Простое классификационное дерево, поддерживающее:
    * real / categorical признаки
    * binary цели (метки могут быть числами или строками)
    * ограничения max_depth, min_samples_split, min_samples_leaf (как в sklearn по смыслу)

    ВНИМАНИЕ: в методе _fit_node ниже могут быть намеренно оставлены некоторые ошибки.
    Их нужно исправить в рамках задания.
    """

    def __init__(
        self,
        feature_types,
        max_depth=None,
        min_samples_split=None,
        min_samples_leaf=None,
        l2: float | None = 1.0,
    ):
        if np.any(
            list(map(lambda x: x != "real" and x != "categorical", feature_types))
        ):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        self._l2 = l2
        self._boosting_mode = False

    def _leaf_value(self, grad_vector, hess_vector):
        l2 = 0.0 if self._l2 is None else self._l2
        return np.sum(grad_vector) / (np.sum(hess_vector) + l2)

    def _make_terminal_node(self, sub_y, node, grad_vector=None, hess_vector=None):
        node["type"] = "terminal"
        if self._boosting_mode:
            node["value"] = self._leaf_value(grad_vector, hess_vector)
        else:
            node["class"] = Counter(sub_y).most_common(1)[0][0]

    def _encode_categorical_feature(
        self, feature_vector, sub_y, grad_vector, hess_vector
    ):
        if self._boosting_mode:
            category_values = {}
            for category in np.unique(feature_vector):
                mask = feature_vector == category
                category_values[category] = self._leaf_value(
                    grad_vector[mask], hess_vector[mask]
                )
            sorted_categories = list(
                map(
                    lambda x: x[0],
                    sorted(category_values.items(), key=lambda x: x[1]),
                )
            )
        else:
            classes = np.unique(sub_y)
            positive_class = classes[1]
            counts = Counter(feature_vector)
            positive_counts = Counter(feature_vector[sub_y == positive_class])
            ratio = {}
            for key, current_count in counts.items():
                ratio[key] = positive_counts[key] / current_count
            sorted_categories = list(
                map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1]))
            )

        categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
        return (
            np.array(list(map(lambda x: categories_map[x], feature_vector))),
            categories_map,
        )

    def _fit_node(
        self, sub_X, sub_y, node, depth=0, grad_vector=None, hess_vector=None
    ):
        if not self._boosting_mode and np.all(sub_y == sub_y[0]):
            self._make_terminal_node(sub_y, node)
            return
        if self._max_depth is not None and depth >= self._max_depth:
            self._make_terminal_node(sub_y, node, grad_vector, hess_vector)
            return
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            self._make_terminal_node(sub_y, node, grad_vector, hess_vector)
            return

        feature_best, threshold_best, score_best, split = None, None, None, None
        encoded_y = None
        if not self._boosting_mode:
            classes = np.unique(sub_y)
            positive_class = classes[1]
            encoded_y = (sub_y == positive_class).astype(int)
        min_samples_leaf = (
            1 if self._min_samples_leaf is None else self._min_samples_leaf
        )

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                feature_vector, categories_map = self._encode_categorical_feature(
                    sub_X[:, feature], sub_y, grad_vector, hess_vector
                )
            else:
                raise ValueError

            thresholds, scores, _, _ = find_best_split(
                feature_vector,
                encoded_y,
                grad_vector,
                hess_vector,
                self._l2,
            )
            if len(thresholds) == 0:
                continue

            valid_thresholds = []
            valid_scores = []
            for threshold, score in zip(thresholds, scores):
                current_split = feature_vector < threshold
                l_size = np.sum(current_split)
                r_size = len(sub_y) - l_size
                if l_size >= min_samples_leaf and r_size >= min_samples_leaf:
                    valid_thresholds.append(threshold)
                    valid_scores.append(score)

            if len(valid_thresholds) == 0:
                continue

            valid_thresholds = np.array(valid_thresholds)
            valid_scores = np.array(valid_scores)
            if self._boosting_mode:
                best_score = np.max(valid_scores)
                threshold = np.min(valid_thresholds[valid_scores == best_score])
                is_better = score_best is None or best_score > score_best
            else:
                best_score = np.min(valid_scores)
                threshold = np.min(valid_thresholds[valid_scores == best_score])
                is_better = score_best is None or best_score < score_best

            if is_better:
                feature_best = feature
                score_best = best_score
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(
                        map(
                            lambda x: x[0],
                            filter(
                                lambda x: x[1] < threshold,
                                categories_map.items(),
                            ),
                        )
                    )
                else:
                    raise ValueError

        if feature_best is None:
            self._make_terminal_node(sub_y, node, grad_vector, hess_vector)
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        if self._boosting_mode:
            self._fit_node(
                sub_X[split],
                sub_y[split],
                node["left_child"],
                depth + 1,
                grad_vector[split],
                hess_vector[split],
            )
            self._fit_node(
                sub_X[np.logical_not(split)],
                sub_y[np.logical_not(split)],
                node["right_child"],
                depth + 1,
                grad_vector[np.logical_not(split)],
                hess_vector[np.logical_not(split)],
            )
            return

        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(
            sub_X[np.logical_not(split)],
            sub_y[np.logical_not(split)],
            node["right_child"],
            depth + 1,
        )

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            if self._boosting_mode:
                return node["value"]
            return node["class"]

        feature = node["feature_split"]
        feature_type = self._feature_types[feature]

        if feature_type == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

        if feature_type == "categorical":
            if x[feature] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])

        raise ValueError("There is unknown feature type")

    def fit(self, X, y, grad_vector=None, hess_vector=None):
        if grad_vector is None and hess_vector is not None:
            grad_vector = y
        elif grad_vector is not None and hess_vector is None:
            raise ValueError("grad_vector and hess_vector should be passed together")

        self._boosting_mode = grad_vector is not None
        self._tree = {}

        if self._boosting_mode:
            grad_vector = np.asarray(grad_vector)
            hess_vector = np.asarray(hess_vector)

        self._fit_node(
            X,
            y,
            self._tree,
            depth=0,
            grad_vector=grad_vector,
            hess_vector=hess_vector,
        )

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
