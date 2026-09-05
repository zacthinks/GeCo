"""First-class classifier configurations, fitting, tuning helpers, and scoring for GeCo."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Literal, Protocol

import numpy as np
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from geometric_coder.geometry.base import Matrix

ClassifierAlgorithm = Literal[
    "logistic_l2",
    "logistic_l1",
    "logistic_elasticnet",
    "complement_nb",
    "linear_svm",
    "decision_tree",
    "knn",
]

SUPPORTED_CLASSIFIER_ALGORITHMS: tuple[ClassifierAlgorithm, ...] = (
    "logistic_l2",
    "logistic_l1",
    "logistic_elasticnet",
    "complement_nb",
    "linear_svm",
    "decision_tree",
    "knn",
)



def _sklearn_uses_l1_ratio_penalty_api() -> bool:
    """Return whether installed scikit-learn uses the post-1.8 penalty API."""
    parts: list[int] = []
    for piece in sklearn_version.split(".")[:2]:
        digits = "".join(character for character in piece if character.isdigit())
        parts.append(int(digits or 0))
    return tuple(parts) >= (1, 8)


_SKLEARN_NEW_LOGISTIC_API = _sklearn_uses_l1_ratio_penalty_api()

CLASSIFIER_ALGORITHM_LABELS: dict[str, str] = {
    "logistic_l2": "Logistic regression · L2",
    "logistic_l1": "Logistic regression · L1",
    "logistic_elasticnet": "Logistic regression · elastic net",
    "complement_nb": "Complement Naive Bayes",
    "linear_svm": "Linear SVM",
    "decision_tree": "Decision tree",
    "knn": "k-nearest neighbors",
}


class ProbabilisticEstimator(Protocol):
    """Minimal persisted-estimator interface used by current GeCo classifiers."""

    def predict(self, matrix: Matrix) -> np.ndarray: ...

    def predict_proba(self, matrix: Matrix) -> np.ndarray: ...

    def decision_function(self, matrix: Matrix) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class ClassifierCapabilities:
    """Stable capability metadata used by UI and training orchestration."""

    supports_sparse: bool
    supports_dense: bool
    requires_nonnegative: bool
    probability_output: bool
    decision_score_output: bool
    uncertainty_kind: str
    tunable_parameters: tuple[str, ...]


_CLASSIFIER_CAPABILITIES: dict[str, ClassifierCapabilities] = {
    "logistic_l2": ClassifierCapabilities(
        True, True, False, True, True, "probability_distance", ("regularization",)
    ),
    "logistic_l1": ClassifierCapabilities(
        True, True, False, True, True, "probability_distance", ("regularization",)
    ),
    "logistic_elasticnet": ClassifierCapabilities(
        True,
        True,
        False,
        True,
        True,
        "probability_distance",
        ("regularization", "l1_ratio"),
    ),
    "complement_nb": ClassifierCapabilities(
        True, True, True, True, False, "probability_distance", ("alpha",)
    ),
    "linear_svm": ClassifierCapabilities(
        True, True, False, True, True, "probability_distance", ("regularization",)
    ),
    "decision_tree": ClassifierCapabilities(
        True,
        True,
        False,
        True,
        False,
        "probability_distance",
        ("max_depth", "min_samples_leaf"),
    ),
    "knn": ClassifierCapabilities(
        True,
        True,
        False,
        True,
        False,
        "probability_distance",
        ("n_neighbors", "weights"),
    ),
}


def classifier_capabilities(algorithm: str) -> ClassifierCapabilities:
    """Return explicit capabilities for one supported classifier family."""
    try:
        return _CLASSIFIER_CAPABILITIES[str(algorithm)]
    except KeyError as error:
        raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}") from error


@dataclass(frozen=True, slots=True)
class ClassifierOutputs:
    """Parallel prediction outputs for a fitted classifier."""

    predicted_labels: np.ndarray
    probabilities: np.ndarray | None
    decision_scores: np.ndarray | None
    uncertainty: np.ndarray
    score_kind: str


@dataclass(frozen=True, slots=True)
class FittedClassifier:
    """One fitted estimator and its outputs on a requested observation matrix."""

    estimator: Any
    outputs: ClassifierOutputs


def classifier_algorithm_label(algorithm: str) -> str:
    """Return the stable user-facing label for one classifier family."""
    try:
        return CLASSIFIER_ALGORITHM_LABELS[str(algorithm)]
    except KeyError as error:
        raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}") from error


def default_hyperparameters(algorithm: str) -> dict[str, Any]:
    """Return stable defaults for one supported classifier algorithm."""
    common_logistic = {
        "regularization": 1.0,
        "class_weight": "balanced",
        "max_iter": 2_000,
        "random_state": 0,
        "threshold": 0.5,
    }
    if algorithm == "logistic_l2":
        return dict(common_logistic)
    if algorithm == "logistic_l1":
        return dict(common_logistic)
    if algorithm == "logistic_elasticnet":
        return {**common_logistic, "l1_ratio": 0.5}
    if algorithm == "complement_nb":
        return {"alpha": 1.0, "fit_prior": True, "threshold": 0.5}
    if algorithm == "linear_svm":
        return {
            "regularization": 1.0,
            "class_weight": "balanced",
            "probability": True,
            "random_state": 0,
            "threshold": 0.5,
        }
    if algorithm == "decision_tree":
        return {
            "max_depth": 4,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
            "random_state": 0,
            "threshold": 0.5,
        }
    if algorithm == "knn":
        return {
            "n_neighbors": 1,
            "weights": "distance",
            "metric": "cosine",
            "threshold": 0.5,
        }
    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def normalize_hyperparameters(
    algorithm: str, hyperparameters: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge and validate user hyperparameters for a supported algorithm."""
    values = default_hyperparameters(algorithm)
    values.update(dict(hyperparameters or {}))

    threshold = float(values.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in the closed interval [0, 1]")
    values["threshold"] = threshold

    if algorithm in {
        "logistic_l2",
        "logistic_l1",
        "logistic_elasticnet",
        "linear_svm",
    }:
        regularization = float(values["regularization"])
        if regularization <= 0:
            raise ValueError("regularization lambda must be greater than zero")
        values["regularization"] = regularization
        values["random_state"] = int(values.get("random_state", 0))
        class_weight = values.get("class_weight", "balanced")
        if class_weight not in {"balanced", None}:
            raise ValueError("class_weight must be 'balanced' or null")
        values["class_weight"] = class_weight
        if algorithm.startswith("logistic_"):
            values["max_iter"] = int(values.get("max_iter", 2_000))
            if values["max_iter"] <= 0:
                raise ValueError("max_iter must be positive")
        if algorithm == "logistic_elasticnet":
            ratio = float(values.get("l1_ratio", 0.5))
            if not 0.0 <= ratio <= 1.0:
                raise ValueError("l1_ratio must lie in the closed interval [0, 1]")
            values["l1_ratio"] = ratio
        if algorithm == "linear_svm":
            values["probability"] = bool(values.get("probability", True))
        return values

    if algorithm == "complement_nb":
        alpha = float(values.get("alpha", 1.0))
        if alpha < 0:
            raise ValueError("alpha must be nonnegative")
        values["alpha"] = alpha
        values["fit_prior"] = bool(values.get("fit_prior", True))
        return values

    if algorithm == "decision_tree":
        depth = values.get("max_depth", 4)
        if depth is not None:
            depth = int(depth)
            if depth <= 0:
                raise ValueError("max_depth must be positive or null")
        leaf = int(values.get("min_samples_leaf", 1))
        if leaf <= 0:
            raise ValueError("min_samples_leaf must be positive")
        class_weight = values.get("class_weight", "balanced")
        if class_weight not in {"balanced", None}:
            raise ValueError("class_weight must be 'balanced' or null")
        values.update(
            {
                "max_depth": depth,
                "min_samples_leaf": leaf,
                "class_weight": class_weight,
                "random_state": int(values.get("random_state", 0)),
            }
        )
        return values

    if algorithm == "knn":
        neighbors = int(values.get("n_neighbors", 1))
        if neighbors <= 0:
            raise ValueError("n_neighbors must be positive")
        weights = str(values.get("weights", "distance"))
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'")
        metric = str(values.get("metric", "cosine"))
        if metric not in {"cosine", "euclidean", "manhattan"}:
            raise ValueError("metric must be cosine, euclidean, or manhattan")
        values.update(
            {"n_neighbors": neighbors, "weights": weights, "metric": metric}
        )
        return values

    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def effective_hyperparameters(
    algorithm: str,
    hyperparameters: dict[str, Any] | None,
    *,
    training_rows: int,
) -> dict[str, Any]:
    """Return normalized parameters adjusted only for hard sample-size constraints."""
    values = normalize_hyperparameters(algorithm, hyperparameters)
    if algorithm == "knn":
        values["n_neighbors"] = min(int(values["n_neighbors"]), max(1, int(training_rows)))
    return values


def tuning_candidates(
    algorithm: str,
    hyperparameters: dict[str, Any] | None,
    *,
    max_neighbors: int | None = None,
) -> list[dict[str, Any]]:
    """Return a compact, family-specific CV search grid.

    The grids intentionally remain small. Develop is an interactive exploratory
    environment, not an AutoML system; Train should improve the current fit without
    turning each recommendation into an expensive model-selection job.
    """
    base = normalize_hyperparameters(algorithm, hyperparameters)
    regularization_grid = [10.0**power for power in (-3, -2, -1, 0, 1, 2, 3)]

    if algorithm in {"logistic_l2", "logistic_l1", "linear_svm"}:
        return [
            normalize_hyperparameters(
                algorithm, {**base, "regularization": float(value)}
            )
            for value in regularization_grid
        ]
    if algorithm == "logistic_elasticnet":
        return [
            normalize_hyperparameters(
                algorithm,
                {
                    **base,
                    "regularization": float(value),
                    "l1_ratio": float(ratio),
                },
            )
            for value, ratio in product(regularization_grid, (0.25, 0.5, 0.75))
        ]
    if algorithm == "complement_nb":
        return [
            normalize_hyperparameters(algorithm, {**base, "alpha": float(alpha)})
            for alpha in (0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0)
        ]
    if algorithm == "decision_tree":
        return [
            normalize_hyperparameters(
                algorithm,
                {**base, "max_depth": depth, "min_samples_leaf": leaf},
            )
            for depth, leaf in product((2, 3, 4, 6, None), (1, 2, 4))
        ]
    if algorithm == "knn":
        maximum = max(1, int(max_neighbors or 1))
        neighbor_values = [value for value in (1, 3, 5, 7, 11) if value <= maximum]
        if maximum not in neighbor_values and maximum <= 11:
            neighbor_values.append(maximum)
        neighbor_values = sorted(set(neighbor_values))
        return [
            normalize_hyperparameters(
                algorithm,
                {**base, "n_neighbors": neighbors, "weights": weights},
            )
            for neighbors, weights in product(neighbor_values, ("uniform", "distance"))
        ]
    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def hyperparameter_summary(algorithm: str, hyperparameters: dict[str, Any] | None) -> str:
    """Return compact user-facing detail for the selected family parameters."""
    values = normalize_hyperparameters(algorithm, hyperparameters)
    if algorithm in {"logistic_l2", "logistic_l1", "linear_svm"}:
        return f"λ={float(values['regularization']):g}"
    if algorithm == "logistic_elasticnet":
        return (
            f"λ={float(values['regularization']):g} · "
            f"L1 ratio={float(values['l1_ratio']):g}"
        )
    if algorithm == "complement_nb":
        return f"α={float(values['alpha']):g}"
    if algorithm == "decision_tree":
        depth = values["max_depth"]
        return f"depth={'unlimited' if depth is None else depth} · leaf≥{values['min_samples_leaf']}"
    if algorithm == "knn":
        return f"k={values['n_neighbors']} · {values['weights']} · {values['metric']}"
    return ""


def _require_nonnegative(matrix: Matrix, *, algorithm: str) -> None:
    if sparse.issparse(matrix):
        if matrix.data.size and float(np.nanmin(matrix.data)) < 0.0:
            raise ValueError(
                f"{classifier_algorithm_label(algorithm)} requires a nonnegative geometry."
            )
        return
    values = np.asarray(matrix)
    if values.size and float(np.nanmin(values)) < 0.0:
        raise ValueError(
            f"{classifier_algorithm_label(algorithm)} requires a nonnegative geometry."
        )


def _build_estimator(
    *, algorithm: str, parameters: dict[str, Any], training_rows: int
) -> Any:
    if algorithm in {"logistic_l2", "logistic_l1", "logistic_elasticnet"}:
        if algorithm == "logistic_l2":
            solver = "liblinear"
            l1_ratio = 0.0
            legacy_penalty = "l2"
        elif algorithm == "logistic_l1":
            solver = "liblinear"
            l1_ratio = 1.0
            legacy_penalty = "l1"
        else:
            solver = "saga"
            l1_ratio = float(parameters["l1_ratio"])
            legacy_penalty = "elasticnet"
        kwargs: dict[str, Any] = {
            "C": 1.0 / float(parameters["regularization"]),
            "solver": solver,
            "class_weight": parameters["class_weight"],
            "max_iter": int(parameters["max_iter"]),
            "random_state": int(parameters["random_state"]),
        }
        if _SKLEARN_NEW_LOGISTIC_API:
            kwargs["l1_ratio"] = l1_ratio
        else:
            kwargs["penalty"] = legacy_penalty
            if algorithm == "logistic_elasticnet":
                kwargs["l1_ratio"] = l1_ratio
        return LogisticRegression(**kwargs)
    if algorithm == "complement_nb":
        return ComplementNB(
            alpha=float(parameters["alpha"]),
            fit_prior=bool(parameters["fit_prior"]),
        )
    if algorithm == "linear_svm":
        return SVC(
            C=1.0 / float(parameters["regularization"]),
            kernel="linear",
            class_weight=parameters["class_weight"],
            probability=bool(parameters["probability"]),
            random_state=int(parameters["random_state"]),
        )
    if algorithm == "decision_tree":
        return DecisionTreeClassifier(
            max_depth=parameters["max_depth"],
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            class_weight=parameters["class_weight"],
            random_state=int(parameters["random_state"]),
        )
    if algorithm == "knn":
        return KNeighborsClassifier(
            n_neighbors=min(int(parameters["n_neighbors"]), max(1, int(training_rows))),
            weights=str(parameters["weights"]),
            metric=str(parameters["metric"]),
            algorithm="brute",
        )
    raise ValueError(f"Unsupported classifier algorithm: {algorithm!r}")


def fit_classifier(
    *,
    algorithm: str,
    hyperparameters: dict[str, Any] | None,
    training_matrix: Matrix,
    labels: list[int] | np.ndarray,
    scoring_matrix: Matrix,
) -> FittedClassifier:
    """Fit one supported classifier and score the supplied observation matrix."""
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1 or training_matrix.shape[0] != y.size:
        raise ValueError("training_matrix and labels must contain parallel rows")
    if y.size < 2 or set(y.tolist()) != {0, 1}:
        raise ValueError("Classifiers require at least one positive and one negative label")

    parameters = effective_hyperparameters(
        algorithm, hyperparameters, training_rows=int(y.size)
    )
    if classifier_capabilities(algorithm).requires_nonnegative:
        _require_nonnegative(training_matrix, algorithm=algorithm)
        _require_nonnegative(scoring_matrix, algorithm=algorithm)

    estimator = _build_estimator(
        algorithm=algorithm, parameters=parameters, training_rows=int(y.size)
    )
    estimator.fit(training_matrix, y)
    return FittedClassifier(
        estimator=estimator,
        outputs=score_classifier(
            estimator=estimator,
            algorithm=algorithm,
            hyperparameters=parameters,
            matrix=scoring_matrix,
        ),
    )


def score_classifier(
    *,
    estimator: Any,
    algorithm: str,
    hyperparameters: dict[str, Any] | None,
    matrix: Matrix,
) -> ClassifierOutputs:
    """Produce explicit labels, probabilities, decision scores, and uncertainty."""
    parameters = normalize_hyperparameters(algorithm, hyperparameters)
    if classifier_capabilities(algorithm).requires_nonnegative:
        _require_nonnegative(matrix, algorithm=algorithm)

    probabilities: np.ndarray | None = None
    if hasattr(estimator, "predict_proba"):
        raw_probabilities = np.asarray(estimator.predict_proba(matrix), dtype=float)
        classes = np.asarray(getattr(estimator, "classes_", [0, 1]))
        positive_positions = np.where(classes == 1)[0]
        if raw_probabilities.ndim == 2 and positive_positions.size == 1:
            probabilities = np.asarray(
                raw_probabilities[:, int(positive_positions[0])], dtype=float
            )

    decision_scores: np.ndarray | None = None
    if hasattr(estimator, "decision_function"):
        decision_scores = np.asarray(estimator.decision_function(matrix), dtype=float)
        if decision_scores.ndim > 1 and decision_scores.shape[1] == 1:
            decision_scores = decision_scores[:, 0]

    if probabilities is not None:
        threshold = float(parameters["threshold"])
        labels = np.where(probabilities >= threshold, 1, 0).astype(int)
        uncertainty = 1.0 - np.minimum(1.0, np.abs(probabilities - 0.5) * 2.0)
        score_kind = "probability"
    else:
        labels = np.asarray(estimator.predict(matrix), dtype=int)
        if decision_scores is None:
            uncertainty = np.full(labels.shape, 0.5, dtype=float)
            score_kind = "label"
        else:
            uncertainty = 1.0 / (1.0 + np.abs(np.asarray(decision_scores, dtype=float)))
            score_kind = "decision_score"

    return ClassifierOutputs(
        predicted_labels=np.asarray(labels, dtype=int),
        probabilities=(
            np.asarray(probabilities, dtype=float) if probabilities is not None else None
        ),
        decision_scores=(
            np.asarray(decision_scores, dtype=float) if decision_scores is not None else None
        ),
        uncertainty=np.asarray(uncertainty, dtype=float),
        score_kind=score_kind,
    )


def primary_scores(outputs: ClassifierOutputs) -> np.ndarray:
    """Return probability-like scores used for recommendation and geometry views."""
    if outputs.probabilities is not None:
        return np.asarray(outputs.probabilities, dtype=float)
    if outputs.decision_scores is None:
        raise ValueError("Classifier exposes neither probabilities nor decision scores.")
    values = np.asarray(outputs.decision_scores, dtype=float)
    if values.size == 0:
        return values
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    if upper <= lower:
        return np.full(values.shape, 0.5, dtype=float)
    return (values - lower) / (upper - lower)
