import numpy as np
import pytest

from roadcond.metrics import (
    accuracy,
    classification_report,
    confusion_matrix,
    format_table,
    macro_f1,
    mean_std,
    precision_recall_f1,
)

# Hand-worked 3-class example:
#   class 0: P = 2/2, R = 2/3, F1 = 0.8
#   class 1: P = 1/2, R = 1/2, F1 = 0.5
#   class 2: P = 1/2, R = 1/1, F1 = 2/3
Y_TRUE = [0, 0, 0, 1, 1, 2]
Y_PRED = [0, 0, 1, 1, 2, 2]


def test_confusion_matrix_rows_are_true_columns_are_predicted():
    cm = confusion_matrix(Y_TRUE, Y_PRED, 3)
    assert cm.tolist() == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]


def test_confusion_matrix_keeps_absent_classes():
    cm = confusion_matrix([0, 0], [0, 0], 4)
    assert cm.shape == (4, 4)
    assert cm.sum() == 2


def test_precision_recall_f1_hand_example():
    precision, recall, f1 = precision_recall_f1(confusion_matrix(Y_TRUE, Y_PRED, 3))
    np.testing.assert_allclose(precision, [1.0, 0.5, 0.5])
    np.testing.assert_allclose(recall, [2 / 3, 0.5, 1.0])
    np.testing.assert_allclose(f1, [0.8, 0.5, 2 / 3])


def test_macro_f1_hand_example():
    assert macro_f1(Y_TRUE, Y_PRED, 3) == pytest.approx((0.8 + 0.5 + 2 / 3) / 3)


def test_macro_f1_perfect_predictions():
    assert macro_f1([0, 1, 2, 1], [0, 1, 2, 1], 3) == pytest.approx(1.0)


def test_macro_f1_counts_missing_class_as_zero():
    # Class 2 never occurs: its F1 is 0 and still drags the macro average down.
    assert macro_f1([0, 1, 0, 1], [0, 1, 0, 1], 3) == pytest.approx(2 / 3)


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)


def test_report_matches_parts():
    report = classification_report(Y_TRUE, Y_PRED, ["dry", "wet", "ice"])
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["macro_f1"] == pytest.approx(macro_f1(Y_TRUE, Y_PRED, 3))
    assert report["per_class"]["dry"]["support"] == 3
    assert report["per_class"]["ice"]["recall"] == pytest.approx(1.0)
    assert report["confusion_matrix"] == [[2, 1, 0], [0, 1, 1], [0, 0, 1]]


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [([0, 3], [0, 1]), ([0, -1], [0, 1]), ([0, 1], [0]), ([0.5, 1], [0, 1])],
)
def test_confusion_matrix_rejects_bad_labels(y_true, y_pred):
    with pytest.raises(ValueError):
        confusion_matrix(y_true, y_pred, 3)


def test_matches_sklearn_on_random_labels():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(0)
    y_true, y_pred = rng.integers(0, 6, 500), rng.integers(0, 6, 500)
    labels = list(range(6))
    np.testing.assert_array_equal(
        confusion_matrix(y_true, y_pred, 6),
        sklearn_metrics.confusion_matrix(y_true, y_pred, labels=labels),
    )
    expected = sklearn_metrics.f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    assert macro_f1(y_true, y_pred, 6) == pytest.approx(expected)


def test_mean_std_and_table():
    assert mean_std([0.5]) == "0.500"
    assert mean_std([0.4, 0.6]) == "0.500 ± 0.141"
    table = format_table(["model", "F1"], [["logreg", "0.5"]])
    assert table.splitlines()[0] == "| model  | F1  |"
    assert table.splitlines()[1] == "|--------|-----|"
