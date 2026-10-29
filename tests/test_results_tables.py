import json

from roadcond.results_tables import (
    END,
    START,
    classification_table,
    detection_table,
    render,
    update_readme,
)


def _run(seed, val, test=None, ece=(0.2, 0.05)):
    run = {"seed": seed, "val": {"macro_f1": val, "accuracy": val}}
    if test is not None:
        run["test"] = {
            "macro_f1": test,
            "accuracy": test + 0.1,
            "calibration": {"ece_before": ece[0], "ece_after": ece[1]},
        }
    return run


def _write(path, models, seeds, eval_test=True):
    path.write_text(json.dumps({"seeds": seeds, "eval_test": eval_test, "models": models}))


def test_classification_table_reports_mean_std_and_marks_missing_models_pending(tmp_path):
    _write(
        tmp_path / "resnet50.json",
        {"resnet50": [_run(0, 0.70, 0.60), _run(1, 0.72, 0.64), _run(2, 0.74, 0.62)]},
        [0, 1, 2],
    )
    text = classification_table(tmp_path)
    row = next(line for line in text.splitlines() if line.startswith("| resnet50"))
    assert "0.720 ± 0.020" in row and "0.620 ± 0.020" in row and "0.200 -> 0.050" in row
    assert "over 3 seed(s)" in text
    logreg = next(line for line in text.splitlines() if line.startswith("| logreg"))
    assert "pending" in logreg


def test_a_model_without_a_test_run_says_so(tmp_path):
    _write(tmp_path / "small_cnn.json", {"small_cnn": [_run(0, 0.5)]}, [0], eval_test=False)
    row = next(line for line in classification_table(tmp_path).splitlines() if "small_cnn" in line)
    assert "not run" in row and "0.500" in row


def test_no_results_at_all_is_honest(tmp_path):
    text = render(tmp_path)
    assert "No classifier results found yet" in text
    assert "Pending: the detection experiments have not been run" in text
    assert "0." not in text.split("Cross-country")[1]  # no invented numbers


def test_detection_table_from_phase3_results(study_results):
    text = detection_table(study_results)
    norway = next(line for line in text.splitlines() if line.startswith("| Norway"))
    assert "0.800" in norway and "0.600" in norway and "25.0%" in norway  # source, zero-shot, gap
    assert "0.620" in norway  # weather augmentation (zero-shot + 0.02)
    assert "pending" not in norway.split("|")[6]  # the 10-label cell has data
    assert "pending" in norway.split("|")[8]  # no 200-label runs in the fixture
    assert "± " in norway  # spread over seeds is shown
    assert "over 3 seed(s)" in text


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{START}\nold table\n{END}\noutro\n")
    assert update_readme(readme, "NEW")
    assert readme.read_text() == f"intro\n{START}\n\nNEW\n\n{END}\noutro\n"
    assert update_readme(readme, "NEWER") and "old" not in readme.read_text()


def test_update_readme_without_markers_changes_nothing(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("no markers here\n")
    assert update_readme(readme, "x") is False
    assert readme.read_text() == "no markers here\n"
