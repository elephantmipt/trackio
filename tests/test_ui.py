import trackio.ui as ui


def test_update_visible_runs():
    assert ui.update_visible_runs(True, "run1", []) == ["run1"]
    assert ui.update_visible_runs(False, "run1", ["run1", "run2"]) == ["run2"]
    assert ui.update_visible_runs(True, "run2", ["run1"]) == ["run1", "run2"]
