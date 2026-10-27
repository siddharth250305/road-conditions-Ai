import roadcond
from roadcond import load_config


def test_import():
    assert isinstance(roadcond.__version__, str)


def test_default_config_loads():
    cfg = load_config()
    assert "rscd" in cfg
    assert "rdd2022" in cfg


def test_friction_classes():
    cfg = load_config()
    assert cfg["rscd"]["classes"] == ["dry", "wet", "water", "fresh_snow", "melted_snow", "ice"]


def test_damage_classes():
    classes = load_config()["rdd2022"]["classes"]
    assert [c["id"] for c in classes] == [0, 1, 2, 3]  # YOLO needs contiguous ids
    assert len({c["code"] for c in classes}) == len(classes)
    assert len({c["name"] for c in classes}) == len(classes)
