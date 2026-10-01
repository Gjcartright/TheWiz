from quant_platform.runtime_types import strict_bool


def test_strict_bool_never_treats_serialized_false_as_true():
    for value in (False, 0, 0.0, "False", "false", "0", "no", "", None):
        assert strict_bool(value) is False

    for value in (True, 1, 1.0, "True", "true", "1", "yes", "Y"):
        assert strict_bool(value) is True
